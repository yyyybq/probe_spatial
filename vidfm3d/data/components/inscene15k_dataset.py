"""
InsScene-15K dataset loader for probe training (identity, depth, point, camera).

Supports two data sources that have identity masks + depth + camera:
  - processed_infinigen: ObjectSegmentation/*.npy, Depth/*.npy, camview/*.npz
  - processed_scannetpp_v2: refined_ins_ids/*.npy, depth/*.png, metadata.npz

Point maps are computed from depth + intrinsics + extrinsics via back-projection.
"""

import hashlib
import json
import logging
import math
import os
from glob import glob
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from safetensors.torch import load_file

from vidfm3d.data.components.video_probe_dataset import invert_pose_ref_and_scale
from vidfm3d.dust3r.datasets.base.easy_dataset import EasyDataset
from vidfm3d.utils.feature_layers import (
    default_feature_channels,
    feature_filename,
)

logger = logging.getLogger(__name__)


@torch.no_grad()
def depth_to_pointmap(depth, intrinsic, extrinsic):
    """Back-project depth map to 3D world-coordinate point map.

    Args:
        depth: (H, W) depth in meters, float32.
        intrinsic: (3, 3) camera intrinsic matrix.
        extrinsic: (3, 4) camera extrinsic [R|t] (world-to-cam).

    Returns:
        pointmap: (H, W, 3) world-coordinate 3D points.
    """
    H, W = depth.shape
    # Build pixel grid
    v, u = torch.meshgrid(
        torch.arange(H, dtype=torch.float32),
        torch.arange(W, dtype=torch.float32),
        indexing="ij",
    )
    ones = torch.ones_like(u)
    uv1 = torch.stack([u, v, ones], dim=-1)  # (H, W, 3)

    # Unproject to camera coordinates: P_cam = K^-1 * [u,v,1]^T * depth
    K_inv = torch.inverse(intrinsic.float())  # (3, 3)
    pts_cam = (uv1 @ K_inv.T) * depth.unsqueeze(-1)  # (H, W, 3)

    # Camera-to-world: P_world = R^T * (P_cam - t)
    R = extrinsic[:3, :3].float()  # (3, 3)
    t = extrinsic[:3, 3].float()   # (3,)
    pts_world = (pts_cam - t) @ R  # (H, W, 3)  equiv to R^T @ (p - t)

    return pts_world


class InsScene15KDataset(EasyDataset):
    """Dataset for training identity mask probe on InsScene-15K data."""

    def __init__(
        self,
        root: str,
        root_vfm: str = None,
        sources: list = None,
        split: str = "train",
        vfm_name: str = "wan",
        feat_postfix: str = "_t749_layer20",
        feature_layer: int = None,
        feature_timestep: int = None,
        feature_prefix: str = "feature",
        feat_pixalign: bool = True,
        seed: int = None,
        num_views: int = 4,
        min_view_interval: int = 5,
        context_len: int = 76,
        query_idx_divisor: int = 4,
        target_h: int = 288,
        target_w: int = 512,
        train_ratio: float = 0.9,
        split_manifest: str = None,
        max_identity_classes: int = 256,
        window_size: int = 0,
        include_pmaps: bool = True,
        streaming_prefix: bool = False,
        prefix_stride: int = 1,
        prefix_min_len: int = 1,
        prefix_max_len: int = None,
        prefix_lengths: list = None,
        streaming_feat_root: str = None,
        streaming_prefix_dir_fmt: str = "prefix_{tail:06d}",
        context_feat_root: str = None,
        context_segment_dir_fmt: str = "context_{start:06d}_{tail:06d}",
        # ---------- Spatial Diagnostic Suite toggles ----------
        diag_overlap: bool = False,        # A2: compute (S,S) overlap matrix
        diag_hidden_obj: bool = False,     # B1: pick hidden object + polar target
        hidden_obj_min_visible_pixels: int = 200,
        streaming_shared_hidden_obj: bool = False,
        streaming_hidden_seed_prefix_len: int = 4,
        streaming_hidden_prefix_lengths: list = None,
        diag_action: bool = False,         # C1: load target_feat + emit action
        diag_abnormal: bool = False,       # A3: load shuffled vfm features
        diag_path_integration: bool = False,  # C2: recurrent multi-action feature prediction
        diag_counterfactual: bool = False,    # C3: multi-horizon action intervention prediction
        target_feat_root: str = None,      # root for C1/C2/C3 isolated target features
        shuffled_feat_root: str = None,    # root for A3 shuffled features
        action_horizons: list = None,      # frame offsets from action-reference frame, e.g. [1, 10, 30]
        counterfactual_min_overlap: float = 0.05,
        scramble_feat: bool = False,       # Control: replace VFM feat with randn (same shape)
        allow_missing_vfm: bool = False,   # Debug only: use dummy zeros if a normal VFM feature is missing
        no_obj_mask: bool = False,         # Ablation: replace obj mask with all-ones (global pool)
        pose_only: bool = False,           # Deprecated alias for unconditional_baseline
        unconditional_baseline: bool = False,  # zero visual/object conditions
        **kwargs,
    ):
        """
        Args:
            root: Path to InsScene-15K data root (containing processed_infinigen/, processed_scannetpp_v2/).
            root_vfm: Path to pre-extracted VFM features (optional, for full pipeline).
            sources: List of data sources to use. Default: ["processed_infinigen", "processed_scannetpp_v2"].
            split: "train" or "val".
            vfm_name: VFM backbone name.
            target_h, target_w: Target resolution to resize images and masks to.
            train_ratio: Fraction of scenes for training split.
            max_identity_classes: Cap on number of identity classes per scene.
            window_size: If > 0, split long videos into overlapping windows of this size.
                         Stride = window_size // 2.  0 or negative disables windowing.
        """
        # Site-specific storage is supplied by the scheduler/environment. This
        # keeps historical YAMLs runnable without baking one cluster's mount
        # points into the scientific configuration.
        root = os.environ.get("INSCENE_DATA_ROOT", root)
        root_vfm = os.environ.get("INSCENE_FEAT_ROOT", root_vfm)
        target_feat_root = os.environ.get("INSCENE_TARGET_FEAT_ROOT", target_feat_root)
        shuffled_feat_root = os.environ.get("INSCENE_SHUFFLED_FEAT_ROOT", shuffled_feat_root)
        context_feat_root = os.environ.get("INSCENE_CONTEXT_FEAT_ROOT", context_feat_root)
        if sources is None:
            sources = ["processed_infinigen", "processed_scannetpp_v2"]

        self.root = root
        self.root_vfm = root_vfm
        self.sources = sources
        self.split = split
        self.vfm_name = vfm_name
        self.feat_postfix = feat_postfix
        self.feature_layer = feature_layer
        self.feature_timestep = feature_timestep
        self.feature_prefix = feature_prefix
        self.feat_pixalign = feat_pixalign
        # Validation/test samples must not change between epochs. Training keeps
        # worker-seeded stochastic frame sampling unless a seed is explicit.
        self.seed = seed if seed is not None else (0 if split in {"val", "test"} else None)
        self.num_views = num_views
        self.min_view_interval = min_view_interval
        self.context_len = context_len
        self.query_idx_divisor = query_idx_divisor
        self.target_h = target_h
        self.target_w = target_w
        self.max_identity_classes = max_identity_classes
        self.split_manifest = split_manifest or os.environ.get("INSCENE_SPLIT_MANIFEST")
        self.window_size = window_size
        self.include_pmaps = include_pmaps
        self.streaming_prefix = streaming_prefix
        self.prefix_stride = max(int(prefix_stride), 1)
        self.prefix_min_len = max(int(prefix_min_len), 1)
        self.prefix_max_len = int(prefix_max_len) if prefix_max_len is not None else None
        if isinstance(prefix_lengths, str):
            prefix_lengths = prefix_lengths.replace(",", " ").split()
        self.prefix_lengths = (
            sorted({int(v) for v in prefix_lengths if int(v) > 0})
            if prefix_lengths is not None else None
        )
        self.streaming_feat_root = streaming_feat_root
        self.streaming_prefix_dir_fmt = streaming_prefix_dir_fmt
        self.context_feat_root = context_feat_root
        self.context_segment_dir_fmt = context_segment_dir_fmt
        self._streaming_prefix_meta_cache = {}
        self._context_segment_meta_cache = {}
        self._target_index_cache = {}
        self._streaming_hidden_obj_cache = {}
        # Diagnostic suite flags
        self.diag_overlap = diag_overlap
        self.diag_hidden_obj = diag_hidden_obj
        self.hidden_obj_min_visible_pixels = int(hidden_obj_min_visible_pixels)
        self.streaming_shared_hidden_obj = bool(streaming_shared_hidden_obj)
        self.streaming_hidden_seed_prefix_len = int(streaming_hidden_seed_prefix_len)
        if isinstance(streaming_hidden_prefix_lengths, str):
            streaming_hidden_prefix_lengths = streaming_hidden_prefix_lengths.replace(",", " ").split()
        self.streaming_hidden_prefix_lengths = (
            sorted({int(v) for v in streaming_hidden_prefix_lengths if int(v) > 0})
            if streaming_hidden_prefix_lengths is not None else None
        )
        self.diag_action = diag_action
        self.diag_abnormal = diag_abnormal
        self.diag_path_integration = diag_path_integration
        self.diag_counterfactual = diag_counterfactual
        self.target_feat_root = target_feat_root
        self.shuffled_feat_root = shuffled_feat_root
        self.action_horizons = sorted({int(h) for h in (action_horizons or []) if int(h) > 0})
        self.counterfactual_min_overlap = counterfactual_min_overlap
        self.scramble_feat = scramble_feat
        self.allow_missing_vfm = allow_missing_vfm
        self.no_obj_mask = no_obj_mask
        self.pose_only = pose_only or unconditional_baseline
        # A2/B1/C1 all need extrinsics normalized via pointmap-based scale
        # (invert_pose_ref_and_scale requires pointmaps).
        if (diag_overlap or diag_hidden_obj or diag_action or diag_path_integration or diag_counterfactual) and not include_pmaps:
            self.include_pmaps = True
        self.kwargs = kwargs

        # Collect all scenes
        self.scenes = []
        for source in sources:
            source_path = os.path.join(root, source)
            if not os.path.isdir(source_path):
                logger.warning(f"Source path {source_path} not found, skipping.")
                continue

            if source == "processed_infinigen":
                self._collect_infinigen_scenes(source_path)
            elif source == "processed_scannetpp_v2":
                self._collect_scannetpp_scenes(source_path)

        # Split at scene level before window expansion. A frozen manifest is
        # required for a genuine test set and strongly recommended otherwise.
        if self.split_manifest:
            with open(self.split_manifest) as f:
                manifest = json.load(f)
            splits = manifest.get("splits", manifest)
            if split not in splits:
                raise ValueError(f"Split {split!r} missing from {self.split_manifest}")
            allowed = set(splits[split])
            self.scenes = [s for s in self.scenes if self._scene_key(s) in allowed]
            missing = allowed - {self._scene_key(s) for s in self.scenes}
            if missing:
                raise ValueError(
                    f"Split manifest references {len(missing)} unavailable scenes; "
                    f"first={sorted(missing)[0]}"
                )
        elif split == "all":
            pass
        elif split == "test":
            raise ValueError("A distinct test split requires split_manifest=...")
        else:
            logger.warning("No split manifest supplied; using legacy deterministic train/val split")
            rng = np.random.default_rng(seed=42)
            indices = rng.permutation(len(self.scenes))
            split_idx = int(len(self.scenes) * train_ratio)
            selected = indices[:split_idx] if split == "train" else indices[split_idx:]
            self.scenes = [self.scenes[i] for i in selected]

        # Expand scenes with windowing for long videos. Streaming prefix is its
        # own indexing scheme: each sample is exactly [0, ..., tail].
        if getattr(self, "streaming_prefix", False):
            self._expand_scenes_with_streaming_prefix()
        elif self.window_size > 0:
            self._expand_scenes_with_windows()

        logger.info(
            f"InsScene15KDataset: {len(self.scenes)} samples for {split} split "
            f"from sources {sources} (window_size={self.window_size}, "
            f"streaming_prefix={self.streaming_prefix})"
        )

    def _expand_scenes_with_streaming_prefix(self):
        """Expand each scene into online prefix samples H_t = [I_0, ..., I_t]."""
        expanded = []
        max_future = max(self._streaming_future_horizons(default=[]), default=0)
        required_hidden_len = (
            max(self._streaming_hidden_prefix_lengths())
            if self._use_shared_streaming_hidden_obj() else 0
        )
        for scene in self.scenes:
            nf = int(scene["num_frames"])
            if required_hidden_len and nf < required_hidden_len:
                continue
            max_len = nf - max_future
            if self.prefix_max_len is not None:
                max_len = min(max_len, self.prefix_max_len)
            if max_len < self.prefix_min_len:
                continue
            if self.prefix_lengths:
                lengths = [
                    length for length in self.prefix_lengths
                    if self.prefix_min_len <= length <= max_len
                ]
            else:
                lengths = range(self.prefix_min_len, max_len + 1, self.prefix_stride)
            for length in lengths:
                tail = length - 1
                prefix_scene = dict(scene)
                prefix_scene["streaming_tail"] = tail
                prefix_scene["streaming_indices"] = list(range(length))
                expanded.append(prefix_scene)
        logger.info(
            f"Streaming prefix expansion: {len(self.scenes)} scenes -> {len(expanded)} samples "
            f"(min_len={self.prefix_min_len}, max_len={self.prefix_max_len}, "
            f"stride={self.prefix_stride}, lengths={self.prefix_lengths}, "
            f"max_future={max_future})"
        )
        self.scenes = expanded

    def _use_shared_streaming_hidden_obj(self):
        return bool(
            self.streaming_prefix
            and self.diag_hidden_obj
            and self.streaming_shared_hidden_obj
        )

    def _streaming_hidden_prefix_lengths(self):
        if self.streaming_hidden_prefix_lengths:
            return list(self.streaming_hidden_prefix_lengths)
        if self.prefix_lengths:
            return list(self.prefix_lengths)
        if self.prefix_max_len is not None:
            return [int(self.prefix_max_len)]
        return [int(self.prefix_min_len)]

    def _streaming_hidden_seed_visible_indices(self):
        # For seed_prefix_len=4, the object must be visible in frames [0,1,2]
        # and hidden at frame 3.
        seed_len = max(int(self.streaming_hidden_seed_prefix_len), 2)
        return list(range(seed_len - 1))

    def _streaming_hidden_tail_indices(self):
        tails = {max(int(length) - 1, 0) for length in self._streaming_hidden_prefix_lengths()}
        tails.add(max(int(self.streaming_hidden_seed_prefix_len) - 1, 0))
        return sorted(tails)

    def _streaming_future_horizons(self, default=None):
        """Future frame offsets predicted from the current streaming tail."""
        if not self.streaming_prefix:
            return default if default is not None else []
        if self.diag_path_integration or self.diag_counterfactual:
            return list(self.action_horizons)
        if getattr(self, "diag_action", False):
            return list(self.action_horizons) if self.action_horizons else [1]
        return default if default is not None else []

    def _expand_scenes_with_windows(self):
        """Expand scenes into overlapping windows for long videos.

        Scenes shorter than window_size are kept as-is.
        Longer scenes are split with stride = window_size // 2.
        """
        min_window = self.num_views * max(self.min_view_interval, 1)
        stride = max(self.window_size // 2, 1)
        expanded = []
        for scene in self.scenes:
            nf = scene["num_frames"]
            if nf <= self.window_size:
                expanded.append(scene)
            else:
                start = 0
                while start < nf:
                    end = min(start + self.window_size, nf)
                    if end - start < min_window:
                        break
                    win_scene = dict(scene)  # shallow copy (valid_frames list shared)
                    win_scene["window_start"] = start
                    win_scene["window_end"] = end
                    expanded.append(win_scene)
                    start += stride
        logger.info(
            f"Window expansion: {len(self.scenes)} scenes -> {len(expanded)} samples "
            f"(window_size={self.window_size}, stride={stride})"
        )
        self.scenes = expanded

    def _try_add_infinigen_scene(self, candidate_dir):
        """Try to register one infinigen scene/subscene directory."""
        frames_dir = os.path.join(candidate_dir, "frames")
        img_dir = os.path.join(frames_dir, "Image", "camera_0")
        mask_dir = os.path.join(frames_dir, "ObjectSegmentation", "camera_0")
        cam_dir = os.path.join(frames_dir, "camview", "camera_0")

        if not all(os.path.isdir(d) for d in [img_dir, mask_dir, cam_dir]):
            return False

        img_files = sorted(
            [f for f in os.listdir(img_dir) if f.endswith(".png")]
        )
        if len(img_files) < self.num_views:
            return False

        self.scenes.append(
            {
                "source": "infinigen",
                "scene_dir": candidate_dir,
                "num_frames": len(img_files),
            }
        )
        return True

    def _collect_infinigen_scenes(self, source_path):
        """Collect scenes from processed_infinigen directory.
        
        Supports two layouts:
          - Flat:     source_path/scene_XXX/frames/...  (sample data)
          - Nested:   source_path/scene_XXX/<subscene_hash>/frames/...  (full dataset)
        """
        for scene_dir in sorted(glob(os.path.join(source_path, "scene_*"))):
            # Try flat layout first (scene_dir itself has frames/)
            if self._try_add_infinigen_scene(scene_dir):
                continue
            # Try nested layout (subscene dirs inside scene_dir)
            for sub_dir in sorted(glob(os.path.join(scene_dir, "*"))):
                if os.path.isdir(sub_dir):
                    self._try_add_infinigen_scene(sub_dir)

    def _collect_scannetpp_scenes(self, source_path):
        """Collect scenes from processed_scannetpp_v2 directory.
        
        Supports nested extraction layout where zip extracts to:
          source_path/processed_scannetpp_v2/<scene_id>/...
        """
        # Handle nested directory from zip extraction
        nested = os.path.join(source_path, "processed_scannetpp_v2")
        if os.path.isdir(nested):
            source_path = nested

        for scene_id in sorted(os.listdir(source_path)):
            scene_dir = os.path.join(source_path, scene_id)
            if not os.path.isdir(scene_dir):
                continue

            img_dir = os.path.join(scene_dir, "images")
            mask_dir = os.path.join(scene_dir, "refined_ins_ids")
            meta_path = os.path.join(scene_dir, "scene_iphone_metadata.npz")

            if not all(
                os.path.exists(p) for p in [img_dir, mask_dir, meta_path]
            ):
                continue

            # Count frames that have both image and mask
            img_files = sorted(
                [f for f in os.listdir(img_dir) if f.endswith(".jpg")]
            )
            mask_files = set(os.listdir(mask_dir))
            valid_frames = [
                f
                for f in img_files
                if f"{f}.npy" in mask_files
            ]

            if len(valid_frames) < self.num_views:
                continue

            self.scenes.append(
                {
                    "source": "scannetpp",
                    "scene_dir": scene_dir,
                    "num_frames": len(valid_frames),
                    "valid_frames": valid_frames,
                }
            )

    def __len__(self):
        return len(self.scenes)

    def _scene_key(self, scene_info):
        return f"{scene_info['source']}/{self._feat_scene_name(scene_info)}"

    def get_stats(self):
        return f"{len(self)} scenes"

    def _sample_query_frames(self, rng, n, total, offset=0):
        """Sample n frame indices with minimum gap constraint."""
        min_gap = self.min_view_interval or 0
        if min_gap <= 0 or n <= 1:
            return (
                torch.linspace(offset, offset + total - 1, n, dtype=torch.float32)
                .round()
                .to(torch.long)
            )

        needed = (n - 1) * min_gap + 1
        if needed > total:
            # Fallback: evenly spaced
            return (
                torch.linspace(offset, offset + total - 1, n, dtype=torch.float32)
                .round()
                .to(torch.long)
            )

        slack = total - needed
        cuts = np.sort(rng.integers(0, slack + 1, size=n - 1, dtype=int))
        extras = np.diff(np.concatenate(([0], cuts, [slack])))

        idxs = [offset]
        for extra in extras[:-1]:
            idxs.append(idxs[-1] + min_gap + int(extra))
        return torch.as_tensor(idxs, dtype=torch.long)

    def _load_infinigen_scene(self, scene_info, sel_indices):
        """Load frames from an infinigen scene."""
        frames_dir = os.path.join(scene_info["scene_dir"], "frames")
        img_dir = os.path.join(frames_dir, "Image", "camera_0")
        mask_dir = os.path.join(frames_dir, "ObjectSegmentation", "camera_0")
        cam_dir = os.path.join(frames_dir, "camview", "camera_0")
        depth_dir = os.path.join(frames_dir, "Depth", "camera_0")

        # Get sorted list of frame indices from filenames
        img_files = sorted(
            [f for f in os.listdir(img_dir) if f.endswith(".png")]
        )

        images = []
        masks = []
        depthmaps = []
        intrinsics = []
        extrinsics = []

        for idx in sel_indices:
            idx = idx.item()
            if idx >= len(img_files):
                idx = len(img_files) - 1

            img_name = img_files[idx]
            frame_id = img_name.replace("Image_", "").replace(".png", "")

            # Load image
            img_path = os.path.join(img_dir, img_name)
            img = np.array(Image.open(img_path).convert("RGB"))
            images.append(torch.from_numpy(img).float().permute(2, 0, 1))

            # Load identity mask
            mask_path = os.path.join(
                mask_dir, f"ObjectSegmentation_{frame_id}.npy"
            )
            if os.path.exists(mask_path):
                mask = np.load(mask_path).astype(np.int64)
            else:
                mask = np.zeros(
                    (images[-1].shape[1], images[-1].shape[2]), dtype=np.int64
                )
            masks.append(torch.from_numpy(mask).long())

            # Load depth (float32, meters; 1e10 = invalid/sky)
            depth_path = os.path.join(depth_dir, f"Depth_{frame_id}.npy")
            if os.path.exists(depth_path):
                depth = np.load(depth_path).astype(np.float32)
                depth[depth > 1e6] = 0.0  # mark invalid
            else:
                depth = np.zeros(
                    (images[-1].shape[1], images[-1].shape[2]), dtype=np.float32
                )
            depthmaps.append(torch.from_numpy(depth))

            # Load camera parameters
            cam_path = os.path.join(cam_dir, f"camview_{frame_id}.npz")
            cam = np.load(cam_path)
            K = torch.from_numpy(cam["K"].astype(np.float32))
            T = torch.from_numpy(cam["T"].astype(np.float32))
            intrinsics.append(K)
            extrinsics.append(T[:3, :4])

        return (
            torch.stack(images),      # (S, 3, H, W)
            torch.stack(masks),       # (S, H, W)
            torch.stack(depthmaps),   # (S, H, W)
            torch.stack(intrinsics),  # (S, 3, 3)
            torch.stack(extrinsics),  # (S, 3, 4)
        )

    def _load_scannetpp_scene(self, scene_info, sel_indices):
        """Load frames from a scannetpp scene."""
        scene_dir = scene_info["scene_dir"]
        img_dir = os.path.join(scene_dir, "images")
        mask_dir = os.path.join(scene_dir, "refined_ins_ids")
        depth_dir = os.path.join(scene_dir, "depth")
        meta_path = os.path.join(scene_dir, "scene_iphone_metadata.npz")

        valid_frames = scene_info["valid_frames"]
        meta = np.load(meta_path)
        all_images_list = list(meta["images"])

        images = []
        masks = []
        depthmaps = []
        intrinsics = []
        extrinsics = []

        for idx in sel_indices:
            idx = idx.item()
            if idx >= len(valid_frames):
                idx = len(valid_frames) - 1

            frame_name = valid_frames[idx]

            # Load image
            img_path = os.path.join(img_dir, frame_name)
            img = np.array(Image.open(img_path).convert("RGB"))
            images.append(torch.from_numpy(img).float().permute(2, 0, 1))

            # Load identity mask
            mask_path = os.path.join(mask_dir, f"{frame_name}.npy")
            mask = np.load(mask_path).astype(np.int64)
            masks.append(torch.from_numpy(mask).long())

            # Load depth (uint16 PNG in millimeters -> convert to meters)
            depth_stem = os.path.splitext(frame_name)[0]
            depth_path = os.path.join(depth_dir, f"{depth_stem}.png")
            if os.path.exists(depth_path):
                depth = np.array(Image.open(depth_path)).astype(np.float32) / 1000.0
            else:
                depth = np.zeros(
                    (images[-1].shape[1], images[-1].shape[2]), dtype=np.float32
                )
            depthmaps.append(torch.from_numpy(depth))

            # Find this frame's index in the metadata
            try:
                meta_idx = all_images_list.index(frame_name)
            except ValueError:
                meta_idx = all_images_list.index(
                    os.path.splitext(frame_name)[0]
                ) if os.path.splitext(frame_name)[0] in all_images_list else 0

            K = torch.from_numpy(meta["intrinsics"][meta_idx].astype(np.float32))
            T = torch.from_numpy(meta["trajectories"][meta_idx].astype(np.float32))
            intrinsics.append(K)
            extrinsics.append(T[:3, :4])

        return (
            torch.stack(images),      # (S, 3, H, W)
            torch.stack(masks),       # (S, H, W)
            torch.stack(depthmaps),   # (S, H, W)
            torch.stack(intrinsics),  # (S, 3, 3)
            torch.stack(extrinsics),  # (S, 3, 4)
        )

    def _load_identity_masks_scene(self, scene_info, sel_indices):
        """Load only raw instance masks for the requested frame indices."""
        masks = []
        if scene_info["source"] == "infinigen":
            frames_dir = os.path.join(scene_info["scene_dir"], "frames")
            img_dir = os.path.join(frames_dir, "Image", "camera_0")
            mask_dir = os.path.join(frames_dir, "ObjectSegmentation", "camera_0")
            img_files = sorted([f for f in os.listdir(img_dir) if f.endswith(".png")])
            for idx in sel_indices:
                idx = min(int(idx.item()), len(img_files) - 1)
                img_name = img_files[idx]
                frame_id = img_name.replace("Image_", "").replace(".png", "")
                mask_path = os.path.join(mask_dir, f"ObjectSegmentation_{frame_id}.npy")
                if os.path.exists(mask_path):
                    mask = np.load(mask_path).astype(np.int64)
                else:
                    with Image.open(os.path.join(img_dir, img_name)) as img:
                        w, h = img.size
                    mask = np.zeros((h, w), dtype=np.int64)
                masks.append(torch.from_numpy(mask).long())
        elif scene_info["source"] == "scannetpp":
            scene_dir = scene_info["scene_dir"]
            mask_dir = os.path.join(scene_dir, "refined_ins_ids")
            valid_frames = scene_info["valid_frames"]
            for idx in sel_indices:
                idx = min(int(idx.item()), len(valid_frames) - 1)
                frame_name = valid_frames[idx]
                mask_path = os.path.join(mask_dir, f"{frame_name}.npy")
                masks.append(torch.from_numpy(np.load(mask_path).astype(np.int64)).long())
        else:
            raise ValueError(f"Unknown source: {scene_info['source']}")
        return torch.stack(masks)

    def _load_camera_depth_scene(self, scene_info, sel_indices):
        """Load only depth, intrinsics, and extrinsics for feature-only probes."""
        depthmaps = []
        intrinsics = []
        extrinsics = []

        if scene_info["source"] == "infinigen":
            frames_dir = os.path.join(scene_info["scene_dir"], "frames")
            img_dir = os.path.join(frames_dir, "Image", "camera_0")
            cam_dir = os.path.join(frames_dir, "camview", "camera_0")
            depth_dir = os.path.join(frames_dir, "Depth", "camera_0")
            img_files = sorted([f for f in os.listdir(img_dir) if f.endswith(".png")])

            for idx in sel_indices:
                idx = min(int(idx.item()), len(img_files) - 1)
                frame_id = img_files[idx].replace("Image_", "").replace(".png", "")

                depth_path = os.path.join(depth_dir, f"Depth_{frame_id}.npy")
                if os.path.exists(depth_path):
                    depth = np.load(depth_path).astype(np.float32)
                    depth[depth > 1e6] = 0.0
                else:
                    # Rare fallback; use the RGB size without keeping the image.
                    img_path = os.path.join(img_dir, img_files[idx])
                    with Image.open(img_path) as img:
                        w, h = img.size
                    depth = np.zeros((h, w), dtype=np.float32)
                depthmaps.append(torch.from_numpy(depth))

                cam_path = os.path.join(cam_dir, f"camview_{frame_id}.npz")
                cam = np.load(cam_path)
                intrinsics.append(torch.from_numpy(cam["K"].astype(np.float32)))
                T = torch.from_numpy(cam["T"].astype(np.float32))
                extrinsics.append(T[:3, :4])

        elif scene_info["source"] == "scannetpp":
            scene_dir = scene_info["scene_dir"]
            depth_dir = os.path.join(scene_dir, "depth")
            meta_path = os.path.join(scene_dir, "scene_iphone_metadata.npz")
            valid_frames = scene_info["valid_frames"]
            meta = np.load(meta_path)
            all_images_list = list(meta["images"])

            for idx in sel_indices:
                idx = min(int(idx.item()), len(valid_frames) - 1)
                frame_name = valid_frames[idx]
                depth_stem = os.path.splitext(frame_name)[0]
                depth_path = os.path.join(depth_dir, f"{depth_stem}.png")
                if os.path.exists(depth_path):
                    depth = np.array(Image.open(depth_path)).astype(np.float32) / 1000.0
                else:
                    depth = np.zeros((self.target_h, self.target_w), dtype=np.float32)
                depthmaps.append(torch.from_numpy(depth))

                try:
                    meta_idx = all_images_list.index(frame_name)
                except ValueError:
                    stem = os.path.splitext(frame_name)[0]
                    meta_idx = all_images_list.index(stem) if stem in all_images_list else 0
                intrinsics.append(torch.from_numpy(meta["intrinsics"][meta_idx].astype(np.float32)))
                T = torch.from_numpy(meta["trajectories"][meta_idx].astype(np.float32))
                extrinsics.append(T[:3, :4])
        else:
            raise ValueError(f"Unknown source: {scene_info['source']}")

        return (
            torch.stack(depthmaps),
            torch.stack(intrinsics),
            torch.stack(extrinsics),
        )

    def _resize_depth_to_target(self, depthmaps, intrinsics):
        """Resize depth maps and rescale intrinsics without loading RGB/masks."""
        _, orig_h, orig_w = depthmaps.shape
        scale_h = self.target_h / orig_h
        scale_w = self.target_w / orig_w
        depthmaps = F.interpolate(
            depthmaps.unsqueeze(1),
            size=(self.target_h, self.target_w),
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)
        intrinsics = intrinsics.clone()
        intrinsics[:, 0, 0] *= scale_w
        intrinsics[:, 0, 2] *= scale_w
        intrinsics[:, 1, 1] *= scale_h
        intrinsics[:, 1, 2] *= scale_h
        return depthmaps, intrinsics

    def _resize_to_target(self, images, masks, depthmaps, intrinsics):
        """Resize images, masks, depth to target resolution and rescale intrinsics."""
        _, _, orig_h, orig_w = images.shape
        scale_h = self.target_h / orig_h
        scale_w = self.target_w / orig_w

        # Resize images: (S, 3, H, W) -> (S, 3, target_h, target_w)
        images = F.interpolate(
            images, size=(self.target_h, self.target_w), mode="bilinear", align_corners=False
        )
        # Resize masks: (S, H, W) -> (S, target_h, target_w) using nearest to preserve IDs
        masks = (
            F.interpolate(
                masks.unsqueeze(1).float(),
                size=(self.target_h, self.target_w),
                mode="nearest",
            )
            .squeeze(1)
            .long()
        )
        # Resize depth: (S, H, W) -> (S, target_h, target_w) using bilinear
        depthmaps = F.interpolate(
            depthmaps.unsqueeze(1),
            size=(self.target_h, self.target_w),
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)

        # Rescale intrinsics for new resolution
        intrinsics = intrinsics.clone()
        intrinsics[:, 0, 0] *= scale_w  # fx
        intrinsics[:, 0, 2] *= scale_w  # cx
        intrinsics[:, 1, 1] *= scale_h  # fy
        intrinsics[:, 1, 2] *= scale_h  # cy

        return images, masks, depthmaps, intrinsics

    def _resize_masks_to_target(self, masks):
        """Resize raw instance masks to the dataset target resolution."""
        return (
            F.interpolate(
                masks.unsqueeze(1).float(),
                size=(self.target_h, self.target_w),
                mode="nearest",
            )
            .squeeze(1)
            .long()
        )

    def _remap_identity_ids(self, masks):
        """Remap identity IDs to contiguous range [0, N) and cap at max_identity_classes."""
        unique_ids = masks.unique()
        if len(unique_ids) > self.max_identity_classes:
            # Keep the most frequent classes
            flat = masks.reshape(-1)
            counts = torch.bincount(flat[flat >= 0])
            top_ids = counts.argsort(descending=True)[: self.max_identity_classes]
            keep_set = set(top_ids.tolist())
            # Mark others as -1 (ignore)
            new_masks = torch.full_like(masks, -1)
            for new_id, old_id in enumerate(sorted(keep_set)):
                new_masks[masks == old_id] = new_id
            return new_masks

        # Simple remap
        id_map = {}
        new_masks = torch.full_like(masks, -1)
        for new_id, old_id in enumerate(unique_ids.tolist()):
            if old_id < 0:
                continue
            id_map[old_id] = new_id
            new_masks[masks == old_id] = new_id
        return new_masks

    # --------------- Spatial Diagnostic helpers ---------------
    def _feat_scene_name(self, scene_info):
        """Match the directory naming convention used by features/run_inscene15k.py."""
        scene_dir = scene_info["scene_dir"]
        if scene_info["source"] == "infinigen":
            parent = os.path.basename(os.path.dirname(scene_dir))
            base = os.path.basename(scene_dir)
            return f"{parent}__{base}" if parent.startswith("scene_") else base
        return os.path.basename(scene_dir)

    def _feat_filename(self) -> str:
        """Filename used inside a per-scene feature directory.

        The convention matches features/run_inscene15k.py:
          - wan / cogvideox: feature{feat_postfix}.sft  (postfix carries _t<t>_layer<n>)
          - vjepa2:          feature_layer<n>.sft       (postfix carries _layer<n> too;
                                                         user is expected to set
                                                         feat_postfix accordingly)
          - vjepa:           feature.sft
        ``feat_postfix`` remains supported for old configs.  New layer-sweep
        configs can pass ``feature_layer`` instead, which keeps the current
        default filenames but makes layer changes an ordinary Hydra override.
        """
        return feature_filename(
            self.vfm_name,
            feat_postfix=self.feat_postfix,
            feature_layer=self.feature_layer,
            feature_timestep=self.feature_timestep,
            feature_prefix=self.feature_prefix,
        )

    def _streaming_prefix_dir_name(self, scene_info) -> str:
        return self.streaming_prefix_dir_fmt.format(
            tail=int(scene_info["streaming_tail"]),
            length=len(scene_info["streaming_indices"]),
        )

    def _streaming_prefix_scene_dir(self, scene_info) -> str | None:
        root = self.streaming_feat_root or self.root_vfm
        if root is None:
            return None
        return os.path.join(
            root,
            self.vfm_name,
            scene_info["source"],
            self._feat_scene_name(scene_info),
        )

    def _streaming_prefix_record(self, scene_info) -> dict | None:
        scene_dir = self._streaming_prefix_scene_dir(scene_info)
        if scene_dir is None:
            return None

        meta_path = os.path.join(scene_dir, "prefix_index.npy")
        if meta_path not in self._streaming_prefix_meta_cache:
            records_by_tail = {}
            if os.path.exists(meta_path):
                try:
                    records = np.load(meta_path, allow_pickle=True).tolist()
                    if isinstance(records, dict):
                        records = records.values()
                    for record in records:
                        if isinstance(record, dict) and "tail" in record:
                            records_by_tail[int(record["tail"])] = record
                except Exception as e:
                    logger.warning(f"Failed to read streaming prefix metadata {meta_path}: {e}")
            self._streaming_prefix_meta_cache[meta_path] = records_by_tail

        return self._streaming_prefix_meta_cache[meta_path].get(
            int(scene_info["streaming_tail"])
        )

    def _context_segment_dir_name(self, start: int, tail: int) -> str:
        return self.context_segment_dir_fmt.format(start=int(start), tail=int(tail))

    def _context_segment_scene_dir(self, scene_info) -> str | None:
        root = self.context_feat_root
        if root is None:
            return None
        return os.path.join(
            root,
            self.vfm_name,
            scene_info["source"],
            self._feat_scene_name(scene_info),
        )

    def _context_segment_records(self, scene_info) -> dict:
        scene_dir = self._context_segment_scene_dir(scene_info)
        if scene_dir is None:
            return {}

        meta_path = os.path.join(scene_dir, "context_index.npy")
        if meta_path not in self._context_segment_meta_cache:
            records_by_key = {}
            if os.path.exists(meta_path):
                try:
                    records = np.load(meta_path, allow_pickle=True).tolist()
                    if isinstance(records, dict):
                        records = records.values()
                    for record in records:
                        if isinstance(record, dict) and "start" in record and "tail" in record:
                            key = (int(record["start"]), int(record["tail"]))
                            records_by_key[key] = record
                except Exception as e:
                    logger.warning(f"Failed to read context segment metadata {meta_path}: {e}")
            self._context_segment_meta_cache[meta_path] = records_by_key

        return self._context_segment_meta_cache[meta_path]

    def _context_select_indices(self, context_start: int, context_tail: int) -> torch.Tensor:
        """Return a fixed-length list of frame ids for batching context features."""
        valid = list(range(int(context_start), int(context_tail) + 1))
        target_len = max(int(self.context_len), 1)
        if len(valid) >= target_len:
            valid = valid[-target_len:]
        else:
            valid = valid + [int(context_tail)] * (target_len - len(valid))
        return torch.as_tensor(valid, dtype=torch.long)

    def _target_indices(self, scene_info):
        """Return the exact frame ids represented by a target cache."""
        if self.target_feat_root is None:
            return None
        cache_key = (scene_info["scene_dir"], self.vfm_name)
        if cache_key in self._target_index_cache:
            return self._target_index_cache[cache_key]
        path = os.path.join(
            self.target_feat_root,
            self.vfm_name,
            scene_info["source"],
            self._feat_scene_name(scene_info),
            "target_indices.npy",
        )
        indices = torch.from_numpy(np.load(path)).long() if os.path.exists(path) else None
        self._target_index_cache[cache_key] = indices
        return indices

    def _shared_streaming_hidden_obj_id(self, scene_info):
        """Return the scene-level object id shared by all B streaming prefixes."""
        lengths = tuple(self._streaming_hidden_prefix_lengths())
        seed_len = int(self.streaming_hidden_seed_prefix_len)
        key = (
            scene_info["source"],
            scene_info["scene_dir"],
            lengths,
            seed_len,
            self.hidden_obj_min_visible_pixels,
        )
        if key in self._streaming_hidden_obj_cache:
            cached = self._streaming_hidden_obj_cache[key]
            return None if cached is None else int(cached)

        from vidfm3d.utils.spatial_diag import select_streaming_hidden_object_id

        required_len = max(max(lengths), seed_len)
        sel = torch.arange(required_len, dtype=torch.long)
        masks = self._load_identity_masks_scene(scene_info, sel)
        masks = self._resize_masks_to_target(masks)
        obj_id = select_streaming_hidden_object_id(
            masks,
            seed_visible_indices=self._streaming_hidden_seed_visible_indices(),
            hidden_tail_indices=self._streaming_hidden_tail_indices(),
            min_visible_pixels=self.hidden_obj_min_visible_pixels,
        )
        self._streaming_hidden_obj_cache[key] = obj_id
        return None if obj_id is None else int(obj_id)

    def _load_target_feat(self, scene_info, target_global_idx, num_frames):
        """Load an isolated feature for one exact frame.

        Expected file layout:
            {target_feat_root}/{vfm}/{source}/{scene}/feature{feat_postfix}.sft
        which contains:
            feat:           (M, H_f, W_f, C)   — features for M target frames
            target_indices: (M,) long          — original frame indices

        Returns the feature for the exact target index, or None if missing.
        Nearest-neighbour targets are scientifically invalid because the action
        and camera reference would then describe a different frame.
        """
        if self.target_feat_root is None:
            return None
        path = os.path.join(
            self.target_feat_root,
            self.vfm_name,
            scene_info["source"],
            self._feat_scene_name(scene_info),
            self._feat_filename(),
        )
        if not os.path.exists(path):
            return None
        try:
            data = load_file(path)
        except Exception as e:
            logger.warning(f"Failed to read target feat {path}: {e}")
            return None
        feat = data["feat"].float()                            # (M, H_f, W_f, C)
        tgt_idx = self._target_indices(scene_info)
        if tgt_idx is None:
            return None
        matches = torch.nonzero(tgt_idx == int(target_global_idx), as_tuple=False).flatten()
        return feat[int(matches[0].item())] if matches.numel() else None

    def _load_target_feats(self, scene_info, target_global_indices, num_frames):
        """Load multiple isolated frame features with one safetensors read."""
        if self.target_feat_root is None:
            return [None for _ in target_global_indices]
        path = os.path.join(
            self.target_feat_root,
            self.vfm_name,
            scene_info["source"],
            self._feat_scene_name(scene_info),
            self._feat_filename(),
        )
        if not os.path.exists(path):
            return [None for _ in target_global_indices]
        try:
            data = load_file(path)
        except Exception as e:
            logger.warning(f"Failed to read target feat {path}: {e}")
            return [None for _ in target_global_indices]

        feat = data["feat"].float()
        tgt_idx = self._target_indices(scene_info)
        if tgt_idx is None:
            return [None for _ in target_global_indices]
        index_to_row = {int(frame): row for row, frame in enumerate(tgt_idx.tolist())}
        return [
            feat[index_to_row[int(target_idx)]]
            if int(target_idx) in index_to_row else None
            for target_idx in target_global_indices
        ]

    def _load_vfm_feat_for_selection(self, scene_info, num_frames, sel_global):
        """Load normal VFM features and select the requested global frame indices."""
        if self.root_vfm is None:
            return (
                torch.zeros(self.num_views, 18, 32, 1536, dtype=torch.float32),
                torch.arange(self.num_views),
            )

        vfm_feat_path = os.path.join(
            self.root_vfm,
            self.vfm_name,
            scene_info["source"],
            self._feat_scene_name(scene_info),
            self._feat_filename(),
        )
        if not os.path.exists(vfm_feat_path):
            msg = f"VFM feature not found at {vfm_feat_path}"
            if not self.allow_missing_vfm:
                raise FileNotFoundError(
                    msg + "; run feature extraction first or set allow_missing_vfm=True for debugging"
                )
            logger.warning(f"{msg}, using dummy because allow_missing_vfm=True.")
            dummy_channels = default_feature_channels(self.vfm_name)
            return (
                torch.zeros(self.num_views, 18, 32, dummy_channels, dtype=torch.float32),
                torch.arange(self.num_views),
            )

        vfm_feat = load_file(vfm_feat_path)["feat"].float()
        if self.vfm_name == "cogvideox":
            vfm_feat = vfm_feat.reshape(-1, *vfm_feat.shape[2:])

        T = vfm_feat.shape[0]
        vfm_idx = (
            torch.round(sel_global.float() / max(num_frames - 1, 1) * (T - 1))
            .long()
            .clamp(0, T - 1)
        )
        if self.feat_pixalign:
            vfm_feat = vfm_feat[vfm_idx]
            vfm_idx = torch.arange(vfm_feat.shape[0], device=vfm_feat.device)

        if self.scramble_feat:
            g = torch.Generator()
            _seed_n = int(hashlib.md5((scene_info["scene_dir"] + ":n").encode()).hexdigest()[:8], 16) & 0xFFFFFFFF
            g.manual_seed(_seed_n)
            vfm_feat = torch.randn(vfm_feat.shape, generator=g, dtype=vfm_feat.dtype)
        if self.pose_only:
            vfm_feat = torch.zeros_like(vfm_feat)

        return vfm_feat, vfm_idx

    def _load_streaming_prefix_feat(self, scene_info, sel_global):
        """Load the precomputed VFM feature for one online prefix sample."""
        scene_dir = self._streaming_prefix_scene_dir(scene_info)
        if scene_dir is None:
            return (
                torch.zeros(len(sel_global), 18, 32, 1536, dtype=torch.float32),
                torch.arange(len(sel_global)),
            )

        vfm_feat_path = os.path.join(
            scene_dir,
            self._streaming_prefix_dir_name(scene_info),
            self._feat_filename(),
        )
        if not os.path.exists(vfm_feat_path):
            msg = f"Streaming prefix VFM feature not found at {vfm_feat_path}"
            if not self.allow_missing_vfm:
                raise FileNotFoundError(
                    msg + "; run feature extraction with --mode streaming_prefix first "
                    "or set allow_missing_vfm=True for debugging"
                )
            logger.warning(f"{msg}, using dummy because allow_missing_vfm=True.")
            dummy_channels = default_feature_channels(self.vfm_name)
            return (
                torch.zeros(len(sel_global), 18, 32, dummy_channels, dtype=torch.float32),
                torch.arange(len(sel_global)),
            )

        vfm_feat = load_file(vfm_feat_path)["feat"].float()
        if self.vfm_name == "cogvideox":
            vfm_feat = vfm_feat.reshape(-1, *vfm_feat.shape[2:])
        if vfm_feat.ndim != 4:
            raise ValueError(
                f"Expected streaming prefix VFM feature feat to be 4D (T,H,W,C), "
                f"got {tuple(vfm_feat.shape)} from {vfm_feat_path}."
            )

        T = vfm_feat.shape[0]
        prefix_len = len(sel_global)
        meta_record = self._streaming_prefix_record(scene_info)
        input_length = None
        valid_length = prefix_len
        pad_mode = None
        if meta_record is not None:
            input_length = int(meta_record.get("input_length", 0) or 0)
            valid_length = int(meta_record.get("valid_length", prefix_len))
            pad_mode = meta_record.get("pad_mode")

        if input_length is not None and input_length > 1 and T > 1:
            # Prefix extraction pads short inputs by repeating the current tail
            # frame. Historical frames use their real positions in the padded
            # input; the tail can use its final repeated occurrence.
            positions = torch.arange(prefix_len, dtype=torch.float32)
            if (
                pad_mode == "repeat_tail"
                and prefix_len > 0
                and input_length > valid_length
            ):
                positions[-1] = float(input_length - 1)
            vfm_idx = (
                positions / float(input_length - 1) * float(T - 1)
            ).round().long().clamp(0, T - 1)
        elif prefix_len <= 1:
            vfm_idx = torch.zeros(prefix_len, dtype=torch.long)
        else:
            # Backward-compatible fallback for old caches without prefix_index.npy.
            vfm_idx = (
                torch.linspace(0, T - 1, prefix_len, dtype=torch.float32)
                .round()
                .long()
                .clamp(0, T - 1)
            )
        if self.feat_pixalign:
            vfm_feat = vfm_feat[vfm_idx]
            vfm_idx = torch.arange(vfm_feat.shape[0], device=vfm_feat.device)

        if self.scramble_feat:
            g = torch.Generator()
            seed_key = f"{scene_info['scene_dir']}:prefix:{scene_info['streaming_tail']}:n"
            _seed_n = int(hashlib.md5(seed_key.encode()).hexdigest()[:8], 16) & 0xFFFFFFFF
            g.manual_seed(_seed_n)
            vfm_feat = torch.randn(vfm_feat.shape, generator=g, dtype=vfm_feat.dtype)
        if self.pose_only:
            vfm_feat = torch.zeros_like(vfm_feat)

        return vfm_feat, vfm_idx

    def _load_context_segment_feat(
        self,
        scene_info,
        context_start: int,
        context_tail: int,
        select_indices=None,
    ):
        """Load features from one causal input segment [start, ..., tail].

        The segment was forwarded as a video clip without any target/future
        frames. ``select_indices=None`` returns all frames in the segment,
        approximately aligned to the cached VFM temporal axis.
        """
        scene_dir = self._context_segment_scene_dir(scene_info)
        if scene_dir is None:
            return None, None

        context_start = int(context_start)
        context_tail = int(context_tail)
        segment_dir = os.path.join(
            scene_dir,
            self._context_segment_dir_name(context_start, context_tail),
        )
        path = os.path.join(segment_dir, self._feat_filename())
        if not os.path.exists(path):
            return None, None

        try:
            vfm_feat = load_file(path)["feat"].float()
        except Exception as e:
            logger.warning(f"Failed to read context segment feat {path}: {e}")
            return None, None
        if self.vfm_name == "cogvideox":
            vfm_feat = vfm_feat.reshape(-1, *vfm_feat.shape[2:])
        if vfm_feat.ndim != 4:
            raise ValueError(
                f"Expected context segment VFM feature feat to be 4D (T,H,W,C), "
                f"got {tuple(vfm_feat.shape)} from {path}."
            )

        record = self._context_segment_records(scene_info).get((context_start, context_tail), {})
        input_length = int(record.get("input_length", 0) or 0)
        valid_length = int(record.get("valid_length", context_tail - context_start + 1))
        pad_mode = record.get("pad_mode")

        if select_indices is None:
            select_indices = torch.arange(context_start, context_tail + 1, dtype=torch.long)
        else:
            select_indices = torch.as_tensor(select_indices, dtype=torch.long)
        if select_indices.numel() == 0:
            return torch.zeros(0, *vfm_feat.shape[1:], dtype=vfm_feat.dtype), torch.zeros(0, dtype=torch.long)
        if int(select_indices.min().item()) < context_start or int(select_indices.max().item()) > context_tail:
            raise ValueError(
                f"Context select indices [{int(select_indices.min())}, {int(select_indices.max())}] "
                f"outside segment [{context_start}, {context_tail}]"
            )

        T = vfm_feat.shape[0]
        positions = (select_indices - context_start).float()
        if (
            pad_mode == "repeat_tail"
            and input_length > valid_length
            and (select_indices == context_tail).any()
        ):
            positions = positions.clone()
            positions[select_indices == context_tail] = float(input_length - 1)
        denom = float(max(input_length - 1, valid_length - 1, 1))
        if T > 1:
            vfm_idx = (positions / denom * float(T - 1)).round().long().clamp(0, T - 1)
        else:
            vfm_idx = torch.zeros_like(select_indices)
        if self.feat_pixalign:
            vfm_feat = vfm_feat[vfm_idx]
            vfm_idx = torch.arange(vfm_feat.shape[0], device=vfm_feat.device)

        if self.scramble_feat:
            g = torch.Generator()
            seed_key = f"{scene_info['scene_dir']}:context:{context_start}:{context_tail}:n"
            _seed_n = int(hashlib.md5(seed_key.encode()).hexdigest()[:8], 16) & 0xFFFFFFFF
            g.manual_seed(_seed_n)
            vfm_feat = torch.randn(vfm_feat.shape, generator=g, dtype=vfm_feat.dtype)
        if self.pose_only:
            vfm_feat = torch.zeros_like(vfm_feat)

        return vfm_feat, vfm_idx

    def _getitem_feature_action_diag(self, scene_info, num_frames, sel_global):
        """Fast path for C1/C2/C3 probes that do not consume RGB, masks, or identity IDs."""
        from vidfm3d.utils.spatial_diag import (
            compute_overlap_ratio,
            encode_relative_pose,
        )

        depthmaps, intrinsics, extrinsics = self._load_camera_depth_scene(
            scene_info, sel_global
        )
        depthmaps, intrinsics = self._resize_depth_to_target(depthmaps, intrinsics)
        confmaps = (depthmaps > 0).float()

        pointmaps = torch.stack([
            depth_to_pointmap(depthmaps[i], intrinsics[i], extrinsics[i])
            for i in range(depthmaps.shape[0])
        ])
        extrinsics, pointmaps_scaled, depthmaps_hw1 = invert_pose_ref_and_scale(
            extrinsics,
            pointmaps,
            depthmaps=depthmaps.unsqueeze(-1),
            ref_idx=0,
            scale_by_points=True,
        )
        depthmaps = depthmaps_hw1.squeeze(-1)

        target_global = sel_global[1:]
        K = int(target_global.numel())

        if getattr(self, "streaming_prefix", False):
            prefix_global = torch.as_tensor(
                scene_info["streaming_indices"], dtype=torch.long
            )
            input_feat, vfm_idx = self._load_streaming_prefix_feat(
                scene_info, prefix_global
            )
            output_vfm_feat = input_feat
            input_valid = True
        else:
            vf, vfm_idx = self._load_vfm_feat_for_selection(
                scene_info, num_frames, sel_global
            )
            output_vfm_feat = vf
            context_tail = int(sel_global[0].item())
            context_start = max(
                int(scene_info.get("window_start", 0)),
                context_tail - int(self.context_len) + 1,
            )
            input_feat, _ = self._load_context_segment_feat(
                scene_info,
                context_start,
                context_tail,
                select_indices=self._context_select_indices(context_start, context_tail),
            )
            input_valid = input_feat is not None
            if not input_valid:
                input_feat = torch.zeros(int(self.context_len), *vf.shape[1:], dtype=vf.dtype)

        loaded_target_feats = self._load_target_feats(
            scene_info,
            [int(target_idx.item()) for target_idx in target_global],
            num_frames,
        )
        target_feats = []
        target_valid = []
        for target_feat in loaded_target_feats:
            if target_feat is None:
                target_feats.append(torch.zeros_like(input_feat[0]))
                target_valid.append(False)
            else:
                target_feats.append(target_feat)
                target_valid.append(True)

        if getattr(self, "diag_action", False):
            target_feat = target_feats[0] if target_feats else torch.zeros_like(input_feat[0])
            target_ok = bool(target_valid[0]) if target_valid else False
            return {
                "vfm_feat": output_vfm_feat,
                "vfm_idx": vfm_idx,
                "input_feat": input_feat.clone(),
                "target_feat": target_feat,
                "action": encode_relative_pose(extrinsics[0], extrinsics[1]).cpu()
                if K > 0 else torch.zeros(9, dtype=torch.float32),
                "target_frame_idx": target_global[0].to(torch.long).cpu()
                if K > 0 else torch.tensor(-1, dtype=torch.long),
                "dyn_valid": torch.tensor(bool(input_valid and target_ok), dtype=torch.bool),
                "rng": int.from_bytes(self._rng.bytes(4), "big"),
                "scene_path": scene_info["scene_dir"],
                "vfm_name": self.vfm_name,
            }

        if K > 0:
            target_feat_seq = torch.stack(target_feats)
            horizon_valid = (
                torch.as_tensor(target_valid, dtype=torch.bool)
                & bool(input_valid)
            )
            path_actions = torch.stack([
                encode_relative_pose(extrinsics[i], extrinsics[i + 1]).cpu()
                for i in range(K)
            ])
            counterfactual_actions = torch.stack([
                encode_relative_pose(extrinsics[0], extrinsics[i + 1]).cpu()
                for i in range(K)
            ])
            horizons = (target_global - sel_global[0]).to(torch.long).cpu()
        else:
            target_feat_seq = torch.zeros(0, *input_feat.shape[1:], dtype=input_feat.dtype)
            horizon_valid = torch.zeros(0, dtype=torch.bool)
            path_actions = torch.zeros(0, 9, dtype=torch.float32)
            counterfactual_actions = torch.zeros(0, 9, dtype=torch.float32)
            horizons = torch.zeros(0, dtype=torch.long)

        overlap_anchor = torch.ones(K, dtype=torch.float32)
        if self.diag_counterfactual and K > 0:
            overlap_full = compute_overlap_ratio(
                pointmaps_scaled,
                intrinsics,
                extrinsics,
                depthmaps > 0,
            )
            overlap_anchor = overlap_full[0, 1:].cpu()

        counterfactual_valid = horizon_valid & (
            overlap_anchor >= float(self.counterfactual_min_overlap)
        )

        return {
            "vfm_feat": output_vfm_feat,
            "vfm_idx": vfm_idx,
            "input_feat_seq": input_feat.clone(),
            "target_feat_seq": target_feat_seq,
            "path_actions": path_actions,
            "counterfactual_actions": counterfactual_actions,
            "target_extrinsics": extrinsics[1:].cpu(),
            "start_extrinsic": extrinsics[0].cpu(),
            "action_horizons": horizons,
            "counterfactual_overlap": overlap_anchor,
            "path_horizon_valid": horizon_valid,
            "path_valid": torch.tensor(bool(horizon_valid.all().item()), dtype=torch.bool),
            "counterfactual_valid": counterfactual_valid,
            "rng": int.from_bytes(self._rng.bytes(4), "big"),
            "scene_path": scene_info["scene_dir"],
            "vfm_name": self.vfm_name,
        }

    def _load_shuffled_feat(self, scene_info, num_frames, sel_global):
        """Load shuffled-context vfm features for A3.

        Expected layout mirrors the standard feature layout:
            {shuffled_feat_root}/{vfm}/{source}/{scene}/feature{feat_postfix}.sft
        with feat shape (T, H_f, W_f, C) indexed the same way as the original
        (i.e. shuffled[i] = feature of frame i extracted under scrambled context).
        """
        if self.shuffled_feat_root is None:
            return None
        path = os.path.join(
            self.shuffled_feat_root,
            self.vfm_name,
            scene_info["source"],
            self._feat_scene_name(scene_info),
            self._feat_filename(),
        )
        if not os.path.exists(path):
            return None
        try:
            shuf = load_file(path)["feat"].float()
        except Exception as e:
            logger.warning(f"Failed to read shuffled feat {path}: {e}")
            return None
        if self.vfm_name == "cogvideox":
            shuf = shuf.reshape(-1, *shuf.shape[2:])
        T = shuf.shape[0]
        vfm_idx = (
            torch.round(sel_global.float() / max(num_frames - 1, 1) * (T - 1))
            .long().clamp(0, T - 1)
        )
        if self.feat_pixalign:
            shuf = shuf[vfm_idx]
        return shuf

    def __getitem__(self, idx):
        if self.seed is not None:
            self._rng = np.random.default_rng(seed=self.seed + idx)
        elif not hasattr(self, "_rng"):
            seed = torch.initial_seed()
            self._rng = np.random.default_rng(seed=seed)

        scene_info = self.scenes[idx]
        num_frames = scene_info["num_frames"]

        if self.streaming_prefix:
            prefix_global = torch.as_tensor(
                scene_info["streaming_indices"], dtype=torch.long
            )
            future_horizons = self._streaming_future_horizons(default=[])
            if (
                future_horizons
                and (self.diag_action or self.diag_path_integration or self.diag_counterfactual)
                and not (self.diag_overlap or self.diag_hidden_obj or self.diag_abnormal)
            ):
                tail = int(scene_info["streaming_tail"])
                sel_global = torch.as_tensor(
                    [tail] + [tail + int(h) for h in future_horizons],
                    dtype=torch.long,
                )
            else:
                sel_global = prefix_global
        else:
            # Window bounds (defaults to full scene)
            window_start = scene_info.get("window_start", 0)
            window_end = scene_info.get("window_end", num_frames)
            window_frames = window_end - window_start

            # Sample frame indices within the window. Multi-action diagnostic probes
            # use explicit offsets from one action-reference frame so horizons
            # have a stable interpretation: [t, t+1, t+10, t+30, ...].
            explicit_action_horizons = (
                (self.diag_path_integration or self.diag_counterfactual)
                and len(self.action_horizons) > 0
                and window_frames > max(self.action_horizons)
            )
            if explicit_action_horizons:
                max_start = min(window_frames, self.context_len) - max(self.action_horizons)
                max_start = max(max_start, 1)
                action_ref = int(self._rng.integers(0, max_start))
                sel = torch.as_tensor(
                    [action_ref] + [action_ref + h for h in self.action_horizons],
                    dtype=torch.long,
                )
            else:
                sample_range = min(window_frames, self.context_len)
                sel = self._sample_query_frames(self._rng, self.num_views, sample_range)

            if self.query_idx_divisor is not None and not explicit_action_horizons:
                sel = (
                    torch.floor((sel - 1) / self.query_idx_divisor) * self.query_idx_divisor
                    + 1
                )
                sel = sel.clamp(min=0, max=window_frames - 1).long()

            # Convert to global frame indices for loading
            sel_global = sel + window_start

            # C1 target-isolated caches may contain a sparse set of frames.
            # Align only the target before loading geometry. Input features are
            # loaded from a causal context_segment cache ending at sel_global[-2].
            if self.diag_action:
                cached_targets = self._target_indices(scene_info)
                if cached_targets is not None and cached_targets.numel() > 0:
                    candidates = cached_targets[
                        (cached_targets >= window_start)
                        & (cached_targets < window_end)
                    ].sort().values
                    target_candidates = candidates[candidates > sel_global[-2]]
                    if target_candidates.numel() > 0:
                        nearest = (target_candidates - sel_global[-1]).abs().argmin()
                        target = target_candidates[nearest]
                        sel_global[-1] = target

        if (
            (self.diag_action or self.diag_path_integration or self.diag_counterfactual)
            and not (self.diag_overlap or self.diag_hidden_obj or self.diag_abnormal)
            and (self.streaming_prefix or not self.diag_action)
        ):
            return self._getitem_feature_action_diag(scene_info, num_frames, sel_global)

        # Load data based on source
        if scene_info["source"] == "infinigen":
            images, masks, depthmaps, intrinsics, extrinsics = (
                self._load_infinigen_scene(scene_info, sel_global)
            )
        elif scene_info["source"] == "scannetpp":
            images, masks, depthmaps, intrinsics, extrinsics = (
                self._load_scannetpp_scene(scene_info, sel_global)
            )
        else:
            raise ValueError(f"Unknown source: {scene_info['source']}")

        # Resize to target resolution (also rescales intrinsics)
        images, masks, depthmaps, intrinsics = self._resize_to_target(
            images, masks, depthmaps, intrinsics
        )

        # Normalize images
        images = images / 255.0

        # Build confidence maps from depth validity (1 where depth > 0)
        confmaps = (depthmaps > 0).float()  # (S, H, W)

        # Compute world-coordinate point maps only when needed (v3 can disable this).
        if self.include_pmaps:
            S, _, H, W = images.shape
            pointmaps = []
            for i in range(S):
                pts = depth_to_pointmap(depthmaps[i], intrinsics[i], extrinsics[i])
                pointmaps.append(pts)
            pointmaps = torch.stack(pointmaps)  # (S, H, W, 3)
        else:
            pointmaps = None

        # Normalize scene. Streaming prefix uses the current tail frame as ego;
        # standard sampled clips keep the historical first-frame reference.
        # If point maps are disabled, scale by depth instead of points.
        depthmaps_metric = depthmaps.clone() if self.streaming_prefix else None
        ref_idx = len(sel_global) - 1 if self.streaming_prefix else 0
        extrinsics, pointmaps_scaled, depthmaps_hw1 = invert_pose_ref_and_scale(
            extrinsics,                        # (S, 3, 4)
            pointmaps,
            depthmaps=depthmaps.unsqueeze(-1),  # (S, H, W, 1)
            ref_idx=ref_idx,
            scale_by_points=self.include_pmaps,
        )
        if self.streaming_prefix:
            # The online protocol rebases camera pose to the current tail frame
            # but leaves per-frame depth scalars in their original metric form.
            depthmaps = depthmaps_metric
        else:
            depthmaps = depthmaps_hw1.squeeze(-1)  # (S, H, W)

        # Keep raw ids for streaming B shared-object selection. Remapped ids are
        # still used by the normal pixel/identity outputs.
        raw_identity_ids = masks.clone()

        # Remap identity IDs
        masks = self._remap_identity_ids(masks)

        # Prepare output
        output = {}
        output["image"] = images  # (S, 3, H, W)
        output["identity_ids"] = masks  # (S, H, W) - integer identity IDs
        output["intrinsics"] = intrinsics  # (S, 3, 3)
        output["extrinsics"] = extrinsics  # (S, 3, 4)
        output["cmaps"] = confmaps.unsqueeze(1)  # (S, 1, H, W)
        output["dmaps"] = depthmaps.unsqueeze(1)  # (S, 1, H, W)
        if self.include_pmaps and pointmaps_scaled is not None:
            output["pmaps"] = pointmaps_scaled.permute(0, 3, 1, 2)  # (S, 3, H, W)

        # VFM features
        if self.streaming_prefix:
            vfm_feat, vfm_idx = self._load_streaming_prefix_feat(
                scene_info, sel_global
            )
            output["vfm_feat"] = vfm_feat
            output["vfm_idx"] = vfm_idx
        elif self.root_vfm is not None:
            source_name = scene_info["source"]
            feat_scene_name = self._feat_scene_name(scene_info)
            vfm_feat_path = os.path.join(
                self.root_vfm,
                self.vfm_name,
                source_name,
                feat_scene_name,
                self._feat_filename(),
            )
            if os.path.exists(vfm_feat_path):
                # NOTE: cast to fp32 to match `_load_shuffled_feat` which does
                # `.float()`. A dtype mismatch (fp16 vs fp32) between the two
                # branches lets a probe trivially detect the dtype quantization
                # signature instead of using content — see A3 ctrl dtype-leak
                # finding in PROGRESS_REPORT.md.
                vfm_feat = load_file(vfm_feat_path)["feat"].float()
                if self.vfm_name == "cogvideox":
                    # CogVideoX: (2, T_clip, H, W, C) → merge clips → (2*T_clip, H, W, C)
                    vfm_feat = vfm_feat.reshape(-1, *vfm_feat.shape[2:])

                if vfm_feat.ndim != 4:
                    raise ValueError(
                        f"Expected VFM feature feat to be 4D (T,H,W,C), got {tuple(vfm_feat.shape)} "
                        f"from {vfm_feat_path}. For token models, save activations as a token grid."
                    )

                T = vfm_feat.shape[0]
                vfm_idx = torch.round(
                    sel_global.float() / max(num_frames - 1, 1) * (T - 1)
                ).long().clamp(0, T - 1)
                if self.feat_pixalign:
                    vfm_feat = vfm_feat[vfm_idx]
                    vfm_idx = torch.arange(
                        vfm_feat.shape[0], device=vfm_feat.device
                    )
                if self.scramble_feat:
                    # Control experiment: replace with unit-normal noise (same shape).
                    # Seed on scene_dir (not idx) so train/val index overlap does not let
                    # the probe memorise noise patterns across splits. See A3 ctrl seed-leak
                    # finding in PROGRESS_REPORT.md.
                    g = torch.Generator()
                    _seed_n = int(hashlib.md5((scene_info["scene_dir"] + ":n").encode()).hexdigest()[:8], 16) & 0xFFFFFFFF
                    g.manual_seed(_seed_n)
                    vfm_feat = torch.randn(vfm_feat.shape, generator=g, dtype=vfm_feat.dtype)
                if self.pose_only:
                    # Unconditional ablation: remove all visual feature signal.
                    vfm_feat = torch.zeros_like(vfm_feat)
                output["vfm_feat"] = vfm_feat
                output["vfm_idx"] = vfm_idx
            else:
                msg = f"VFM feature not found at {vfm_feat_path}"
                if not self.allow_missing_vfm:
                    raise FileNotFoundError(
                        msg + "; run feature extraction first or set allow_missing_vfm=True for debugging"
                    )
                logger.warning(f"{msg}, using dummy because allow_missing_vfm=True.")
                dummy_channels = default_feature_channels(self.vfm_name)
                output["vfm_feat"] = torch.zeros(
                    self.num_views, 18, 32, dummy_channels, dtype=torch.float32
                )
                output["vfm_idx"] = torch.arange(self.num_views)
        else:
            # Dummy features for testing
            output["vfm_feat"] = torch.zeros(
                self.num_views, 18, 32, 1536, dtype=torch.float32
            )
            output["vfm_idx"] = torch.arange(self.num_views)

        output["rng"] = int.from_bytes(self._rng.bytes(4), "big")
        output["scene_path"] = scene_info["scene_dir"]
        output["vfm_name"] = self.vfm_name
        if self.streaming_prefix:
            output["streaming_tail"] = torch.tensor(
                int(scene_info["streaming_tail"]), dtype=torch.long
            )
            output["prefix_indices"] = sel_global.cpu()

        # ------------------------------------------------------------ Spatial Diagnostic Suite
        # NOTE: All diagnostic outputs use post-normalization extrinsics + pmaps_scaled.
        # The geometry stays metrically consistent (rigid transform + uniform scale).
        if (
            self.diag_overlap
            or self.diag_hidden_obj
            or self.diag_action
            or self.diag_path_integration
            or self.diag_counterfactual
        ):
            from vidfm3d.utils.spatial_diag import (
                compute_overlap_ratio,
                compute_hidden_object_target,
                compute_object_target_for_id,
                encode_relative_pose,
            )

        if self.diag_overlap:
            assert pointmaps_scaled is not None, "diag_overlap requires include_pmaps=True"
            valid_mask = (depthmaps > 0)                                 # (S, H, W)
            output["overlap_gt"] = compute_overlap_ratio(
                pointmaps_scaled, intrinsics, extrinsics, valid_mask
            )                                                            # (S, S)

        if self.diag_hidden_obj:
            assert pointmaps_scaled is not None, "diag_hidden_obj requires include_pmaps=True"
            if self._use_shared_streaming_hidden_obj():
                obj_id = self._shared_streaming_hidden_obj_id(scene_info)
                target = None if obj_id is None else compute_object_target_for_id(
                    obj_id=obj_id,
                    identity_ids=raw_identity_ids,
                    pmaps_world=pointmaps_scaled,
                    confmaps=confmaps,
                    extrinsics=extrinsics,
                    min_visible_pixels=self.hidden_obj_min_visible_pixels,
                    last_frame_idx=-1,
                    require_hidden=True,
                )
            else:
                target = compute_hidden_object_target(
                    identity_ids=masks,
                    pmaps_world=pointmaps_scaled,
                    confmaps=confmaps,
                    extrinsics=extrinsics,
                    min_visible_pixels=self.hidden_obj_min_visible_pixels,
                    last_frame_idx=-1,
                )
            S, H, W = masks.shape
            # ------------------------------------------------------------------ B2 query token
            # For B2 we emit a 1D appearance signature of the chosen object,
            # masked-pooled from the past frame with the most visible pixels.
            # The mask itself and camera pose are not passed to B2. The pooled
            # backbone feature can still carry spatial/positional information.
            vfm_feat_cur = output["vfm_feat"]   # (S, H_f, W_f, C) -- post scramble/pose_only
            C = vfm_feat_cur.shape[-1]

            if target is None:
                output["hidden_obj_valid"] = torch.zeros((), dtype=torch.bool)
                output["hidden_obj_polar"] = torch.zeros(3, dtype=torch.float32)
                output["hidden_obj_mask"] = torch.zeros(S, H, W, dtype=torch.bool)
                output["hidden_obj_id"] = torch.tensor(-1, dtype=torch.long)
                output["belief_query_feat"] = torch.zeros(C, dtype=vfm_feat_cur.dtype)
                output["belief_query_frame"] = torch.tensor(-1, dtype=torch.long)
            else:
                output["hidden_obj_valid"] = target["valid"].cpu()
                output["hidden_obj_polar"] = target["polar"].cpu()
                per_frame_mask = target["per_frame_mask"].cpu()
                output["hidden_obj_id"] = target["obj_id"].cpu()

                # B2: build the object query token BEFORE any mask ablation,
                # using the past frame with the most masked pixels.
                pix_per_frame = per_frame_mask.flatten(1).sum(dim=1)      # (S,)
                best_frame = int(pix_per_frame.argmax().item())
                H_f, W_f = vfm_feat_cur.shape[1], vfm_feat_cur.shape[2]
                m_best = per_frame_mask[best_frame].unsqueeze(0).unsqueeze(0).float()
                m_feat = torch.nn.functional.interpolate(
                    m_best, size=(H_f, W_f), mode="nearest"
                ).squeeze(0).squeeze(0) > 0.5                              # (H_f, W_f)
                denom = m_feat.float().sum().clamp(min=1.0)
                query_feat = (vfm_feat_cur[best_frame] * m_feat.unsqueeze(-1)).sum(
                    dim=(0, 1)
                ) / denom                                                  # (C,)
                output["belief_query_feat"] = query_feat
                output["belief_query_frame"] = torch.tensor(best_frame, dtype=torch.long)

                if self.no_obj_mask:
                    per_frame_mask = torch.ones_like(per_frame_mask)  # B1 ablation only
                output["hidden_obj_mask"] = per_frame_mask
        if self.diag_action:
            # C1 input is a causal video segment forward that ends before the
            # target frame. Target supervision remains an isolated exact-frame
            # feature.
            context_tail = int(sel_global[-2].item())
            context_start = max(
                int(scene_info.get("window_start", 0)),
                context_tail - int(self.context_len) + 1,
            )
            input_feat, _ = self._load_context_segment_feat(
                scene_info,
                context_start,
                context_tail,
                select_indices=self._context_select_indices(context_start, context_tail),
            )
            target_feat = self._load_target_feat(
                scene_info, sel_global[-1].item(), num_frames
            )
            input_valid = input_feat is not None
            if target_feat is not None and input_valid:
                output["target_feat"] = target_feat                       # (H_f, W_f, C)
                output["input_feat"] = input_feat.clone()                  # (T_ctx, H_f, W_f, C)
                output["action"] = encode_relative_pose(
                    extrinsics[-2], extrinsics[-1]
                ).cpu()
                output["target_frame_idx"] = sel_global[-1].to(torch.long).cpu()
                output["dyn_valid"] = torch.ones((), dtype=torch.bool)
            else:
                # Fall back: emit zero-valued tensors so collation works, mark invalid
                vf = output["vfm_feat"]
                output["target_feat"] = torch.zeros_like(vf[0])
                output["input_feat"] = torch.zeros(int(self.context_len), *vf.shape[1:], dtype=vf.dtype)
                output["action"] = torch.zeros(9, dtype=torch.float32)
                output["target_frame_idx"] = torch.tensor(-1, dtype=torch.long)
                output["dyn_valid"] = torch.zeros((), dtype=torch.bool)

        if self.diag_path_integration or self.diag_counterfactual:
            # C2/C3 input is a causal video segment forward that ends at the
            # action-reference frame. Target horizons remain isolated
            # exact-frame features.
            vf = output["vfm_feat"]
            K = max(vf.shape[0] - 1, 0)
            context_tail = int(sel_global[0].item())
            context_start = max(
                int(scene_info.get("window_start", 0)),
                context_tail - int(self.context_len) + 1,
            )
            input_feat, _ = self._load_context_segment_feat(
                scene_info,
                context_start,
                context_tail,
                select_indices=self._context_select_indices(context_start, context_tail),
            )
            input_valid = input_feat is not None
            if not input_valid:
                input_feat = torch.zeros(int(self.context_len), *vf.shape[1:], dtype=vf.dtype)
            target_feats = []
            target_valid = []
            loaded_target_feats = self._load_target_feats(
                scene_info,
                [int(target_idx.item()) for target_idx in sel_global[1:]],
                num_frames,
            )
            for target_feat in loaded_target_feats:
                if target_feat is None:
                    target_feats.append(torch.zeros_like(vf[0]))
                    target_valid.append(False)
                else:
                    target_feats.append(target_feat)
                    target_valid.append(True)

            if K > 0:
                target_feat_seq = torch.stack(target_feats)
                horizon_valid = (
                    torch.as_tensor(target_valid, dtype=torch.bool)
                    & bool(input_valid)
                )
                path_actions = torch.stack([
                    encode_relative_pose(extrinsics[i], extrinsics[i + 1]).cpu()
                    for i in range(K)
                ])
                counterfactual_actions = torch.stack([
                    encode_relative_pose(extrinsics[0], extrinsics[i + 1]).cpu()
                    for i in range(K)
                ])
                horizons = (sel_global[1:] - sel_global[0]).to(torch.long).cpu()
            else:
                target_feat_seq = torch.zeros(0, *vf.shape[1:], dtype=vf.dtype)
                horizon_valid = torch.zeros(0, dtype=torch.bool)
                path_actions = torch.zeros(0, 9, dtype=torch.float32)
                counterfactual_actions = torch.zeros(0, 9, dtype=torch.float32)
                horizons = torch.zeros(0, dtype=torch.long)

            overlap_anchor = torch.ones(K, dtype=torch.float32)
            if pointmaps_scaled is not None and K > 0:
                valid_mask = (depthmaps > 0)
                overlap_full = compute_overlap_ratio(
                    pointmaps_scaled, intrinsics, extrinsics, valid_mask
                )
                overlap_anchor = overlap_full[0, 1:].cpu()

            counterfactual_valid = horizon_valid & (
                overlap_anchor >= float(self.counterfactual_min_overlap)
            )

            output["input_feat_seq"] = input_feat.clone()
            output["target_feat_seq"] = target_feat_seq
            output["path_actions"] = path_actions
            output["counterfactual_actions"] = counterfactual_actions
            output["target_extrinsics"] = extrinsics[1:].cpu()
            output["start_extrinsic"] = extrinsics[0].cpu()
            output["action_horizons"] = horizons
            output["counterfactual_overlap"] = overlap_anchor
            output["path_horizon_valid"] = horizon_valid
            output["path_valid"] = torch.tensor(bool(horizon_valid.all().item()), dtype=torch.bool)
            output["counterfactual_valid"] = counterfactual_valid

        if self.diag_abnormal:
            shuf = self._load_shuffled_feat(scene_info, num_frames, sel_global)
            output["vfm_feat_shuffled"] = (
                shuf if shuf is not None else torch.zeros_like(output["vfm_feat"])
            )
            # Control consistency: if scramble_feat is on, the "normal" branch
            # (vfm_feat) was replaced by N(0,1) noise above. The shuffled branch
            # must be scrambled too, otherwise the probe trivially distinguishes
            # noise-vs-real-features and gets ~100% acc without using time order.
            if self.scramble_feat and shuf is not None:
                g = torch.Generator()
                _seed_s = int(hashlib.md5((scene_info["scene_dir"] + ":s").encode()).hexdigest()[:8], 16) & 0xFFFFFFFF
                g.manual_seed(_seed_s)
                output["vfm_feat_shuffled"] = torch.randn(
                    output["vfm_feat_shuffled"].shape,
                    generator=g,
                    dtype=output["vfm_feat_shuffled"].dtype,
                )
            output["abnormal_feat_valid"] = torch.tensor(
                shuf is not None, dtype=torch.bool
            )

        return output
