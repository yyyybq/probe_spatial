"""
InsScene-15K dataset loader for probe training (identity, depth, point, camera).

Supports two data sources that have identity masks + depth + camera:
  - processed_infinigen: ObjectSegmentation/*.npy, Depth/*.npy, camview/*.npz
  - processed_scannetpp_v2: refined_ins_ids/*.npy, depth/*.png, metadata.npz

Point maps are computed from depth + intrinsics + extrinsics via back-projection.
"""

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
        feat_pixalign: bool = True,
        seed: int = None,
        num_views: int = 4,
        min_view_interval: int = 5,
        context_len: int = 76,
        query_idx_divisor: int = 4,
        target_h: int = 288,
        target_w: int = 512,
        train_ratio: float = 0.9,
        max_identity_classes: int = 256,
        window_size: int = 0,
        include_pmaps: bool = True,
        # ---------- Spatial Diagnostic Suite toggles ----------
        diag_overlap: bool = False,        # A2: compute (S,S) overlap matrix
        diag_hidden_obj: bool = False,     # B1: pick hidden object + polar target
        diag_action: bool = False,         # C1: load target_feat + emit action
        diag_abnormal: bool = False,       # A3: load shuffled vfm features
        target_feat_root: str = None,      # root for C1 target features
        shuffled_feat_root: str = None,    # root for A3 shuffled features
        scramble_feat: bool = False,       # Control: replace VFM feat with randn (same shape)
        no_obj_mask: bool = False,         # Ablation: replace obj mask with all-ones (global pool)
        pose_only: bool = False,           # Ablation: zero out vfm_feat (only last_pose_enc survives)
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
        if sources is None:
            sources = ["processed_infinigen", "processed_scannetpp_v2"]

        self.root = root
        self.root_vfm = root_vfm
        self.sources = sources
        self.split = split
        self.vfm_name = vfm_name
        self.feat_postfix = feat_postfix
        self.feat_pixalign = feat_pixalign
        self.seed = seed
        self.num_views = num_views
        self.min_view_interval = min_view_interval
        self.context_len = context_len
        self.query_idx_divisor = query_idx_divisor
        self.target_h = target_h
        self.target_w = target_w
        self.max_identity_classes = max_identity_classes
        self.window_size = window_size
        self.include_pmaps = include_pmaps
        # Diagnostic suite flags
        self.diag_overlap = diag_overlap
        self.diag_hidden_obj = diag_hidden_obj
        self.diag_action = diag_action
        self.diag_abnormal = diag_abnormal
        self.target_feat_root = target_feat_root
        self.shuffled_feat_root = shuffled_feat_root
        self.scramble_feat = scramble_feat
        self.no_obj_mask = no_obj_mask
        self.pose_only = pose_only
        # A2/B1/C1 all need extrinsics normalized via pointmap-based scale
        # (invert_pose_ref_and_scale requires pointmaps).
        if (diag_overlap or diag_hidden_obj or diag_action) and not include_pmaps:
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

        # Split scenes
        rng = np.random.default_rng(seed=42)
        indices = rng.permutation(len(self.scenes))
        split_idx = int(len(self.scenes) * train_ratio)
        if split == "train":
            self.scenes = [self.scenes[i] for i in indices[:split_idx]]
        else:
            self.scenes = [self.scenes[i] for i in indices[split_idx:]]

        # Expand scenes with windowing for long videos
        if self.window_size > 0:
            self._expand_scenes_with_windows()

        logger.info(
            f"InsScene15KDataset: {len(self.scenes)} samples for {split} split "
            f"from sources {sources} (window_size={self.window_size})"
        )

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
        Whichever the case, ``feature{feat_postfix}.sft`` is the right pattern as long
        as feat_postfix matches the extractor's choice. We keep that single source of
        truth here.
        """
        if self.vfm_name == "vjepa":
            return "feature.sft"
        return f"feature{self.feat_postfix}.sft"

    def _load_target_feat(self, scene_info, target_global_idx, num_frames):
        """Load a clip-isolated target feature for C1.

        Expected file layout:
            {target_feat_root}/{vfm}/{source}/{scene}/feature{feat_postfix}.sft
        which contains:
            feat:           (M, H_f, W_f, C)   — features for M target frames
            target_indices: (M,) long          — original frame indices

        Returns the feature for the closest target index to target_global_idx,
        or None if missing.
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
        # Find the metadata for target_indices
        meta_path = os.path.join(os.path.dirname(path), "target_indices.npy")
        if os.path.exists(meta_path):
            tgt_idx = torch.from_numpy(np.load(meta_path)).long()
            # Closest match in the precomputed grid
            j = (tgt_idx - target_global_idx).abs().argmin().item()
        else:
            # Fall back: assume evenly spaced
            j = min(feat.shape[0] - 1, int(round(target_global_idx / max(num_frames - 1, 1) * (feat.shape[0] - 1))))
        return feat[j]

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
        if self.seed:
            self._rng = np.random.default_rng(seed=self.seed + idx)
        elif not hasattr(self, "_rng"):
            seed = torch.initial_seed()
            self._rng = np.random.default_rng(seed=seed)

        scene_info = self.scenes[idx]
        num_frames = scene_info["num_frames"]

        # Window bounds (defaults to full scene)
        window_start = scene_info.get("window_start", 0)
        window_end = scene_info.get("window_end", num_frames)
        window_frames = window_end - window_start

        # Sample frame indices within the window
        sample_range = min(window_frames, self.context_len)
        sel = self._sample_query_frames(self._rng, self.num_views, sample_range)

        if self.query_idx_divisor is not None:
            sel = (
                torch.floor((sel - 1) / self.query_idx_divisor) * self.query_idx_divisor
                + 1
            )
            sel = sel.clamp(min=0, max=window_frames - 1).long()

        # Convert to global frame indices for loading
        sel_global = sel + window_start

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

        # Normalize scene: extrinsics relative to first frame.
        # If point maps are disabled, scale by depth instead of points.
        extrinsics, pointmaps_scaled, depthmaps_hw1 = invert_pose_ref_and_scale(
            extrinsics,                        # (S, 3, 4)
            pointmaps,
            depthmaps=depthmaps.unsqueeze(-1),  # (S, H, W, 1)
            ref_idx=0,
            scale_by_points=self.include_pmaps,
        )
        depthmaps = depthmaps_hw1.squeeze(-1)  # (S, H, W)

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
        if self.root_vfm is not None:
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
                vfm_feat = load_file(vfm_feat_path)["feat"]
                if self.vfm_name in ["wan", "opensora", "vjepa2"]:
                    T = vfm_feat.shape[0]
                    vfm_idx = torch.round(
                        sel_global.float() / max(num_frames - 1, 1) * (T - 1)
                    ).long().clamp(0, T - 1)
                    if self.feat_pixalign:
                        vfm_feat = vfm_feat[vfm_idx]
                        vfm_idx = torch.arange(
                            vfm_feat.shape[0], device=vfm_feat.device
                        )
                elif self.vfm_name == "cogvideox":
                    # CogVideoX: (2, T_clip, H, W, C) → merge clips → (2*T_clip, H, W, C)
                    vfm_feat = vfm_feat.reshape(-1, *vfm_feat.shape[2:])
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
                    # Use a deterministic seed so the same sample always gets the same noise.
                    g = torch.Generator()
                    g.manual_seed(hash((int(idx), 0)) & 0xFFFFFFFF)
                    vfm_feat = torch.randn(vfm_feat.shape, generator=g, dtype=vfm_feat.dtype)
                if self.pose_only:
                    # Ablation: zero out VFM feat so only last_pose_enc survives in B1.
                    vfm_feat = torch.zeros_like(vfm_feat)
                output["vfm_feat"] = vfm_feat
                output["vfm_idx"] = vfm_idx
            else:
                logger.warning(
                    f"VFM feature not found at {vfm_feat_path}, using dummy."
                )
                output["vfm_feat"] = torch.zeros(
                    self.num_views, 18, 32, 1536, dtype=torch.float32
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

        # ------------------------------------------------------------ Spatial Diagnostic Suite
        # NOTE: All diagnostic outputs use post-normalization extrinsics + pmaps_scaled.
        # The geometry stays metrically consistent (rigid transform + uniform scale).
        if self.diag_overlap or self.diag_hidden_obj or self.diag_action:
            from vidfm3d.utils.spatial_diag import (
                compute_overlap_ratio,
                compute_hidden_object_target,
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
            target = compute_hidden_object_target(
                identity_ids=masks,
                pmaps_world=pointmaps_scaled,
                confmaps=confmaps,
                extrinsics=extrinsics,
                last_frame_idx=-1,
            )
            S, H, W = masks.shape
            # ------------------------------------------------------------------ B2 query token
            # For B2 we emit a 1D appearance signature of the chosen object,
            # masked-pooled from the past frame with the most visible pixels.
            # No spatial / pose / mask info is exposed to the probe.
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

                # B2: build the appearance-only query token BEFORE any mask ablation,
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
            # Encode last frame's pose as a 9-dim vector for conditioning
            last_extr = extrinsics[-1]
            output["last_pose_enc"] = encode_relative_pose(
                torch.eye(3, 4, dtype=last_extr.dtype), last_extr
            ).cpu()

        if self.diag_action:
            # Input frames = first S-1 of vfm_feat already loaded; target frame = last view.
            # For action_dynamics we need an UNCONTAMINATED target_feat (separate forward).
            target_feat = self._load_target_feat(
                scene_info, sel_global[-1].item(), num_frames
            )
            if target_feat is not None:
                output["target_feat"] = target_feat                       # (H_f, W_f, C)
                output["input_feat"] = output["vfm_feat"][:-1].clone()     # (S-1, H_f, W_f, C)
                output["action"] = encode_relative_pose(
                    extrinsics[-2], extrinsics[-1]
                ).cpu()
                output["dyn_valid"] = torch.ones((), dtype=torch.bool)
            else:
                # Fall back: emit zero-valued tensors so collation works, mark invalid
                vf = output["vfm_feat"]
                output["target_feat"] = torch.zeros_like(vf[0])
                output["input_feat"] = torch.zeros_like(vf[:-1])
                output["action"] = torch.zeros(9, dtype=torch.float32)
                output["dyn_valid"] = torch.zeros((), dtype=torch.bool)

        if self.diag_abnormal:
            shuf = self._load_shuffled_feat(scene_info, num_frames, sel_global)
            output["vfm_feat_shuffled"] = (
                shuf if shuf is not None else torch.zeros_like(output["vfm_feat"])
            )
            output["abnormal_feat_valid"] = torch.tensor(
                shuf is not None, dtype=torch.bool
            )

        return output
