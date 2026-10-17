"""Geometry / encoding utilities for the Spatial Representation Diagnostic Suite.

This module collects the geometry helpers shared by the four diagnostic probes:
    A2  view consistency       -> compute_overlap_ratio
    A3  abnormal video         -> (uses dataset-level shuffled features only)
    B1  ego-centric belief     -> compute_hidden_object_target
    C1  action-conditioned     -> encode_relative_pose

All functions are pure-tensor / pure-numpy and do not depend on Lightning or hydra.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import torch


# ---------------------------------------------------------------------------
# A2: View consistency overlap
# ---------------------------------------------------------------------------
def compute_overlap_ratio(
    pmaps_world: torch.Tensor,        # (S, H, W, 3) -- per-pixel 3D points in world coords
    intrinsics: torch.Tensor,         # (S, 3, 3)
    extrinsics: torch.Tensor,         # (S, 3, 4) -- world->camera
    valid_mask: Optional[torch.Tensor] = None,  # (S, H, W) bool, valid depth pixels
    min_depth: float = 1e-3,
) -> torch.Tensor:
    """Compute pairwise visual overlap ratios for a batch of S frames of one scene.

    overlap(i -> j) is defined as: fraction of valid pixels of frame i that, when
    projected into frame j using intrinsics_j, extrinsics_j, fall inside frame j's
    image bounds AND have positive depth in frame j's camera coords.

    Returns a symmetric overlap matrix (S, S): overlap_sym[i, j] = min(o(i->j), o(j->i)).
    Diagonal entries are 1.0.
    """
    S, H, W, _ = pmaps_world.shape
    device = pmaps_world.device
    dtype = pmaps_world.dtype

    if valid_mask is None:
        valid_mask = torch.ones(S, H, W, dtype=torch.bool, device=device)

    pts_world_flat = pmaps_world.reshape(S, -1, 3)            # (S, HW, 3)
    valid_flat = valid_mask.reshape(S, -1)                    # (S, HW) bool
    n_valid = valid_flat.sum(dim=1).clamp(min=1).float()      # (S,)

    R_all = extrinsics[:, :3, :3].to(dtype)                   # (S, 3, 3)
    t_all = extrinsics[:, :3, 3].to(dtype)                    # (S, 3)
    K_all = intrinsics.to(dtype)                              # (S, 3, 3)

    # Vectorize over j: for each source frame i, project its points into every j.
    # pts_w (S, HW, 3) -> per j (S_src, HW, 3) -> ... compute pairwise.
    # We loop over i (S is small, e.g. 4) but vectorize over j to keep memory bounded.
    overlap_dir = torch.zeros(S, S, dtype=torch.float32, device=device)
    for i in range(S):
        pts_w = pts_world_flat[i]                             # (HW, 3)
        valid_i = valid_flat[i]
        # Transform to every j's camera frame at once: (S, HW, 3)
        pts_cam = torch.einsum("jab,nb->jna", R_all, pts_w) + t_all.unsqueeze(1)
        z = pts_cam[..., 2]                                   # (S, HW)
        in_front = z > min_depth
        uv = torch.einsum("jab,jnb->jna", K_all, pts_cam)     # (S, HW, 3)
        z_safe = z.clamp(min=min_depth)
        u = uv[..., 0] / z_safe
        v = uv[..., 1] / z_safe
        in_bounds = (u >= 0) & (u < W) & (v >= 0) & (v < H)
        visible = valid_i.unsqueeze(0) & in_front & in_bounds  # (S, HW)
        overlap_dir[i] = visible.sum(dim=1).float() / n_valid[i]
        overlap_dir[i, i] = 1.0

    overlap_sym = torch.minimum(overlap_dir, overlap_dir.T)
    return overlap_sym


# ---------------------------------------------------------------------------
# B1: Hidden object localization target
# ---------------------------------------------------------------------------
def _world_centroid_of_object(
    obj_id: int,
    identity_ids: torch.Tensor,   # (S, H, W) long
    pmaps_world: torch.Tensor,    # (S, H, W, 3)
    confmaps: torch.Tensor,       # (S, H, W) -- per-pixel confidence (>0 means valid)
) -> Tuple[torch.Tensor, float]:
    """Return (centroid_world: (3,), score: float)."""
    obj_mask = (identity_ids == obj_id) & (confmaps > 0)
    n_pix = obj_mask.sum().item()
    if n_pix == 0:
        return torch.zeros(3, dtype=pmaps_world.dtype, device=pmaps_world.device), 0.0
    pts = pmaps_world[obj_mask]                       # (N, 3)
    centroid = pts.mean(dim=0)
    return centroid, float(n_pix)


def robust_past_only_world_centroid(
    obj_id: int,
    identity_ids: torch.Tensor,
    pmaps_world: torch.Tensor,
    confmaps: torch.Tensor,
    target_frame_idx: int = -1,
    min_visible_pixels: int = 200,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Estimate an object's world centroid without reading target-frame pixels.

    Only history frames whose *raw* instance-mask area reaches
    ``min_visible_pixels`` contribute.  Valid-depth object points from all such
    frames are concatenated and aggregated with a coordinate-wise median.  The
    median is deliberately used instead of a mean so that one noisy depth tail
    or one unusually large view cannot dominate the label.

    This helper intentionally does not enforce target-frame invisibility.  The
    strict sample constructor below performs that independent eligibility check;
    keeping the operations separate makes the past-only invariant directly
    testable.
    """
    S = int(identity_ids.shape[0])
    target_idx = target_frame_idx if target_frame_idx >= 0 else S + target_frame_idx
    if target_idx < 0 or target_idx >= S:
        raise IndexError(f"target_frame_idx={target_frame_idx} outside sequence of length {S}")

    history_points = []
    history_counts = []
    visible_indices = []
    for frame_idx in range(S):
        if frame_idx == target_idx:
            continue
        raw_mask = identity_ids[frame_idx] == int(obj_id)
        raw_count = int(raw_mask.sum().item())
        if raw_count < int(min_visible_pixels):
            continue
        point_mask = raw_mask & (confmaps[frame_idx] > 0)
        points = pmaps_world[frame_idx][point_mask]
        if points.numel() == 0:
            continue
        finite = torch.isfinite(points).all(dim=-1)
        points = points[finite]
        if points.numel() == 0:
            continue
        history_points.append(points)
        history_counts.append(raw_count)
        visible_indices.append(frame_idx)

    if not history_points:
        raise ValueError("object has no qualifying finite past-only 3D points")

    points = torch.cat(history_points, dim=0)
    centroid = points.median(dim=0).values
    metadata = {
        "visible_history_indices": torch.as_tensor(
            visible_indices, dtype=torch.long, device=identity_ids.device
        ),
        "visible_history_raw_pixels": torch.as_tensor(
            history_counts, dtype=torch.long, device=identity_ids.device
        ),
        "n_history_points": torch.tensor(
            int(points.shape[0]), dtype=torch.long, device=identity_ids.device
        ),
    }
    return centroid, metadata


def polar_from_world_centroid(
    centroid_world: torch.Tensor,
    world_to_camera: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert one world centroid to evaluator-compatible polar coordinates."""
    dtype = centroid_world.dtype
    R = world_to_camera[:3, :3].to(dtype)
    t = world_to_camera[:3, 3].to(dtype)
    centroid_cam = R @ centroid_world + t
    x, y, z = centroid_cam.unbind(0)
    dist = torch.linalg.norm(centroid_cam).clamp(min=1e-3)
    azimuth = torch.atan2(x, z)
    elevation = torch.asin(((-y) / dist).clamp(-1.0, 1.0))
    return torch.stack([azimuth, elevation, torch.log(dist)]).to(torch.float32), centroid_cam


def compute_strict_hidden_object_target_for_id(
    obj_id: int,
    identity_ids: torch.Tensor,
    pmaps_world: torch.Tensor,
    confmaps: torch.Tensor,
    extrinsics: torch.Tensor,
    min_visible_pixels: int = 200,
    target_frame_idx: int = -1,
) -> Optional[Dict[str, torch.Tensor]]:
    """Construct a B2 strict-zero-hidden, past-only target for a fixed object.

    Eligibility uses raw masks: the target frame must contain exactly zero
    pixels of ``obj_id`` and at least one earlier frame must contain at least
    ``min_visible_pixels``.  Neither target RGB/depth/mask pixels nor target
    point-map values participate in centroid estimation.  The target camera
    extrinsic is used only to express the past-only world centroid in the label
    coordinate system.
    """
    S, H, W = identity_ids.shape
    target_idx = target_frame_idx if target_frame_idx >= 0 else S + target_frame_idx
    if target_idx < 0 or target_idx >= S:
        return None
    target_raw_pixels = int((identity_ids[target_idx] == int(obj_id)).sum().item())
    if target_raw_pixels != 0:
        return None

    try:
        centroid_world, centroid_meta = robust_past_only_world_centroid(
            obj_id=obj_id,
            identity_ids=identity_ids,
            pmaps_world=pmaps_world,
            confmaps=confmaps,
            target_frame_idx=target_idx,
            min_visible_pixels=min_visible_pixels,
        )
    except ValueError:
        return None

    per_frame_mask = torch.zeros(
        S, H, W, dtype=torch.bool, device=identity_ids.device
    )
    for frame_idx in centroid_meta["visible_history_indices"].tolist():
        per_frame_mask[frame_idx] = identity_ids[frame_idx] == int(obj_id)

    polar, centroid_cam = polar_from_world_centroid(
        centroid_world, extrinsics[target_idx]
    )
    visible_indices = centroid_meta["visible_history_indices"]
    pixel_counts = centroid_meta["visible_history_raw_pixels"]
    last_visible_frame = int(visible_indices.max().item())
    no_update_polar, _ = polar_from_world_centroid(
        centroid_world, extrinsics[last_visible_frame]
    )
    camera_centers = []
    for pose in extrinsics[last_visible_frame : target_idx + 1]:
        rotation = pose[:3, :3].to(centroid_world.dtype)
        translation = pose[:3, 3].to(centroid_world.dtype)
        camera_centers.append(-(rotation.T @ translation))
    camera_centers = torch.stack(camera_centers)
    if camera_centers.shape[0] > 1:
        accumulated_translation = torch.linalg.norm(
            camera_centers[1:] - camera_centers[:-1], dim=-1
        ).sum()
        accumulated_rotation = centroid_world.new_zeros(())
        pose_slice = extrinsics[last_visible_frame : target_idx + 1]
        for idx in range(pose_slice.shape[0] - 1):
            relative = pose_slice[idx + 1, :3, :3] @ pose_slice[idx, :3, :3].T
            cosine = ((torch.trace(relative) - 1.0) / 2.0).clamp(-1.0, 1.0)
            accumulated_rotation = accumulated_rotation + torch.acos(cosine)
    else:
        accumulated_translation = centroid_world.new_zeros(())
        accumulated_rotation = centroid_world.new_zeros(())
    return {
        "valid": torch.ones((), dtype=torch.bool, device=identity_ids.device),
        "polar": polar,
        "delta_world": centroid_world.to(torch.float32),
        "centroid_cam": centroid_cam.to(torch.float32),
        "per_frame_mask": per_frame_mask,
        "obj_id": torch.tensor(int(obj_id), dtype=torch.long, device=identity_ids.device),
        "target_raw_pixels": torch.tensor(
            target_raw_pixels, dtype=torch.long, device=identity_ids.device
        ),
        "visible_history_indices": visible_indices,
        "visible_history_raw_pixels": pixel_counts,
        "n_history_points": centroid_meta["n_history_points"],
        "last_visible_frame": torch.tensor(
            last_visible_frame, dtype=torch.long, device=identity_ids.device
        ),
        "last_visible_pixel_area": pixel_counts[visible_indices.argmax()],
        "no_update_polar": no_update_polar,
        "hidden_duration": torch.tensor(
            target_idx - last_visible_frame, dtype=torch.long, device=identity_ids.device
        ),
        "accumulated_camera_translation": accumulated_translation.to(torch.float32),
        "accumulated_camera_rotation": accumulated_rotation.to(torch.float32),
    }


def compute_hidden_object_target(
    identity_ids: torch.Tensor,    # (S, H, W) long
    pmaps_world: torch.Tensor,     # (S, H, W, 3) world coords
    confmaps: torch.Tensor,        # (S, H, W)  -- 0 means invalid pixel
    extrinsics: torch.Tensor,      # (S, 3, 4) world->camera
    min_visible_pixels: int = 200,
    last_frame_idx: int = -1,
) -> Optional[Dict[str, torch.Tensor]]:
    """Pick the highest-quality "seen-before but hidden-now" object and return
    its location relative to the last frame's camera.

    Hidden-now condition: object_id is NOT present (or has < min_visible_pixels px)
    in the last frame.
    Seen-before condition: object_id has >= min_visible_pixels in at least one
    earlier frame.

    Score for tie-breaking:
        score = num_visible_frames * 1000 + total_visible_pixels

    Returns a dict with:
        valid:        scalar bool tensor
        polar:        (3,) tensor (azimuth, elevation, log_distance) in last cam coords
        delta_world:  (3,) tensor (xyz centroid in world coords)
        per_frame_mask: (S, H, W) bool -- where the chosen object is in each
                        frame (zeros in invisible frames). Used by the head
                        for masked feature pooling.
        obj_id:       scalar long
    """
    S, _H, _W = identity_ids.shape

    # Per-frame valid pixel count per object id
    valid_pix = (confmaps > 0)
    # Build a (S, max_id+1) count tensor
    last_idx = last_frame_idx if last_frame_idx >= 0 else (S + last_frame_idx)

    # Collect candidate ids
    visible_in_past_count = {}
    visible_pixels_total = {}
    for s in range(S):
        if s == last_idx:
            continue
        ids_s = identity_ids[s][valid_pix[s]]
        if ids_s.numel() == 0:
            continue
        unique_ids, counts = torch.unique(ids_s, return_counts=True)
        for oid, cnt in zip(unique_ids.tolist(), counts.tolist()):
            if oid < 0:
                continue
            if cnt < min_visible_pixels:
                continue
            visible_in_past_count[oid] = visible_in_past_count.get(oid, 0) + 1
            visible_pixels_total[oid] = visible_pixels_total.get(oid, 0) + int(cnt)

    # Filter: not visible (or < min_visible_pixels) in last frame
    last_ids = identity_ids[last_idx][valid_pix[last_idx]]
    if last_ids.numel() == 0:
        last_count = {}
    else:
        u, c = torch.unique(last_ids, return_counts=True)
        last_count = {int(k): int(v) for k, v in zip(u.tolist(), c.tolist())}

    candidates = []
    for oid, n_frames in visible_in_past_count.items():
        if last_count.get(oid, 0) >= min_visible_pixels:
            continue
        score = n_frames * 1_000_000 + visible_pixels_total[oid]
        candidates.append((score, oid))

    if not candidates:
        return None

    candidates.sort(reverse=True)
    chosen_id = candidates[0][1]

    return compute_object_target_for_id(
        obj_id=chosen_id,
        identity_ids=identity_ids,
        pmaps_world=pmaps_world,
        confmaps=confmaps,
        extrinsics=extrinsics,
        min_visible_pixels=min_visible_pixels,
        last_frame_idx=last_frame_idx,
        require_hidden=True,
    )


def compute_visible_object_target(
    identity_ids: torch.Tensor,    # (S, H, W) long
    pmaps_world: torch.Tensor,     # (S, H, W, 3) world coords
    confmaps: torch.Tensor,        # (S, H, W)  -- 0 means invalid pixel
    extrinsics: torch.Tensor,      # (S, 3, 4) world->camera
    min_visible_pixels: int = 200,
    last_frame_idx: int = -1,
) -> Optional[Dict[str, torch.Tensor]]:
    """Pick a current-view visible object and return its location target.

    This is a B2 sanity target: the object must be visible in the reference
    frame itself, and the returned per-frame mask includes that reference-frame
    visibility. It asks whether the probe can solve basic visible-object
    localization before we ask it to infer hidden-object location from memory.
    """
    S, _H, _W = identity_ids.shape
    valid_pix = confmaps > 0
    last_idx = last_frame_idx if last_frame_idx >= 0 else (S + last_frame_idx)
    if last_idx < 0 or last_idx >= S:
        return None

    ids_last = identity_ids[last_idx][valid_pix[last_idx]]
    if ids_last.numel() == 0:
        return None
    unique_ids, counts = torch.unique(ids_last, return_counts=True)

    candidates = []
    for oid, cnt in zip(unique_ids.tolist(), counts.tolist()):
        if oid <= 0 or cnt < min_visible_pixels:
            continue
        total_count = int(((identity_ids == int(oid)) & valid_pix).sum().item())
        score = int(cnt) * 1_000_000 + total_count
        candidates.append((score, int(oid)))
    if not candidates:
        return None

    candidates.sort(reverse=True)
    return compute_object_target_for_id(
        obj_id=candidates[0][1],
        identity_ids=identity_ids,
        pmaps_world=pmaps_world,
        confmaps=confmaps,
        extrinsics=extrinsics,
        min_visible_pixels=min_visible_pixels,
        last_frame_idx=last_frame_idx,
        require_hidden=False,
        centroid_frame_idx=last_idx,
    )


def select_streaming_hidden_object_id(
    identity_ids: torch.Tensor,       # (S, H, W) long, raw instance ids
    seed_visible_indices: list[int],
    hidden_tail_indices: list[int],
    min_visible_pixels: int = 200,
) -> Optional[int]:
    """Select one prefix-invariant hidden object for streaming B probes.

    The object identity is chosen only from the seed prefix, normally frames
    [0,1,2] for a prefix_len=4 task. It must be hidden at every requested
    streaming tail, normally frames [3,7,15,31,63]. The returned id is the raw
    dataset instance id so it can be reused across different prefix lengths.
    """
    S = int(identity_ids.shape[0])
    seed_visible_indices = [int(i) for i in seed_visible_indices]
    hidden_tail_indices = [int(i) for i in hidden_tail_indices]
    if not seed_visible_indices or not hidden_tail_indices:
        return None
    if min(seed_visible_indices + hidden_tail_indices) < 0:
        return None
    if max(seed_visible_indices + hidden_tail_indices) >= S:
        return None

    visible_frame_count = {}
    visible_pixels_total = {}
    for s in seed_visible_indices:
        ids_s = identity_ids[s]
        if ids_s.numel() == 0:
            continue
        unique_ids, counts = torch.unique(ids_s, return_counts=True)
        for oid, cnt in zip(unique_ids.tolist(), counts.tolist()):
            if oid < 0 or cnt < min_visible_pixels:
                continue
            visible_frame_count[oid] = visible_frame_count.get(oid, 0) + 1
            visible_pixels_total[oid] = visible_pixels_total.get(oid, 0) + int(cnt)

    candidates = []
    for oid, n_frames in visible_frame_count.items():
        hidden_at_all_tails = True
        for tail in hidden_tail_indices:
            n_tail = int((identity_ids[tail] == oid).sum().item())
            if n_tail >= min_visible_pixels:
                hidden_at_all_tails = False
                break
        if not hidden_at_all_tails:
            continue
        score = n_frames * 1_000_000 + visible_pixels_total[oid]
        candidates.append((score, oid))

    if not candidates:
        return None
    candidates.sort(reverse=True)
    return int(candidates[0][1])


def select_strict_zero_hidden_object_id(
    identity_ids: torch.Tensor,
    target_frame_idx: int = -1,
    min_visible_pixels: int = 200,
    history_start_idx: int = 0,
) -> Optional[int]:
    """Select a deterministic object that is visible in history and zero at target.

    Counts deliberately use the raw instance mask rather than depth-valid pixels.
    Positive instance ids only are eligible, avoiding the background-id shortcut.
    """
    S = int(identity_ids.shape[0])
    target_idx = target_frame_idx if target_frame_idx >= 0 else S + target_frame_idx
    history_start_idx = max(0, int(history_start_idx))
    if target_idx <= history_start_idx or target_idx >= S:
        return None

    candidates = []
    ids = torch.unique(identity_ids[history_start_idx:target_idx])
    for obj_tensor in ids:
        obj_id = int(obj_tensor.item())
        if obj_id <= 0:
            continue
        if int((identity_ids[target_idx] == obj_id).sum().item()) != 0:
            continue
        counts = [
            int((identity_ids[i] == obj_id).sum().item())
            for i in range(history_start_idx, target_idx)
        ]
        qualifying = [count for count in counts if count >= int(min_visible_pixels)]
        if not qualifying:
            continue
        score = len(qualifying) * 1_000_000_000 + max(qualifying) * 1_000 + sum(qualifying)
        candidates.append((score, obj_id))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    return int(candidates[0][1])


def select_common_history_hidden_object_id(
    identity_ids: torch.Tensor,       # (S, H, W) long, raw instance ids
    confmaps: torch.Tensor,           # (S, H, W), >0 means valid geometry
    history_len: int = 8,
    hidden_tail_indices: list[int] | None = None,
    required_visible_indices: list[int] | None = None,
    preferred_visible_indices: list[int] | None = None,
    min_visible_pixels: int = 200,
    min_history_visible_frames: int = 3,
    min_query_pixels: int = 1024,
    min_border_px: int = 16,
    strict_post_history: bool = False,
) -> Optional[int]:
    """Select one object visible in common history and hidden afterwards.

    This is the main B1/B2 streaming selector for temporal ScanNet++ windows:
    the object query is grounded in the common history, while every compared
    prefix tail after the history must be hidden.  Prefixes at or before the
    common-history tail can be used as visible-current baselines by listing the
    tail in ``preferred_visible_indices``; this biases selection toward objects
    visible at that tail without making visibility mandatory.
    """
    S, H, W = identity_ids.shape
    device = identity_ids.device
    valid = confmaps.to(device) > 0
    history_len = max(1, min(int(history_len), S))
    hidden_tail_indices = [int(i) for i in (hidden_tail_indices or []) if 0 <= int(i) < S]
    required_visible_indices = [
        int(i) for i in (required_visible_indices or []) if 0 <= int(i) < S
    ]
    preferred_visible_indices = [
        int(i) for i in (preferred_visible_indices or []) if 0 <= int(i) < S
    ]
    if not hidden_tail_indices and not required_visible_indices and not preferred_visible_indices:
        return None

    hidden_checks = set(hidden_tail_indices)
    if strict_post_history:
        hidden_checks.update(range(history_len, S))

    visible_frame_count: dict[int, int] = {}
    visible_pixels_total: dict[int, int] = {}
    max_pixels: dict[int, int] = {}
    best_frame: dict[int, int] = {}

    for s in range(history_len):
        ids_s = identity_ids[s]
        valid_s = valid[s]
        unique_ids = torch.unique(ids_s[valid_s])
        for oid_t in unique_ids:
            oid = int(oid_t.item())
            if oid < 0:
                continue
            cnt = int(((ids_s == oid) & valid_s).sum().item())
            if cnt < min_visible_pixels:
                continue
            visible_frame_count[oid] = visible_frame_count.get(oid, 0) + 1
            visible_pixels_total[oid] = visible_pixels_total.get(oid, 0) + cnt
            if cnt > max_pixels.get(oid, 0):
                max_pixels[oid] = cnt
                best_frame[oid] = s

    candidates = []
    for oid, n_frames in visible_frame_count.items():
        if n_frames < int(min_history_visible_frames):
            continue
        if max_pixels.get(oid, 0) < int(min_query_pixels):
            continue

        bf = best_frame[oid]
        m_best = (identity_ids[bf] == oid) & valid[bf]
        ys, xs = torch.nonzero(m_best, as_tuple=True)
        if xs.numel() == 0:
            continue
        border = min(
            int(xs.min().item()),
            int(ys.min().item()),
            int(W - 1 - xs.max().item()),
            int(H - 1 - ys.max().item()),
        )
        if border < int(min_border_px):
            continue

        visible_ok = True
        for s in required_visible_indices:
            cnt = int(((identity_ids[s] == oid) & valid[s]).sum().item())
            if cnt < min_visible_pixels:
                visible_ok = False
                break
        if not visible_ok:
            continue

        hidden_ok = True
        for s in hidden_checks:
            cnt = int(((identity_ids[s] == oid) & valid[s]).sum().item())
            if cnt >= min_visible_pixels:
                hidden_ok = False
                break
        if not hidden_ok:
            continue

        preferred_visible = 0
        preferred_pixels = 0
        for s in preferred_visible_indices:
            cnt = int(((identity_ids[s] == oid) & valid[s]).sum().item())
            if cnt >= min_visible_pixels:
                preferred_visible += 1
                preferred_pixels += cnt

        score = (
            preferred_visible * 1_000_000_000_000
            + n_frames * 1_000_000_000
            + max_pixels.get(oid, 0) * 1_000
            + preferred_pixels
            + visible_pixels_total.get(oid, 0)
        )
        candidates.append((score, oid))

    if not candidates:
        return None
    candidates.sort(reverse=True)
    return int(candidates[0][1])


def compute_object_target_for_id(
    obj_id: int,
    identity_ids: torch.Tensor,    # (S, H, W) long
    pmaps_world: torch.Tensor,     # (S, H, W, 3) world coords
    confmaps: torch.Tensor,        # (S, H, W) -- 0 means invalid pixel
    extrinsics: torch.Tensor,      # (S, 3, 4) world->camera
    min_visible_pixels: int = 200,
    last_frame_idx: int = -1,
    require_hidden: bool = True,
    centroid_frame_idx: Optional[int] = None,
) -> Optional[Dict[str, torch.Tensor]]:
    """Return the current-prefix polar target and masks for a fixed object id."""
    S, H, W = identity_ids.shape
    device = identity_ids.device
    dtype = pmaps_world.dtype
    obj_id = int(obj_id)
    valid_pix = confmaps > 0
    last_idx = last_frame_idx if last_frame_idx >= 0 else (S + last_frame_idx)

    last_count = int(((identity_ids[last_idx] == obj_id) & valid_pix[last_idx]).sum().item())
    if require_hidden and last_count >= min_visible_pixels:
        return None

    # Build per-frame mask for the current prefix only. Hidden-object targets
    # mask out the reference frame; visible-object sanity targets keep it.
    per_frame_mask = torch.zeros(S, H, W, dtype=torch.bool, device=device)
    seen_before = False
    for s in range(S):
        if require_hidden and s == last_idx:
            continue
        m = (identity_ids[s] == obj_id) & valid_pix[s]
        if m.sum().item() >= min_visible_pixels:
            per_frame_mask[s] = m
            seen_before = True
    if not seen_before:
        return None

    if centroid_frame_idx is None:
        centroid_world, _ = _world_centroid_of_object(
            obj_id, identity_ids, pmaps_world, confmaps
        )
    else:
        centroid_idx = centroid_frame_idx if centroid_frame_idx >= 0 else (S + centroid_frame_idx)
        if centroid_idx < 0 or centroid_idx >= S:
            return None
        centroid_world, centroid_score = _world_centroid_of_object(
            obj_id,
            identity_ids[centroid_idx : centroid_idx + 1],
            pmaps_world[centroid_idx : centroid_idx + 1],
            confmaps[centroid_idx : centroid_idx + 1],
        )
        if centroid_score < min_visible_pixels:
            return None

    # Transform centroid to last frame camera coords
    R = extrinsics[last_idx, :3, :3].to(dtype)
    t = extrinsics[last_idx, :3, 3].to(dtype)
    centroid_cam = R @ centroid_world + t            # (3,)

    x, y, z = centroid_cam.unbind(0)
    dist = torch.linalg.norm(centroid_cam).clamp(min=1e-3)
    # Camera convention assumed: +X right, +Y down, +Z forward
    # (matches OpenCV / the convention used by depth_to_pointmap upstream).
    # azimuth   in [-pi, pi] : right(+) / left(-) around the y-axis
    # elevation in [-pi/2, pi/2]: up(+) / down(-).  +Y down -> use -y in asin.
    azimuth = torch.atan2(x, z)
    elevation = torch.asin((-y) / dist)
    log_dist = torch.log(dist)

    polar = torch.stack([azimuth, elevation, log_dist])

    return {
        "valid": torch.ones((), dtype=torch.bool, device=device),
        "polar": polar.to(torch.float32),
        "delta_world": centroid_world.to(torch.float32),
        "per_frame_mask": per_frame_mask,
        "obj_id": torch.tensor(obj_id, dtype=torch.long, device=device),
    }


# ---------------------------------------------------------------------------
# C1: Action encoding (relative camera pose, 6D rotation + 3 translation)
# ---------------------------------------------------------------------------
def encode_relative_pose(
    extr_from: torch.Tensor,   # (3, 4) world->cam_from
    extr_to: torch.Tensor,     # (3, 4) world->cam_to
    scene_scale: float = 1.0,
) -> torch.Tensor:
    """Return a 9-dim action vector: 6D continuous rotation + 3 translation.

    The action is the rigid transform that maps a point expressed in cam_from
    coords into cam_to coords:

        X_to = R_rel @ X_from + t_rel,
        where  R_rel = R_to @ R_from^T,
               t_rel = t_to - R_rel @ t_from.

    The 6D rotation representation = first two rows of R_rel, flattened (Zhou et
    al. 2019). Translation is divided by scene_scale.
    """
    R_from = extr_from[:3, :3]
    t_from = extr_from[:3, 3]
    R_to = extr_to[:3, :3]
    t_to = extr_to[:3, 3]

    R_rel = R_to @ R_from.T
    t_rel = t_to - R_rel @ t_from

    rot6 = R_rel[:2].reshape(-1)        # (6,)
    trans = t_rel / max(scene_scale, 1e-6)
    return torch.cat([rot6, trans], dim=0).to(torch.float32)


def relative_pose_matrix(
    extr_from: torch.Tensor, extr_to: torch.Tensor
) -> torch.Tensor:
    """Return a 4x4 relative pose mapping cam_from coords -> cam_to coords."""
    R_from = extr_from[:3, :3]
    t_from = extr_from[:3, 3]
    R_to = extr_to[:3, :3]
    t_to = extr_to[:3, 3]
    R_rel = R_to @ R_from.T
    t_rel = t_to - R_rel @ t_from
    out = torch.eye(4, dtype=extr_from.dtype, device=extr_from.device)
    out[:3, :3] = R_rel
    out[:3, 3] = t_rel
    return out
