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
    S, H, W = identity_ids.shape
    device = identity_ids.device
    dtype = pmaps_world.dtype

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

    centroid_world, _ = _world_centroid_of_object(
        chosen_id, identity_ids, pmaps_world, confmaps
    )

    # Build per-frame mask (only on past frames where the object is visible)
    per_frame_mask = torch.zeros(S, H, W, dtype=torch.bool, device=device)
    for s in range(S):
        if s == last_idx:
            continue
        m = (identity_ids[s] == chosen_id) & valid_pix[s]
        if m.sum().item() >= min_visible_pixels:
            per_frame_mask[s] = m

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
        "obj_id": torch.tensor(chosen_id, dtype=torch.long, device=device),
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
