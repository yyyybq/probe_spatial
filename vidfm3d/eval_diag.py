"""Per-sample evaluation dump for the Spatial Diagnostic Suite.

Loads a trained ProbeExtensionLitModule from a checkpoint, runs the validation
dataloader of the supplied experiment config, and writes:

  {output_dir}/eval/{split}_predictions.pt    -- list[dict] of per-sample records
  {output_dir}/eval/{split}_summary.json      -- aggregated metrics

The probe-type-specific record schema:

  view_consistency:  {scene_path, overlap_gt (S,S), overlap_pred (S,S)}
  abnormal:          {scene_path, prob_normal, prob_shuffled, valid}
  ego_belief:        {scene_path, polar_gt (3,), polar_pred (3,), valid, obj_id}
  ego_belief_v2:     {scene_path, polar_gt (3,), pred bins/errors, valid}
  action_dynamics:   {scene_path, action (9,), pred (C,), target (C,), valid}
  path_integration:  {scene_path, horizons, pred (K,C), target (K,C), poses, valid}
  counterfactual:    {scene_path, horizons, pred (K,C), target (K,C), overlap, valid}

Usage:
  python vidfm3d/eval_diag.py \
      experiment=inscene15k_ext/view_consistency_wan_v1 \
      ckpt_path=/path/to/last.ckpt \
      eval_split=val
"""

from __future__ import annotations

import json
import math
import os
import re
import random
from pathlib import Path
from collections import OrderedDict
from typing import Any, Dict, List

import hydra
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf

from vidfm3d.utils import pylogger

log = pylogger.RankedLogger(__name__, rank_zero_only=True)


_LEGACY_UNUSED_CONSTANT_KEYS = frozenset({
    "probe.constant_context",
    "probe.constant_action",
})

# v2: records carry ``prediction_finite``; summaries exclude non-finite
# predictions from every mean and report them in ``n_nonfinite_pred``.
# Previously a NaN logit became ``pred_bin == 0`` (argmax of NaN) or a
# "negative" overlap decision, so diverged checkpoints produced plausible
# constant scores (timestep CogVideoX/MiniMax B2, MiniMax layer 20/25 A2).
EVALUATOR_VERSION = "eval_diag_v2_nonfinite_aware"

# Default A2 decision thresholds; ``main`` overrides them from the trained
# module's hyperparameters so evaluation matches the training contract.
DEFAULT_OVERLAP_POS_THRESHOLD = 0.4
DEFAULT_OVERLAP_NEG_THRESHOLD = 0.05


def _row_is_finite(*tensors: torch.Tensor, index: int) -> bool:
    return all(bool(torch.isfinite(t[index]).all()) for t in tensors if t is not None)


def _finite_records(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Records whose predictions are finite; v1 records without the flag pass."""
    return [r for r in records if r.get("prediction_finite", True)]


def _load_checkpoint_strict_with_legacy_loose_compat(
    model: torch.nn.Module,
    state_dict: Dict[str, Any],
) -> Dict[str, Any] | None:
    """Strictly load a checkpoint, with one narrowly guarded legacy exception.

    Some historical action/path checkpoints predate two current probe parameters
    used only by ablation conditioning modes.  A full-conditioning checkpoint
    has no values for them and the full forward path never reads them.  The
    display-only loose evaluator may therefore restore the historical module
    shape *in memory* and still use ``strict=True``.  This is deliberately not
    a permissive loader: any other state-dict difference remains fatal.
    """
    if not isinstance(state_dict, dict):
        raise TypeError("checkpoint state_dict must be a dictionary")
    legacy_mode = os.environ.get("Q00_EVALUATION_MODE") == "LEGACY_AS_IS_LOOSE"
    allow_compat = os.environ.get("Q00_LEGACY_AS_IS_STRICT_CONSTANTS_COMPAT") == "1"
    if not (legacy_mode and allow_compat):
        model.load_state_dict(state_dict, strict=True)
        return None

    current_keys = set(model.state_dict())
    checkpoint_keys = set(state_dict)
    missing = current_keys - checkpoint_keys
    unexpected = checkpoint_keys - current_keys
    probe_type = str(getattr(model, "probe_type", ""))
    probe = getattr(model, "probe", None)
    conditioning = str(getattr(probe, "conditioning", ""))
    eligible = (
        probe_type in {"action_dynamics", "path_integration"}
        and conditioning == "full"
        and missing == _LEGACY_UNUSED_CONSTANT_KEYS
        and not unexpected
        and probe is not None
    )
    if not eligible:
        # Preserve the ordinary strict failure (or success) outside the one
        # exact historical architecture transition.
        model.load_state_dict(state_dict, strict=True)
        return None

    for key in ("constant_context", "constant_action"):
        if not hasattr(probe, key):
            raise RuntimeError(f"legacy strict-load compatibility missing expected probe parameter: {key}")
        delattr(probe, key)
    model.load_state_dict(state_dict, strict=True)
    return {
        "mode": "removed_runtime_only_legacy_parameters",
        "keys": sorted(_LEGACY_UNUSED_CONSTANT_KEYS),
        "probe_type": probe_type,
        "conditioning": conditioning,
        "checkpoint_unchanged": True,
    }


def _execution_device() -> tuple[torch.device, Dict[str, Any]]:
    """Select the eval device and capture the exact CUDA execution context."""
    cuda_available = bool(torch.cuda.is_available())
    cuda_device_count = int(torch.cuda.device_count())
    if os.environ.get("Q00_REQUIRE_CUDA") == "1" and (
        not cuda_available or cuda_device_count < 1
    ):
        raise RuntimeError(
            "Q00_REQUIRE_CUDA=1 but CUDA is unavailable or no CUDA device is visible"
        )

    device = torch.device("cuda:0" if cuda_available else "cpu")
    cuda_device_name = None
    cuda_device_uuid = None
    if device.type == "cuda":
        cuda_device_name = str(torch.cuda.get_device_name(device))
        device_uuid = getattr(torch.cuda.get_device_properties(device), "uuid", None)
        if device_uuid is not None:
            cuda_device_uuid = str(device_uuid)

    return device, {
        "execution_device": str(device),
        "cuda_available": cuda_available,
        "cuda_device_count": cuda_device_count,
        "cuda_device_name": cuda_device_name,
        "cuda_device_uuid": cuda_device_uuid,
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }


def _detach_cpu(t: torch.Tensor) -> torch.Tensor:
    return t.detach().to("cpu")


def _source_from_path(path: str) -> str:
    if "processed_infinigen" in path:
        return "infinigen"
    if "processed_scannetpp" in path:
        return "scannetpp"
    return "unknown"


@torch.no_grad()
def _record_view_consistency(probe_module, batch) -> List[Dict[str, Any]]:
    feat = batch["vfm_feat"]
    logits = probe_module.probe(feat)
    pred = torch.sigmoid(logits)
    out = []
    scene_paths = batch.get("scene_path", ["?"] * feat.shape[0])
    manifest_sample_ids = batch.get("manifest_sample_id", [None] * feat.shape[0])
    for b in range(feat.shape[0]):
        out.append({
            "scene_path": scene_paths[b] if isinstance(scene_paths, list) else str(scene_paths[b]),
            "manifest_sample_id": manifest_sample_ids[b],
            "streaming_window_id": int(batch.get("streaming_window_id", torch.full((feat.shape[0],), -1, device=feat.device, dtype=torch.long))[b].item()),
            "streaming_tail_raw": int(batch.get("streaming_tail_raw", torch.full((feat.shape[0],), -1, device=feat.device, dtype=torch.long))[b].item()),
            "overlap_gt": _detach_cpu(batch["overlap_gt"][b]),
            "overlap_pred": _detach_cpu(pred[b]),
            "prediction_finite": _row_is_finite(logits, index=b),
        })
    return out


@torch.no_grad()
def _record_abnormal(probe_module, batch) -> List[Dict[str, Any]]:
    feat_n = batch["vfm_feat"]
    feat_s = batch["vfm_feat_shuffled"]
    valid = batch.get("abnormal_feat_valid",
                      torch.ones(feat_n.shape[0], dtype=torch.bool, device=feat_n.device))
    logit_n = probe_module.probe(feat_n)
    logit_s = probe_module.probe(feat_s)
    p_n = torch.sigmoid(logit_n)
    p_s = torch.sigmoid(logit_s)
    out = []
    scene_paths = batch.get("scene_path", ["?"] * feat_n.shape[0])
    manifest_sample_ids = batch.get("manifest_sample_id", [None] * feat_n.shape[0])
    for b in range(feat_n.shape[0]):
        out.append({
            "scene_path": scene_paths[b] if isinstance(scene_paths, list) else str(scene_paths[b]),
            "manifest_sample_id": manifest_sample_ids[b],
            "prob_normal": float(p_n[b].item()),
            "prob_shuffled": float(p_s[b].item()),
            "valid": bool(valid[b].item()),
        })
    return out


@torch.no_grad()
def _record_ego_belief(probe_module, batch) -> List[Dict[str, Any]]:
    from vidfm3d.models.probe_ext_module import _resize_mask_to_feat
    valid = batch["hidden_obj_valid"]
    out = []
    scene_paths = batch.get("scene_path", ["?"] * valid.shape[0])
    manifest_sample_ids = batch.get("manifest_sample_id", [None] * valid.shape[0])
    if int(valid.sum().item()) == 0:
        return out
    vfm_feat = batch["vfm_feat"][valid]
    per_pix = batch["hidden_obj_mask"][valid]
    polar_gt = batch["hidden_obj_polar"][valid]
    H_f, W_f = vfm_feat.shape[2], vfm_feat.shape[3]
    obj_mask_feat = _resize_mask_to_feat(per_pix, H_f, W_f)
    polar_pred = probe_module.probe(vfm_feat, obj_mask_feat)
    valid_idx = torch.nonzero(valid, as_tuple=False).flatten().tolist()
    obj_ids = batch.get("hidden_obj_id", torch.full_like(valid, -1, dtype=torch.long))
    for k, b in enumerate(valid_idx):
        out.append({
            "scene_path": scene_paths[b] if isinstance(scene_paths, list) else str(scene_paths[b]),
            "polar_gt": _detach_cpu(polar_gt[k]),
            "polar_pred": _detach_cpu(polar_pred[k]),
            "manifest_sample_id": manifest_sample_ids[b],
            "obj_id": int(obj_ids[b].item()),
            "valid": True,
            "prediction_finite": _row_is_finite(polar_pred, index=k),
        })
    return out


@torch.no_grad()
def _record_ego_belief_v2(probe_module, batch) -> List[Dict[str, Any]]:
    valid = batch["hidden_obj_valid"]
    out_records = []
    scene_paths = batch.get("scene_path", ["?"] * valid.shape[0])
    if int(valid.sum().item()) == 0:
        return out_records

    vfm_feat = batch["vfm_feat"][valid]
    query_feat = batch["belief_query_feat"][valid]
    polar_gt = batch["hidden_obj_polar"][valid]

    n_az = probe_module.probe.n_az_bins
    n_el = probe_module.probe.n_el_bins
    az_bin = ((polar_gt[:, 0] + math.pi) / (2 * math.pi) * n_az).long().clamp(0, n_az - 1)
    el_bin = ((polar_gt[:, 1] + math.pi / 2) / math.pi * n_el).long().clamp(0, n_el - 1)
    joint_bin = az_bin * n_el + el_bin

    out = probe_module.probe(vfm_feat, query_feat)
    logits = out["logits"]
    log_dist_pred = out["log_dist"]
    flat_logits = logits.reshape(-1, n_az * n_el)
    pred_bin = flat_logits.argmax(dim=-1)
    top3 = (flat_logits.topk(3, dim=-1).indices == joint_bin.unsqueeze(-1)).any(-1)

    pred_az_bin = pred_bin // n_el
    pred_el_bin = pred_bin % n_el
    pred_az = (pred_az_bin.float() + 0.5) * (2 * math.pi / n_az) - math.pi
    pred_el = (pred_el_bin.float() + 0.5) * (math.pi / n_el) - math.pi / 2
    cos_ang = (
        torch.sin(pred_el) * torch.sin(polar_gt[:, 1])
        + torch.cos(pred_el) * torch.cos(polar_gt[:, 1])
        * torch.cos(pred_az - polar_gt[:, 0])
    ).clamp(-1.0, 1.0)
    ang_err_deg = torch.acos(cos_ang) * (180.0 / math.pi)
    az_err_deg = torch.atan2(
        torch.sin(pred_az - polar_gt[:, 0]),
        torch.cos(pred_az - polar_gt[:, 0]),
    ).abs() * (180.0 / math.pi)
    el_err_deg = (pred_el - polar_gt[:, 1]).abs() * (180.0 / math.pi)
    logd_err = (log_dist_pred - polar_gt[:, 2]).abs()

    valid_idx = torch.nonzero(valid, as_tuple=False).flatten().tolist()
    sample_ids = batch.get("sample_id", [None] * valid.shape[0])
    paired_ids = batch.get("paired_sample_id", [None] * valid.shape[0])
    manifest_sample_ids = batch.get("manifest_sample_id", [None] * valid.shape[0])
    for k, b in enumerate(valid_idx):
        out_records.append({
            "scene_path": scene_paths[b] if isinstance(scene_paths, list) else str(scene_paths[b]),
            "polar_gt": _detach_cpu(polar_gt[k]),
            "pred_bin": int(pred_bin[k].item()),
            "target_bin": int(joint_bin[k].item()),
            "top1": bool((pred_bin[k] == joint_bin[k]).item()),
            "top3": bool(top3[k].item()),
            "ang_err_deg": float(ang_err_deg[k].item()),
            "az_err_deg": float(az_err_deg[k].item()),
            "el_err_deg": float(el_err_deg[k].item()),
            "logd_err": float(logd_err[k].item()),
            "obj_id": int(batch.get("hidden_obj_id", torch.full_like(valid, -1, dtype=torch.long))[b].item()),
            "sample_id": sample_ids[b],
            "manifest_sample_id": manifest_sample_ids[b],
            "paired_sample_id": paired_ids[b],
            "streaming_window_id": int(batch.get("streaming_window_id", torch.full_like(valid, -1))[b].item()),
            "streaming_tail_raw": int(batch.get("streaming_tail_raw", torch.full_like(valid, -1))[b].item()),
            "target_raw_object_pixels": int(batch.get("target_raw_object_pixels", torch.full_like(valid, -1, dtype=torch.long))[b].item()),
            "visible_history_frames": int(batch.get("visible_history_frames", torch.zeros_like(valid, dtype=torch.long))[b].item()),
            "last_visible_frame": int(batch.get("last_visible_frame", torch.full_like(valid, -1, dtype=torch.long))[b].item()),
            "last_visible_pixel_area": int(batch.get("last_visible_pixel_area", torch.zeros_like(valid, dtype=torch.long))[b].item()),
            "no_update_polar": _detach_cpu(batch.get("no_update_polar", torch.zeros(valid.shape[0], 3, device=valid.device))[b]),
            "hidden_duration": int(batch.get("hidden_duration", torch.zeros_like(valid, dtype=torch.long))[b].item()),
            "accumulated_camera_translation": float(batch.get("accumulated_camera_translation", torch.zeros_like(valid, dtype=torch.float32))[b].item()),
            "accumulated_camera_rotation": float(batch.get("accumulated_camera_rotation", torch.zeros_like(valid, dtype=torch.float32))[b].item()),
            "valid": True,
            # argmax over NaN logits silently returns bin 0; flag it instead.
            "prediction_finite": _row_is_finite(flat_logits, log_dist_pred, index=k),
        })
    return out_records


@torch.no_grad()
def _record_action_dynamics(probe_module, batch) -> List[Dict[str, Any]]:
    valid = batch.get("dyn_valid",
                      torch.ones(batch["input_feat"].shape[0], dtype=torch.bool,
                                 device=batch["input_feat"].device))
    out = []
    scene_paths = batch.get("scene_path", ["?"] * valid.shape[0])
    if int(valid.sum().item()) == 0:
        return out
    input_feat = batch["input_feat"][valid]
    action = batch["action"][valid]
    target_feat = batch["target_feat"][valid]
    target_pooled = target_feat.mean(dim=(1, 2))
    pred = probe_module.probe(input_feat, action)
    shuffled_action = torch.roll(action, shifts=1, dims=0) if action.shape[0] > 1 else torch.zeros_like(action)
    pred_no_action = probe_module.probe(input_feat, torch.zeros_like(action))
    pred_shuffled_action = probe_module.probe(input_feat, shuffled_action)
    last_observation = input_feat[:, -1].mean(dim=(1, 2))
    anchor_isolated = batch.get("anchor_isolated_feat")
    anchor_isolated_valid = batch.get(
        "anchor_isolated_valid", torch.zeros_like(valid, dtype=torch.bool)
    )
    if anchor_isolated is not None:
        anchor_isolated = anchor_isolated[valid].mean(dim=(1, 2))
    valid_idx = torch.nonzero(valid, as_tuple=False).flatten().tolist()
    sample_ids = batch.get("sample_id", [None] * valid.shape[0])
    paired_ids = batch.get("paired_sample_id", [None] * valid.shape[0])
    manifest_sample_ids = batch.get("manifest_sample_id", [None] * valid.shape[0])
    action_set_ids = batch.get("action_set_id", manifest_sample_ids)
    candidate_ids = batch.get("candidate_id", batch.get("action_horizon", [None] * valid.shape[0]))
    equivalence_groups = batch.get("action_equivalence_group", [None] * valid.shape[0])

    def _batch_value(values, index):
        if isinstance(values, torch.Tensor):
            value = values[index]
            return value.item() if value.numel() == 1 else _detach_cpu(value)
        return values[index]

    for k, b in enumerate(valid_idx):
        out.append({
            "scene_path": scene_paths[b] if isinstance(scene_paths, list) else str(scene_paths[b]),
            "action": _detach_cpu(action[k]),
            "pred": _detach_cpu(pred[k]),
            "pred_no_action": _detach_cpu(pred_no_action[k]),
            "pred_shuffled_action": _detach_cpu(pred_shuffled_action[k]),
            "last_observation": _detach_cpu(last_observation[k]),
            "anchor_isolated": _detach_cpu(anchor_isolated[k]) if anchor_isolated is not None else None,
            "anchor_isolated_valid": bool(anchor_isolated_valid[b].item()),
            "target": _detach_cpu(target_pooled[k]),
            "target_frame_idx": int(batch.get("target_frame_idx", torch.full_like(valid, -1))[b].item()),
            "action_horizon": int(batch["action_horizon"][b].item()) if "action_horizon" in batch else None,
            "raw_frame_offset": int(batch["raw_frame_offset"][b].item()) if "raw_frame_offset" in batch else None,
            "sample_id": sample_ids[b],
            "manifest_sample_id": manifest_sample_ids[b],
            "paired_sample_id": paired_ids[b],
            "action_set_id": str(_batch_value(action_set_ids, b)),
            "candidate_id": _batch_value(candidate_ids, b),
            "action_equivalence_group": _batch_value(equivalence_groups, b),
            "anchor_frame_idx": int(batch.get("anchor_frame_idx", torch.full_like(valid, -1))[b].item()),
            "streaming_window_id": int(batch.get("streaming_window_id", torch.full_like(valid, -1))[b].item()),
            "streaming_tail_raw": int(batch.get("streaming_tail_raw", torch.full_like(valid, -1))[b].item()),
            "valid": True,
            "prediction_finite": _row_is_finite(pred, pred_no_action, pred_shuffled_action, index=k),
        })
    return out


@torch.no_grad()
def _record_c1_action_sensitivity(probe_module, batch) -> List[Dict[str, Any]]:
    """Run a C1 head independently over same-anchor candidate actions.

    This is an evaluation-only C3 protocol.  It deliberately calls the C1
    signature ``probe(context, one_action)`` and never instantiates or restores
    a separately trained counterfactual head.

    Preferred batch keys are ``candidate_actions``, ``candidate_target_feat``
    and ``candidate_valid``.  The historical counterfactual key names are
    accepted as a data-only compatibility bridge; their model head is not.
    """
    context = batch.get("input_feat", batch.get("input_feat_seq"))
    actions = batch.get("candidate_actions", batch.get("counterfactual_actions"))
    targets = batch.get("candidate_target_feat", batch.get("target_feat_seq"))
    valid = batch.get("candidate_valid", batch.get("counterfactual_valid"))
    if context is None or actions is None or targets is None:
        raise ValueError(
            "C3-from-C1 evaluation requires a context, candidate actions, and "
            "matched candidate target features"
        )
    if actions.ndim != 3 or actions.shape[-1] != 9:
        raise ValueError("candidate actions must have shape (B, K, 9)")
    if context.shape[0] != actions.shape[0] or targets.shape[:2] != actions.shape[:2]:
        raise ValueError("context/actions/targets disagree on batch or candidate dimensions")
    if valid is None:
        valid = torch.ones(actions.shape[:2], dtype=torch.bool, device=actions.device)
    valid = valid.bool()

    batch_size, candidates = actions.shape[:2]
    expanded_context = context[:, None].expand(-1, candidates, *context.shape[1:])
    pred = probe_module.probe(
        expanded_context.reshape(batch_size * candidates, *context.shape[1:]),
        actions.reshape(batch_size * candidates, actions.shape[-1]),
    ).reshape(batch_size, candidates, -1)
    if targets.ndim == 5:
        target_pooled = targets.mean(dim=(2, 3))
    elif targets.ndim == 3:
        target_pooled = targets
    else:
        raise ValueError("candidate targets must have shape (B,K,H,W,C) or (B,K,C)")

    scene_paths = batch.get("scene_path", ["?"] * batch_size)
    manifest_ids = batch.get("manifest_sample_id", [None] * batch_size)
    action_set_ids = batch.get("action_set_id", manifest_ids)
    candidate_ids = batch.get("candidate_ids", batch.get("action_horizons"))
    equivalence = batch.get("action_equivalence_group")
    pred_finite = torch.isfinite(pred).all(dim=-1)
    out = []
    for b in range(batch_size):
        action_set_id = action_set_ids[b]
        if isinstance(action_set_id, torch.Tensor):
            action_set_id = action_set_id.item()
        set_finite = bool(pred_finite[b][valid[b]].all())
        out.append({
            "scene_path": scene_paths[b] if isinstance(scene_paths, list) else str(scene_paths[b]),
            "manifest_sample_id": manifest_ids[b],
            "action_set_id": str(action_set_id),
            "candidate_ids": _detach_cpu(candidate_ids[b]) if isinstance(candidate_ids, torch.Tensor) else (
                list(candidate_ids[b]) if candidate_ids is not None else list(range(candidates))
            ),
            "actions": _detach_cpu(actions[b]),
            "pred": _detach_cpu(pred[b]),
            "target": _detach_cpu(target_pooled[b]),
            # A set with a non-finite prediction is not scored at all, rather
            # than scored over its remaining candidates with a different chance.
            "valid": _detach_cpu(valid[b] & set_finite),
            "prediction_finite": set_finite,
            "action_equivalence_group": (
                _detach_cpu(equivalence[b]) if isinstance(equivalence, torch.Tensor)
                else list(equivalence[b]) if equivalence is not None else None
            ),
        })
    return out


def _group_c1_records_for_action_sensitivity(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert flat C1 eval records into same-anchor candidate sets."""
    grouped: "OrderedDict[str, List[Dict[str, Any]]]" = OrderedDict()
    for record in records:
        group_id = record.get("action_set_id") or record.get("manifest_sample_id")
        if group_id in (None, "", "None"):
            raise ValueError("C3-from-C1 flat records require action_set_id or manifest_sample_id")
        grouped.setdefault(str(group_id), []).append(record)

    output = []
    for group_id, rows in grouped.items():
        if len(rows) < 2:
            raise ValueError(
                f"action set {group_id} has only {len(rows)} valid candidate; "
                "C3 action sensitivity requires at least two"
            )
        if len({str(row.get("scene_path")) for row in rows}) != 1:
            raise ValueError(f"action set {group_id} crosses scene/context lineage")
        anchor_ids = {
            int(row["anchor_frame_idx"])
            for row in rows
            if row.get("anchor_frame_idx") is not None and int(row["anchor_frame_idx"]) >= 0
        }
        if len(anchor_ids) > 1:
            raise ValueError(f"action set {group_id} contains multiple anchor frames")
        for lineage_key in ("streaming_window_id", "streaming_tail_raw"):
            lineage_values = {
                int(row[lineage_key])
                for row in rows
                if row.get(lineage_key) is not None and int(row[lineage_key]) >= 0
            }
            if len(lineage_values) > 1:
                raise ValueError(
                    f"action set {group_id} contains multiple {lineage_key} values"
                )
        candidate_ids = [row.get("candidate_id") for row in rows]
        if len({str(value) for value in candidate_ids}) != len(candidate_ids):
            raise ValueError(f"duplicate candidate_id in action set {group_id}")
        equivalence = [row.get("action_equivalence_group") for row in rows]
        pred = torch.stack([row["pred"] for row in rows])
        # The C1 recorder flags rows whose predictions (or controls) are non-finite.
        # Grouping must keep that flag: otherwise argmax over a NaN similarity
        # column scores the set and n_nonfinite_pred_records stays 0.
        finite = [
            row.get("prediction_finite", True) is not False and bool(torch.isfinite(p).all())
            for row, p in zip(rows, pred)
        ]
        set_finite = all(finite)
        output.append({
            "scene_path": rows[0].get("scene_path", "?"),
            "manifest_sample_id": rows[0].get("manifest_sample_id"),
            "action_set_id": group_id,
            "candidate_ids": candidate_ids,
            "actions": torch.stack([row["action"] for row in rows]),
            "pred": pred,
            "target": torch.stack([row["target"] for row in rows]),
            "valid": torch.full((len(rows),), set_finite, dtype=torch.bool),
            "prediction_finite": set_finite,
            "n_nonfinite_candidates": len(rows) - sum(finite),
            "action_equivalence_group": None if all(value is None for value in equivalence) else equivalence,
        })
    return output


@torch.no_grad()
def _record_path_integration(probe_module, batch) -> List[Dict[str, Any]]:
    valid = batch.get("path_horizon_valid")
    if valid is None:
        valid = torch.ones(batch["path_actions"].shape[:2], dtype=torch.bool,
                           device=batch["path_actions"].device)
    pred = probe_module.probe(batch["input_feat_seq"], batch["path_actions"])
    target = batch["target_feat_seq"].mean(dim=(2, 3))
    out = []
    scene_paths = batch.get("scene_path", ["?"] * pred.shape[0])
    sample_ids = batch.get("sample_id", [None] * pred.shape[0])
    paired_ids = batch.get("paired_sample_id", [None] * pred.shape[0])
    manifest_sample_ids = batch.get("manifest_sample_id", [None] * pred.shape[0])
    for b in range(pred.shape[0]):
        out.append({
            "scene_path": scene_paths[b] if isinstance(scene_paths, list) else str(scene_paths[b]),
            "horizons": _detach_cpu(batch["action_horizons"][b]),
            "pred": _detach_cpu(pred[b]),
            "target": _detach_cpu(target[b]),
            "valid": _detach_cpu(valid[b].bool()),
            "start_extrinsic": _detach_cpu(batch["start_extrinsic"][b]),
            "target_extrinsics": _detach_cpu(batch["target_extrinsics"][b]),
            "path_actions": _detach_cpu(batch["path_actions"][b]),
            "sample_id": sample_ids[b],
            "manifest_sample_id": manifest_sample_ids[b],
            "paired_sample_id": paired_ids[b],
            "streaming_window_id": int(batch.get("streaming_window_id", torch.full((pred.shape[0],), -1, device=pred.device, dtype=torch.long))[b].item()),
            "streaming_tail_raw": int(batch.get("streaming_tail_raw", torch.full((pred.shape[0],), -1, device=pred.device, dtype=torch.long))[b].item()),
        })
    return out


@torch.no_grad()
def _record_counterfactual(probe_module, batch) -> List[Dict[str, Any]]:
    valid = batch.get("counterfactual_valid")
    if valid is None:
        valid = torch.ones(batch["counterfactual_actions"].shape[:2], dtype=torch.bool,
                           device=batch["counterfactual_actions"].device)
    pred = probe_module.probe(batch["input_feat_seq"], batch["counterfactual_actions"])
    target = batch["target_feat_seq"].mean(dim=(2, 3))
    out = []
    scene_paths = batch.get("scene_path", ["?"] * pred.shape[0])
    sample_ids = batch.get("sample_id", [None] * pred.shape[0])
    paired_ids = batch.get("paired_sample_id", [None] * pred.shape[0])
    manifest_sample_ids = batch.get("manifest_sample_id", [None] * pred.shape[0])
    for b in range(pred.shape[0]):
        out.append({
            "scene_path": scene_paths[b] if isinstance(scene_paths, list) else str(scene_paths[b]),
            "sample_id": sample_ids[b],
            "paired_sample_id": paired_ids[b],
            "manifest_sample_id": manifest_sample_ids[b],
            "streaming_window_id": int(batch.get("streaming_window_id", torch.full((pred.shape[0],), -1, device=pred.device, dtype=torch.long))[b].item()),
            "streaming_tail_raw": int(batch.get("streaming_tail_raw", torch.full((pred.shape[0],), -1, device=pred.device, dtype=torch.long))[b].item()),
            "horizons": _detach_cpu(batch["action_horizons"][b]),
            "pred": _detach_cpu(pred[b]),
            "target": _detach_cpu(target[b]),
            "valid": _detach_cpu(valid[b].bool()),
            "overlap": _detach_cpu(batch["counterfactual_overlap"][b]),
            "start_extrinsic": _detach_cpu(batch["start_extrinsic"][b]),
            "target_extrinsics": _detach_cpu(batch["target_extrinsics"][b]),
        })
    return out


RECORDERS = {
    "view_consistency": _record_view_consistency,
    "abnormal": _record_abnormal,
    "ego_belief": _record_ego_belief,
    "ego_belief_v2": _record_ego_belief_v2,
    "action_dynamics": _record_action_dynamics,
    "path_integration": _record_path_integration,
    "counterfactual": _record_counterfactual,
}


def _binary_auc(scores: List[float], labels: List[int]) -> float:
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and scores[order[j]] == scores[order[i]]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[order[k]] = avg_rank
        i = j

    sum_pos_ranks = sum(ranks[i] for i, label in enumerate(labels) if label == 1)
    return (sum_pos_ranks - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def _binary_average_precision(scores: List[float], labels: List[int]) -> float:
    n_pos = sum(labels)
    if n_pos == 0:
        return float("nan")
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    hits = 0
    precision_sum = 0.0
    for rank, idx in enumerate(order, start=1):
        if labels[idx]:
            hits += 1
            precision_sum += hits / rank
    return precision_sum / n_pos


def _feature_retrieval_summary(
    preds: torch.Tensor,
    tgts: torch.Tensor,
    compute_calibrated: bool = True,
) -> Dict[str, float]:
    n = preds.shape[0]
    if n == 0:
        return {"mean_cos": float("nan"), "global_R@1": float("nan"),
                "global_R@5": float("nan"), "mean_rank": float("nan"),
                "median_rank": float("nan"), "centered_cos": float("nan"),
                "whitened_cos": float("nan")}
    coss = F.cosine_similarity(preds, tgts, dim=-1)
    sim = F.normalize(preds, dim=-1) @ F.normalize(tgts, dim=-1).T
    rank = (sim.argsort(dim=-1, descending=True) ==
            torch.arange(n).unsqueeze(-1)).float().argmax(dim=-1)
    target_mean = tgts.mean(dim=0, keepdim=True)
    preds_centered = preds - target_mean
    tgts_centered = tgts - target_mean
    centered_cos = F.cosine_similarity(preds_centered, tgts_centered, dim=-1)
    whitened_cos = torch.full_like(centered_cos, float("nan"))
    if compute_calibrated and n > 1:
        # Shrinkage whitening in the target-data subspace.  This avoids the
        # singular CxC covariance that occurs when feature dimension > n.
        _u, singular, vh = torch.linalg.svd(tgts_centered.float(), full_matrices=False)
        variance = singular.square() / max(n - 1, 1)
        positive = variance > torch.finfo(variance.dtype).eps
        if int(positive.sum().item()) > 0:
            basis = vh[positive].T
            floor = variance[positive].median().clamp(min=1e-8) * 1e-3
            scale = torch.rsqrt(variance[positive] + floor)
            pred_white = (preds_centered.float() @ basis) * scale
            tgt_white = (tgts_centered.float() @ basis) * scale
            whitened_cos = F.cosine_similarity(pred_white, tgt_white, dim=-1)
    return {
        "mean_cos": float(coss.mean().item()),
        "centered_cos": float(centered_cos.mean().item()),
        "whitened_cos": float(torch.nanmean(whitened_cos).item()),
        "global_R@1": float((rank == 0).float().mean().item()),
        "global_R@5": float((rank < 5).float().mean().item()),
        "mean_rank": float(rank.float().mean().item()),
        "median_rank": float(rank.float().median().item()),
    }


def _translation_error(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.linalg.norm(a[:3, 3].float() - b[:3, 3].float()).item())


def _path_length(start: torch.Tensor, targets: torch.Tensor) -> float:
    poses = torch.cat([start.unsqueeze(0), targets], dim=0)
    if poses.shape[0] <= 1:
        return 0.0
    diffs = poses[1:, :3, 3].float() - poses[:-1, :3, 3].float()
    return float(torch.linalg.norm(diffs, dim=-1).sum().item())


def _pearson(values_a: List[float], values_b: List[float]) -> float:
    if len(values_a) < 2 or len(values_a) != len(values_b):
        return float("nan")
    a = torch.as_tensor(values_a, dtype=torch.float64)
    b = torch.as_tensor(values_b, dtype=torch.float64)
    a = a - a.mean()
    b = b - b.mean()
    denom = torch.linalg.norm(a) * torch.linalg.norm(b)
    if float(denom) <= 1e-12:
        return float("nan")
    return float((a @ b / denom).item())


def _summarize_c1_action_sensitivity(
    records: List[Dict[str, Any]], *, include_global_retrieval: bool = True
) -> Dict[str, float]:
    """Summarize evaluation-only C3 behavior from a frozen C1 predictor.

    Candidate matching tests whether predictions move toward the target paired
    with each action. Pairwise metrics quantify how much the prediction changes
    as the action changes. If an eval set supplies ``action_equivalence_group``,
    sensitivity is split into within-equivalence invariance and between-class
    separation.
    """
    flat_pred, flat_target = [], []
    correct_cos: List[float] = []
    exact_candidate_hits: List[float] = []
    exact_candidate_chances: List[float] = []
    intervention_hits: List[float] = []
    intervention_chances: List[float] = []
    intervention_margins: List[float] = []
    prediction_distances: List[float] = []
    target_distances: List[float] = []
    action_distances: List[float] = []
    equivalent_prediction_distances: List[float] = []
    equivalent_target_distances: List[float] = []
    distinct_prediction_distances: List[float] = []
    distinct_target_distances: List[float] = []
    valid_sets = 0

    for record in records:
        if record.get("prediction_finite", True) is False:
            continue
        valid = record["valid"].bool()
        if int(valid.sum().item()) < 2:
            continue
        pred = record["pred"][valid].float()
        target = record["target"][valid].float()
        actions = record["actions"][valid].float()
        equivalence = record.get("action_equivalence_group")
        if equivalence is not None:
            if isinstance(equivalence, torch.Tensor):
                equivalence = equivalence[valid].tolist()
            else:
                equivalence = [value for value, keep in zip(equivalence, valid.tolist()) if keep]

        valid_sets += 1
        flat_pred.append(pred)
        flat_target.append(target)
        similarity = F.normalize(pred, dim=-1) @ F.normalize(target, dim=-1).T
        diagonal = similarity.diag()
        correct_cos.extend(diagonal.tolist())
        best = similarity.argmax(dim=-1)
        exact_candidate_hits.extend(
            (best == torch.arange(len(pred), device=best.device)).float().tolist()
        )
        exact_candidate_chances.extend([1.0 / len(pred)] * len(pred))
        if equivalence is None:
            intervention_hits.extend(
                (best == torch.arange(len(pred), device=best.device)).float().tolist()
            )
            intervention_chances.extend([1.0 / len(pred)] * len(pred))
            off_diagonal = similarity.masked_fill(
                torch.eye(len(pred), dtype=torch.bool, device=similarity.device), float("-inf")
            )
            intervention_margins.extend((diagonal - off_diagonal.max(dim=-1).values).tolist())
        else:
            # Equivalent actions may legitimately share an outcome. Exact
            # diagonal matching would mark tied, same-class targets as errors,
            # so the primary validity/margin operate on equivalence classes.
            for index, group in enumerate(equivalence):
                positive = torch.as_tensor(
                    [value == group for value in equivalence],
                    dtype=torch.bool,
                    device=similarity.device,
                )
                negative = ~positive
                intervention_hits.append(float(positive[best[index]].item()))
                intervention_chances.append(float(positive.float().mean().item()))
                if bool(negative.any()):
                    positive_score = similarity[index][positive].max()
                    negative_score = similarity[index][negative].max()
                    intervention_margins.append(float((positive_score - negative_score).item()))

        pred_distance = 1.0 - F.normalize(pred, dim=-1) @ F.normalize(pred, dim=-1).T
        target_distance = 1.0 - F.normalize(target, dim=-1) @ F.normalize(target, dim=-1).T
        action_distance = torch.cdist(actions, actions, p=2)
        for left in range(len(pred)):
            for right in range(left + 1, len(pred)):
                pred_value = float(pred_distance[left, right].item())
                target_value = float(target_distance[left, right].item())
                prediction_distances.append(pred_value)
                target_distances.append(target_value)
                action_distances.append(float(action_distance[left, right].item()))
                if equivalence is not None:
                    if equivalence[left] == equivalence[right]:
                        equivalent_prediction_distances.append(pred_value)
                        equivalent_target_distances.append(target_value)
                    else:
                        distinct_prediction_distances.append(pred_value)
                        distinct_target_distances.append(target_value)

    def mean_or_nan(values: List[float]) -> float:
        return float(sum(values) / len(values)) if values else float("nan")

    if flat_pred:
        pred_all = torch.cat(flat_pred)
        target_all = torch.cat(flat_target)
        output = {
            "n": len(records),
            "n_valid_action_sets": valid_sets,
            "n_valid_candidates": int(pred_all.shape[0]),
        }
        if include_global_retrieval:
            output.update(_feature_retrieval_summary(pred_all, target_all))
    else:
        output = {"n": len(records), "n_valid_action_sets": 0, "n_valid_candidates": 0}
    exact_validity = mean_or_nan(exact_candidate_hits)
    exact_chance = mean_or_nan(exact_candidate_chances)
    intervention_validity = mean_or_nan(intervention_hits)
    intervention_chance = mean_or_nan(intervention_chances)

    def chance_adjusted(value: float, chance: float) -> float:
        if not (math.isfinite(value) and math.isfinite(chance)) or chance >= 1.0:
            return float("nan")
        return (value - chance) / (1.0 - chance)

    output.update({
        "c3_head_source": "c1_action_dynamics",
        "counterfactual_consistency": mean_or_nan(correct_cos),
        "exact_candidate_validity": exact_validity,
        "exact_candidate_chance": exact_chance,
        "exact_candidate_validity_chance_adjusted": chance_adjusted(exact_validity, exact_chance),
        "intervention_validity": intervention_validity,
        "intervention_chance": intervention_chance,
        "intervention_validity_chance_adjusted": chance_adjusted(
            intervention_validity, intervention_chance
        ),
        "intervention_margin": mean_or_nan(intervention_margins),
        "action_sensitivity_cosine": mean_or_nan(prediction_distances),
        "target_sensitivity_cosine": mean_or_nan(target_distances),
        "action_to_prediction_sensitivity_pearson": _pearson(action_distances, prediction_distances),
        "target_to_prediction_sensitivity_pearson": _pearson(target_distances, prediction_distances),
        "sensitivity_calibration_mae": mean_or_nan([
            abs(predicted - target)
            for predicted, target in zip(prediction_distances, target_distances)
        ]),
        "equivalent_action_prediction_distance": mean_or_nan(equivalent_prediction_distances),
        "equivalent_action_target_distance": mean_or_nan(equivalent_target_distances),
        "distinct_action_prediction_distance": mean_or_nan(distinct_prediction_distances),
        "distinct_action_target_distance": mean_or_nan(distinct_target_distances),
        "n_candidate_pairs": len(prediction_distances),
        "n_equivalent_action_pairs": len(equivalent_prediction_distances),
        "n_distinct_action_pairs": len(distinct_prediction_distances),
    })
    equivalent = output["equivalent_action_prediction_distance"]
    distinct = output["distinct_action_prediction_distance"]
    output["equivalence_separation_gap"] = (
        distinct - equivalent if math.isfinite(equivalent) and math.isfinite(distinct) else float("nan")
    )
    return output


def _bootstrap_c1_action_sensitivity(
    records: List[Dict[str, Any]], reps: int = 200, seed: int = 0
) -> Dict[str, List[float]]:
    """Bootstrap C3 diagnostics with the action set as the sampling unit."""
    if not records or reps <= 0:
        return {}
    metrics = (
        "intervention_validity",
        "intervention_validity_chance_adjusted",
        "exact_candidate_validity",
        "intervention_margin",
        "action_sensitivity_cosine",
        "target_to_prediction_sensitivity_pearson",
        "equivalence_separation_gap",
    )
    rng = random.Random(seed)
    values: Dict[str, List[float]] = {metric: [] for metric in metrics}
    for _ in range(reps):
        sample = [records[rng.randrange(len(records))] for _ in records]
        summary = _summarize_c1_action_sensitivity(
            sample, include_global_retrieval=False
        )
        for metric in metrics:
            value = float(summary.get(metric, float("nan")))
            if math.isfinite(value):
                values[metric].append(value)

    intervals: Dict[str, List[float]] = {}
    for metric, finite_values in values.items():
        if finite_values:
            tensor = torch.as_tensor(finite_values, dtype=torch.float64)
            intervals[metric] = [
                float(torch.quantile(tensor, 0.025).item()),
                float(torch.quantile(tensor, 0.975).item()),
            ]
    return intervals


def _summarize(
    records,
    probe_type,
    compute_calibrated: bool = True,
    overlap_pos_threshold: float = DEFAULT_OVERLAP_POS_THRESHOLD,
    overlap_neg_threshold: float = DEFAULT_OVERLAP_NEG_THRESHOLD,
) -> Dict[str, float]:
    if not records:
        return {"n": 0}
    n = len(records)
    finite_records = _finite_records(records)
    n_nonfinite_pred = n - len(finite_records)
    if probe_type in {"view_consistency", "ego_belief", "ego_belief_v2", "action_dynamics"}:
        # Non-finite predictions are counted, never scored as negatives or bin 0.
        records = finite_records
        if not records:
            return {"n": n, "n_scored": 0, "n_nonfinite_pred": n_nonfinite_pred}
    if probe_type == "view_consistency":
        all_gt = []
        all_pred = []
        for r in records:
            gt, pred = r["overlap_gt"], r["overlap_pred"]
            S = gt.shape[0]
            mask = ~torch.eye(S, dtype=torch.bool)
            all_gt.extend(gt[mask].float().tolist())
            all_pred.extend(pred[mask].float().tolist())

        mae = sum(abs(p - g) for p, g in zip(all_pred, all_gt)) / max(len(all_gt), 1)
        pos = [i for i, g in enumerate(all_gt) if g >= overlap_pos_threshold]
        neg = [i for i, g in enumerate(all_gt) if g <= overlap_neg_threshold]
        hard = pos + neg
        hard_labels = [1] * len(pos) + [0] * len(neg)
        hard_scores = [all_pred[i] for i in hard]
        hard_pred = [1 if s > 0.5 else 0 for s in hard_scores]

        def _acc(indices, label):
            if not indices:
                return float("nan")
            return sum((all_pred[i] > 0.5) == bool(label) for i in indices) / len(indices)

        hard_acc = (
            sum(int(p == y) for p, y in zip(hard_pred, hard_labels)) / len(hard)
            if hard else float("nan")
        )
        pos_acc = _acc(pos, 1)
        neg_acc = _acc(neg, 0)
        balanced = (pos_acc + neg_acc) / 2.0 if pos and neg else float("nan")
        return {
            "n": n,
            "n_scored": len(records),
            "n_nonfinite_pred": n_nonfinite_pred,
            "overlap_pos_threshold": float(overlap_pos_threshold),
            "overlap_neg_threshold": float(overlap_neg_threshold),
            "n_pairs": len(all_gt),
            "mean_overlap_mae": float(mae),
            "hard_acc": float(hard_acc),
            "pos_acc": float(pos_acc),
            "neg_acc": float(neg_acc),
            "balanced_acc": float(balanced),
            "roc_auc": float(_binary_auc(hard_scores, hard_labels)) if hard else float("nan"),
            "pr_auc": float(_binary_average_precision(hard_scores, hard_labels)) if hard else float("nan"),
            "n_pos": len(pos),
            "n_neg": len(neg),
            "pos_frac": float(len(pos) / len(hard)) if hard else float("nan"),
            "trivial_neg_acc": float(len(neg) / len(hard)) if hard else float("nan"),
        }
    if probe_type == "abnormal":
        valids = [r for r in records if r["valid"]]
        if not valids:
            return {"n": n, "n_valid": 0}
        # Discriminability: prob_shuffled - prob_normal (>0 means correct)
        delta = [r["prob_shuffled"] - r["prob_normal"] for r in valids]
        acc = [(r["prob_shuffled"] > 0.5) and (r["prob_normal"] <= 0.5) for r in valids]
        rank_acc = [r["prob_shuffled"] > r["prob_normal"] for r in valids]
        scores = [r["prob_normal"] for r in valids] + [r["prob_shuffled"] for r in valids]
        labels = [0] * len(valids) + [1] * len(valids)
        return {
            "n": n, "n_valid": len(valids),
            "mean_delta": float(sum(delta) / len(delta)),
            "pair_acc": float(sum(acc) / len(acc)),
            "paired_rank_acc": float(sum(rank_acc) / len(rank_acc)),
            "roc_auc": float(_binary_auc(scores, labels)),
            "pr_auc": float(_binary_average_precision(scores, labels)),
        }
    if probe_type == "ego_belief":
        n_scored = len(records)
        az_err = [abs(math.atan2(
            math.sin((r["polar_pred"][0] - r["polar_gt"][0]).item()),
            math.cos((r["polar_pred"][0] - r["polar_gt"][0]).item()),
        )) * 180 / math.pi for r in records]
        el_err = [(r["polar_pred"][1] - r["polar_gt"][1]).abs().item() * 180 / 3.14159265 for r in records]
        ld_err = [(r["polar_pred"][2] - r["polar_gt"][2]).abs().item() for r in records]
        return {
            "n": n,
            "n_scored": n_scored,
            "n_nonfinite_pred": n_nonfinite_pred,
            "mean_az_err_deg": float(sum(az_err) / n_scored),
            "mean_el_err_deg": float(sum(el_err) / n_scored),
            "mean_log_dist_err": float(sum(ld_err) / n_scored),
        }
    if probe_type == "ego_belief_v2":
        n_scored = len(records)
        return {
            "n": n,
            "n_scored": n_scored,
            "n_nonfinite_pred": n_nonfinite_pred,
            "mean_ang_err_deg": float(sum(r["ang_err_deg"] for r in records) / n_scored),
            "mean_az_err_deg": float(sum(r["az_err_deg"] for r in records) / n_scored),
            "mean_el_err_deg": float(sum(r["el_err_deg"] for r in records) / n_scored),
            "top1": float(sum(r["top1"] for r in records) / n_scored),
            "top3": float(sum(r["top3"] for r in records) / n_scored),
            "mean_log_dist_err": float(sum(r["logd_err"] for r in records) / n_scored),
        }
    if probe_type == "action_dynamics":
        preds = torch.stack([r["pred"] for r in records])
        tgts = torch.stack([r["target"] for r in records])
        no_action = torch.stack([r["pred_no_action"] for r in records])
        shuffled = torch.stack([r["pred_shuffled_action"] for r in records])
        last_obs = torch.stack([r["last_observation"] for r in records])
        out = {"n": n, "n_scored": len(records), "n_nonfinite_pred": n_nonfinite_pred,
               **_feature_retrieval_summary(preds, tgts, compute_calibrated)}
        out.update({f"no_action_{k}": v for k, v in _feature_retrieval_summary(no_action, tgts, compute_calibrated).items()})
        out.update({f"shuffled_action_{k}": v for k, v in _feature_retrieval_summary(shuffled, tgts, compute_calibrated).items()})
        out.update({f"last_observation_{k}": v for k, v in _feature_retrieval_summary(last_obs, tgts, compute_calibrated).items()})
        isolated = [r for r in records if r.get("anchor_isolated_valid") and r.get("anchor_isolated") is not None]
        if isolated:
            isolated_pred = torch.stack([r["anchor_isolated"] for r in isolated])
            isolated_tgt = torch.stack([r["target"] for r in isolated])
            out.update({
                f"copy_anchor_isolated_{k}": v
                for k, v in _feature_retrieval_summary(isolated_pred, isolated_tgt, compute_calibrated).items()
            })
            out["copy_anchor_isolated_n"] = len(isolated)
        else:
            out["copy_anchor_isolated_n"] = 0
        out["action_gain_cos"] = out["mean_cos"] - out["no_action_mean_cos"]
        horizons = sorted({
            int(r["action_horizon"]) for r in records
            if r.get("action_horizon") is not None and int(r["action_horizon"]) > 0
        })
        horizon_summaries = []
        for horizon in horizons:
            # Remove only the grouping key so the ordinary C1 summary can be
            # reused without recursive grouping. Retrieval pools are per h.
            subset = [
                {k: v for k, v in r.items() if k != "action_horizon"}
                for r in records if r.get("action_horizon") == horizon
            ]
            summary = _summarize(subset, "action_dynamics", compute_calibrated)
            horizon_summaries.append(summary)
            out.update({f"h{horizon}_{key}": value for key, value in summary.items()
                        if key not in {"n_scored", "n_nonfinite_pred"}})
        if horizon_summaries:
            out["n_horizons"] = len(horizon_summaries)
            for key in ("mean_cos", "global_R@1", "global_R@5", "action_gain_cos"):
                out[f"macro_horizon_{key}"] = sum(s[key] for s in horizon_summaries) / len(horizon_summaries)
        return out
    if probe_type == "path_integration":
        flat_pred, flat_tgt = [], []
        final_errors = []
        drift_rates = []
        step_errors = []
        loop_errors = []
        horizon_err = {}
        n_valid_records = 0
        for r in records:
            valid = r["valid"].bool()
            if int(valid.sum().item()) == 0:
                continue
            n_valid_records += 1
            pred = r["pred"][valid]
            tgt = r["target"][valid]
            poses = r["target_extrinsics"][valid]
            horizons = r["horizons"][valid]
            flat_pred.append(pred)
            flat_tgt.append(tgt)

            sim = F.normalize(pred, dim=-1) @ F.normalize(tgt, dim=-1).T
            retrieved = sim.argmax(dim=-1)
            for i, j in enumerate(retrieved.tolist()):
                err = _translation_error(poses[j], poses[i])
                step_errors.append(err)
                h = int(horizons[i].item())
                horizon_err.setdefault(h, []).append(err)

            final_idx = pred.shape[0] - 1
            final_ret = int(retrieved[final_idx].item())
            final_err = _translation_error(poses[final_ret], poses[final_idx])
            final_errors.append(final_err)
            length = _path_length(r["start_extrinsic"], poses)
            if length > 1e-6:
                drift_rates.append(final_err / length)

            final_to_start = _translation_error(poses[final_idx], r["start_extrinsic"])
            if final_to_start <= 0.1:
                loop_errors.append(_translation_error(poses[final_ret], r["start_extrinsic"]))

        if flat_pred:
            pred_all = torch.cat(flat_pred, dim=0)
            tgt_all = torch.cat(flat_tgt, dim=0)
            out = {"n": n, "n_valid_records": n_valid_records,
                   "n_valid_steps": int(pred_all.shape[0]),
                   **_feature_retrieval_summary(pred_all, tgt_all, compute_calibrated)}
        else:
            out = {"n": n, "n_valid_records": 0, "n_valid_steps": 0}

        out.update({
            "final_pose_error": float(sum(final_errors) / len(final_errors)) if final_errors else float("nan"),
            "drift_rate": float(sum(drift_rates) / len(drift_rates)) if drift_rates else float("nan"),
            "mean_step_pose_error": float(sum(step_errors) / len(step_errors)) if step_errors else float("nan"),
            "loop_closure_error": float(sum(loop_errors) / len(loop_errors)) if loop_errors else float("nan"),
            "n_loop": len(loop_errors),
        })
        for h, errs in sorted(horizon_err.items()):
            out[f"pose_error_h{h}"] = float(sum(errs) / len(errs))
        return out
    if probe_type == "counterfactual":
        flat_pred, flat_tgt = [], []
        correct_cos = []
        intervention_hits = []
        intervention_margins = []
        valid_by_h = {}
        cos_by_h = {}
        for r in records:
            valid = r["valid"].bool()
            if int(valid.sum().item()) == 0:
                continue
            pred = r["pred"][valid]
            tgt = r["target"][valid]
            horizons = r["horizons"][valid]
            flat_pred.append(pred)
            flat_tgt.append(tgt)
            cos = F.cosine_similarity(pred, tgt, dim=-1)
            correct_cos.extend(cos.tolist())
            for h, c in zip(horizons.tolist(), cos.tolist()):
                valid_by_h[int(h)] = valid_by_h.get(int(h), 0) + 1
                cos_by_h.setdefault(int(h), []).append(float(c))

            if pred.shape[0] >= 2:
                sim = F.normalize(pred, dim=-1) @ F.normalize(tgt, dim=-1).T
                diag = sim.diag()
                best = sim.argmax(dim=-1)
                intervention_hits.extend((best == torch.arange(pred.shape[0])).float().tolist())
                offdiag = sim.masked_fill(torch.eye(pred.shape[0], dtype=torch.bool), float("-inf"))
                margin = diag - offdiag.max(dim=-1).values
                intervention_margins.extend(margin.tolist())

        if flat_pred:
            pred_all = torch.cat(flat_pred, dim=0)
            tgt_all = torch.cat(flat_tgt, dim=0)
            out = {"n": n, "n_valid_interventions": int(pred_all.shape[0]),
                   **_feature_retrieval_summary(pred_all, tgt_all, compute_calibrated)}
        else:
            out = {"n": n, "n_valid_interventions": 0}
        out.update({
            "counterfactual_consistency": float(sum(correct_cos) / len(correct_cos)) if correct_cos else float("nan"),
            "intervention_validity": float(sum(intervention_hits) / len(intervention_hits)) if intervention_hits else float("nan"),
            "intervention_margin": float(sum(intervention_margins) / len(intervention_margins)) if intervention_margins else float("nan"),
        })
        for h, vals in sorted(cos_by_h.items()):
            out[f"cf_cos_h{h}"] = float(sum(vals) / len(vals))
            out[f"cf_n_h{h}"] = valid_by_h[h]
        return out
    return {"n": n}


def _bootstrap_primary_metrics(records, probe_type, reps=200, seed=0, **summary_kwargs):
    """Scene-level bootstrap CIs for the headline metrics."""
    primary = {
        "view_consistency": ["balanced_acc", "roc_auc", "mean_overlap_mae"],
        "abnormal": ["paired_rank_acc", "roc_auc", "mean_delta"],
        "ego_belief": ["mean_az_err_deg", "mean_el_err_deg", "mean_log_dist_err"],
        "ego_belief_v2": ["mean_ang_err_deg", "top1", "mean_log_dist_err"],
        "action_dynamics": ["mean_cos", "action_gain_cos"],
    }.get(probe_type, [])
    if not primary or reps <= 0:
        return {}
    grouped = {}
    for record in records:
        grouped.setdefault(record.get("scene_path", "?"), []).append(record)
    scenes = sorted(grouped)
    if len(scenes) < 2:
        return {}
    rng = random.Random(seed)
    values = {key: [] for key in primary}
    for _ in range(reps):
        sampled = [rng.choice(scenes) for _ in scenes]
        sample_records = [record for scene in sampled for record in grouped[scene]]
        summary = _summarize(sample_records, probe_type, compute_calibrated=False, **summary_kwargs)
        for key in primary:
            value = summary.get(key)
            if isinstance(value, (int, float)) and math.isfinite(value):
                values[key].append(float(value))
    result = {}
    for key, vals in values.items():
        if vals:
            vals.sort()
            result[key] = {
                "low": vals[int(0.025 * (len(vals) - 1))],
                "high": vals[int(0.975 * (len(vals) - 1))],
            }
    return result


def _valid_evaluation_record(record, evaluation_protocol):
    """Count usable records, respecting per-horizon and action-set validity.

    A record whose prediction is non-finite is never usable: it must not be
    scored as a wrong (or right) answer.
    """
    if record.get("prediction_finite", True) is False:
        return False
    valid = torch.as_tensor(record.get("valid", True), dtype=torch.bool)
    required = 2 if evaluation_protocol == "c3_action_sensitivity" else 1
    return int(valid.sum().item()) >= required


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> None:
    if not cfg.get("ckpt_path"):
        raise ValueError("Must provide ckpt_path=<path>")
    split = cfg.get("eval_split", "val")
    if split not in {"val", "test"}:
        raise ValueError(f"Unknown eval_split {split!r}; expected val or test")
    device, execution_metadata = _execution_device()

    if split == "test":
        # Existing experiment configs name only validation_datasets. Rebind
        # their explicit split at evaluation time; the dataset then requires a
        # frozen manifest and cannot silently alias test back to validation.
        cfg.data.data_module.validation_datasets = [
            str(spec).replace("split='val'", "split='test'").replace('split="val"', 'split="test"')
            for spec in cfg.data.data_module.validation_datasets
        ]

    log.info(OmegaConf.to_yaml(cfg))

    # Build datamodule + model
    datamodule = hydra.utils.instantiate(cfg.data.data_module)
    datamodule.setup(stage="validate" if split == "val" else "test")
    # Attach a minimal trainer stub so val_dataloader() can check strategy/rank.
    from types import SimpleNamespace
    datamodule.trainer = SimpleNamespace(
        strategy=None, world_size=1, global_rank=0, local_rank=0, num_nodes=1
    )

    if split == "val":
        loader = datamodule.val_dataloader()
        if isinstance(loader, list):
            loader = loader[0]
    else:
        loader = datamodule.test_dataloader() if hasattr(datamodule, "test_dataloader") \
            else datamodule.val_dataloader()
        if isinstance(loader, list):
            loader = loader[0]

    model = hydra.utils.instantiate(cfg.model)
    state = torch.load(cfg.ckpt_path, map_location="cpu", weights_only=False)
    legacy_state_dict_compatibility = _load_checkpoint_strict_with_legacy_loose_compat(
        model, state["state_dict"]
    )
    model.float().to(device).eval()

    probe_type = cfg.model.probe_type
    evaluation_protocol = str(cfg.get("evaluation_protocol", probe_type))
    if evaluation_protocol not in {str(probe_type), "c3_action_sensitivity"}:
        raise ValueError(
            f"Unknown evaluation_protocol={evaluation_protocol!r}; expected "
            f"{probe_type!r} or 'c3_action_sensitivity'"
        )
    if evaluation_protocol == "c3_action_sensitivity" and probe_type != "action_dynamics":
        raise ValueError(
            "evaluation_protocol=c3_action_sensitivity must load a C1 "
            "action_dynamics checkpoint; independently trained C3 heads are not accepted"
        )
    if probe_type not in RECORDERS:
        raise ValueError(f"Unknown probe_type {probe_type}")

    records: List[Dict[str, Any]] = []
    loader_n_samples = 0
    loader_manifest_ids: List[str] = []
    invalid_manifest_ids: List[str] = []
    for i, item in enumerate(loader):
        batch = item[0] if isinstance(item, tuple) else item
        batch = {k: (v.to(device).float() if isinstance(v, torch.Tensor) and v.is_floating_point()
                     else v.to(device) if isinstance(v, torch.Tensor) else v)
                 for k, v in batch.items()}
        if isinstance(batch.get("scene_path"), list):
            batch_n = len(batch["scene_path"])
        elif isinstance(batch.get("vfm_feat"), torch.Tensor):
            batch_n = int(batch["vfm_feat"].shape[0])
        else:
            raise ValueError("Cannot determine eval loader batch cardinality")
        loader_n_samples += batch_n
        batch_manifest_ids = [str(value) for value in batch.get("manifest_sample_id", [])]
        if len(batch_manifest_ids) != batch_n:
            raise ValueError("Missing/invalid manifest_sample_id lineage in eval batch")
        loader_manifest_ids.extend(batch_manifest_ids)
        if probe_type in {"ego_belief", "ego_belief_v2"}:
            row_valid = batch["hidden_obj_valid"].bool().detach().cpu().tolist()
            invalid_manifest_ids.extend(mid for mid, valid in zip(batch_manifest_ids, row_valid) if not valid)
        elif probe_type == "abnormal" and "abnormal_feat_valid" in batch:
            row_valid = batch["abnormal_feat_valid"].bool().detach().cpu().tolist()
            invalid_manifest_ids.extend(mid for mid, valid in zip(batch_manifest_ids, row_valid) if not valid)
        elif probe_type == "action_dynamics" and "dyn_valid" in batch:
            row_valid = batch["dyn_valid"].bool().detach().cpu().tolist()
            invalid_manifest_ids.extend(mid for mid, valid in zip(batch_manifest_ids, row_valid) if not valid)
        if evaluation_protocol == "c3_action_sensitivity" and (
            "candidate_actions" in batch or "counterfactual_actions" in batch
        ):
            records.extend(_record_c1_action_sensitivity(model, batch))
        else:
            records.extend(RECORDERS[probe_type](model, batch))
        if (i + 1) % 20 == 0:
            log.info(f"[{i+1}/{len(loader)}] cum records: {len(records)}")

    out_dir = Path(cfg.paths.output_dir) / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    output_stem = (
        split
        if evaluation_protocol == probe_type
        else f"{split}_{evaluation_protocol}"
    )
    pred_path = out_dir / f"{output_stem}_predictions.pt"
    sum_path = out_dir / f"{output_stem}_summary.json"
    if evaluation_protocol == "c3_action_sensitivity" and records and "actions" not in records[0]:
        records = _group_c1_records_for_action_sensitivity(records)
    torch.save(records, pred_path)
    # Score A2 with the thresholds the module was trained with, not constants.
    summary_kwargs = {}
    if probe_type == "view_consistency":
        hparams = getattr(model, "hparams", {})
        summary_kwargs = {
            "overlap_pos_threshold": float(hparams.get("view_overlap_pos_threshold", DEFAULT_OVERLAP_POS_THRESHOLD)),
            "overlap_neg_threshold": float(hparams.get("view_overlap_neg_threshold", DEFAULT_OVERLAP_NEG_THRESHOLD)),
        }
    summary = (
        _summarize_c1_action_sensitivity(records)
        if evaluation_protocol == "c3_action_sensitivity"
        else _summarize(records, probe_type, **summary_kwargs)
    )
    for record in records:
        record["source"] = _source_from_path(str(record.get("scene_path", "")))
    summary["by_source"] = {
        source: (
            _summarize_c1_action_sensitivity([r for r in records if r["source"] == source])
            if evaluation_protocol == "c3_action_sensitivity"
            else _summarize([r for r in records if r["source"] == source], probe_type, **summary_kwargs)
        )
        for source in sorted({r["source"] for r in records})
    }
    summary["bootstrap_95ci"] = (
        _bootstrap_c1_action_sensitivity(
            records,
            reps=int(cfg.get("bootstrap_reps", 200)),
            seed=int(cfg.get("seed", 0)),
        )
        if evaluation_protocol == "c3_action_sensitivity"
        else _bootstrap_primary_metrics(
            records, probe_type, reps=int(cfg.get("bootstrap_reps", 200)), seed=int(cfg.get("seed", 0)),
            **summary_kwargs,
        )
    )
    summary["probe_type"] = probe_type
    summary["evaluation_protocol"] = evaluation_protocol
    summary["evaluator_version"] = EVALUATOR_VERSION
    summary["n_nonfinite_pred_records"] = sum(1 for r in records if r.get("prediction_finite", True) is False)
    summary["loader_n_samples"] = loader_n_samples
    valid_records = [record for record in records if _valid_evaluation_record(record, evaluation_protocol)]
    summary["valid_record_n"] = len(valid_records)
    if evaluation_protocol == "c3_action_sensitivity":
        summary["loader_count_unit"] = "flat_candidates_or_action_sets_as_emitted_by_dataset"
        summary["valid_record_unit"] = "action_sets"
        summary["invalid_record_n"] = len(records) - int(summary.get("n_valid_action_sets", 0))
        summary["valid_record_fraction"] = (
            int(summary.get("n_valid_action_sets", 0)) / len(records)
            if records else float("nan")
        )
    else:
        summary["invalid_record_n"] = loader_n_samples - len(valid_records)
        summary["valid_record_fraction"] = len(valid_records) / loader_n_samples if loader_n_samples else float("nan")
    summary["loader_manifest_ids"] = loader_manifest_ids
    summary["valid_manifest_ids"] = [str(record.get("manifest_sample_id", "")) for record in valid_records]
    for record in records:
        manifest_id = str(record.get("manifest_sample_id", ""))
        if not _valid_evaluation_record(record, evaluation_protocol) and manifest_id not in invalid_manifest_ids:
            invalid_manifest_ids.append(manifest_id)
    summary["invalid_manifest_ids"] = invalid_manifest_ids
    summary.update(execution_metadata)
    summary["ckpt_path"] = cfg.ckpt_path
    summary["feature_layer"] = cfg.get("feature_layer", None)
    summary["feat_postfix"] = cfg.get("feat_postfix", None)
    summary["feature_timestep"] = cfg.get("feature_timestep", None)
    summary["vfm_name"] = cfg.get("vfm_name", None)
    summary["context_feat_root"] = cfg.get("context_feat_root", None)
    summary["streaming_feat_root"] = cfg.get("streaming_feat_root", None)
    summary["prefix_min_len"] = cfg.get("prefix_min_len", None)
    summary["prefix_max_len"] = cfg.get("prefix_max_len", None)
    summary["prefix_stride"] = cfg.get("prefix_stride", None)
    summary["prefix_len"] = cfg.get("prefix_len", None)
    summary["target_feat_root"] = cfg.get("target_feat_root", None)
    summary["eval_dataset_specs"] = [
        str(value) for value in cfg.data.data_module.validation_datasets
    ]
    if legacy_state_dict_compatibility is not None:
        summary["legacy_state_dict_compatibility"] = legacy_state_dict_compatibility
    if summary["feature_layer"] is None and summary["feat_postfix"] is not None:
        layer_match = re.search(r"layer(-?\d+)", str(summary["feat_postfix"]))
        if layer_match:
            summary["feature_layer"] = int(layer_match.group(1))
    if cfg.get("job_name"):
        summary["job_name"] = str(cfg.job_name)
    for key in ("Q00_UNIVERSE_ID", "Q00_CAMPAIGN", "Q00_MODEL_VARIANT", "Q00_PROTOCOL",
                "Q00_MANIFEST_SHA256", "Q00_CONFIG_SHA256",
                "Q00_RESOLVED_CONFIG_SHA256", "Q00_RESOLVED_CONFIG_SOURCE_KIND"):
        if os.environ.get(key):
            summary[key.lower()] = os.environ[key]
    with open(sum_path, "w") as f:
        json.dump(summary, f, indent=2)
    log.info(f"Wrote {pred_path}")
    log.info(f"Wrote {sum_path}: {summary}")


if __name__ == "__main__":
    main()
