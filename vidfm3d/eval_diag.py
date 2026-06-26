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
from typing import Any, Dict, List

import hydra
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf

from vidfm3d.utils import pylogger

log = pylogger.RankedLogger(__name__, rank_zero_only=True)


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
    for b in range(feat.shape[0]):
        out.append({
            "scene_path": scene_paths[b] if isinstance(scene_paths, list) else str(scene_paths[b]),
            "overlap_gt": _detach_cpu(batch["overlap_gt"][b]),
            "overlap_pred": _detach_cpu(pred[b]),
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
    for b in range(feat_n.shape[0]):
        out.append({
            "scene_path": scene_paths[b] if isinstance(scene_paths, list) else str(scene_paths[b]),
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
            "obj_id": int(obj_ids[b].item()),
            "valid": True,
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
            "valid": True,
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
    valid_idx = torch.nonzero(valid, as_tuple=False).flatten().tolist()
    for k, b in enumerate(valid_idx):
        out.append({
            "scene_path": scene_paths[b] if isinstance(scene_paths, list) else str(scene_paths[b]),
            "action": _detach_cpu(action[k]),
            "pred": _detach_cpu(pred[k]),
            "pred_no_action": _detach_cpu(pred_no_action[k]),
            "pred_shuffled_action": _detach_cpu(pred_shuffled_action[k]),
            "last_observation": _detach_cpu(last_observation[k]),
            "target": _detach_cpu(target_pooled[k]),
            "target_frame_idx": int(batch.get("target_frame_idx", torch.full_like(valid, -1))[b].item()),
            "valid": True,
        })
    return out


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
    for b in range(pred.shape[0]):
        out.append({
            "scene_path": scene_paths[b] if isinstance(scene_paths, list) else str(scene_paths[b]),
            "horizons": _detach_cpu(batch["action_horizons"][b]),
            "pred": _detach_cpu(pred[b]),
            "target": _detach_cpu(target[b]),
            "valid": _detach_cpu(valid[b].bool()),
            "start_extrinsic": _detach_cpu(batch["start_extrinsic"][b]),
            "target_extrinsics": _detach_cpu(batch["target_extrinsics"][b]),
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
    for b in range(pred.shape[0]):
        out.append({
            "scene_path": scene_paths[b] if isinstance(scene_paths, list) else str(scene_paths[b]),
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


def _feature_retrieval_summary(preds: torch.Tensor, tgts: torch.Tensor) -> Dict[str, float]:
    n = preds.shape[0]
    if n == 0:
        return {"mean_cos": float("nan"), "global_R@1": float("nan"),
                "global_R@5": float("nan"), "mean_rank": float("nan")}
    coss = F.cosine_similarity(preds, tgts, dim=-1)
    sim = F.normalize(preds, dim=-1) @ F.normalize(tgts, dim=-1).T
    rank = (sim.argsort(dim=-1, descending=True) ==
            torch.arange(n).unsqueeze(-1)).float().argmax(dim=-1)
    return {
        "mean_cos": float(coss.mean().item()),
        "global_R@1": float((rank == 0).float().mean().item()),
        "global_R@5": float((rank < 5).float().mean().item()),
        "mean_rank": float(rank.float().mean().item()),
    }


def _translation_error(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.linalg.norm(a[:3, 3].float() - b[:3, 3].float()).item())


def _path_length(start: torch.Tensor, targets: torch.Tensor) -> float:
    poses = torch.cat([start.unsqueeze(0), targets], dim=0)
    if poses.shape[0] <= 1:
        return 0.0
    diffs = poses[1:, :3, 3].float() - poses[:-1, :3, 3].float()
    return float(torch.linalg.norm(diffs, dim=-1).sum().item())


def _summarize(records, probe_type) -> Dict[str, float]:
    if not records:
        return {"n": 0}
    n = len(records)
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
        pos = [i for i, g in enumerate(all_gt) if g >= 0.4]
        neg = [i for i, g in enumerate(all_gt) if g <= 0.05]
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
        if not records:
            return {"n": 0}
        az_err = [abs(math.atan2(
            math.sin((r["polar_pred"][0] - r["polar_gt"][0]).item()),
            math.cos((r["polar_pred"][0] - r["polar_gt"][0]).item()),
        )) * 180 / math.pi for r in records]
        el_err = [(r["polar_pred"][1] - r["polar_gt"][1]).abs().item() * 180 / 3.14159265 for r in records]
        ld_err = [(r["polar_pred"][2] - r["polar_gt"][2]).abs().item() for r in records]
        return {
            "n": n,
            "mean_az_err_deg": float(sum(az_err) / n),
            "mean_el_err_deg": float(sum(el_err) / n),
            "mean_log_dist_err": float(sum(ld_err) / n),
        }
    if probe_type == "ego_belief_v2":
        if not records:
            return {"n": 0}
        return {
            "n": n,
            "mean_ang_err_deg": float(sum(r["ang_err_deg"] for r in records) / n),
            "mean_az_err_deg": float(sum(r["az_err_deg"] for r in records) / n),
            "mean_el_err_deg": float(sum(r["el_err_deg"] for r in records) / n),
            "top1": float(sum(r["top1"] for r in records) / n),
            "top3": float(sum(r["top3"] for r in records) / n),
            "mean_log_dist_err": float(sum(r["logd_err"] for r in records) / n),
        }
    if probe_type == "action_dynamics":
        preds = torch.stack([r["pred"] for r in records])
        tgts = torch.stack([r["target"] for r in records])
        no_action = torch.stack([r["pred_no_action"] for r in records])
        shuffled = torch.stack([r["pred_shuffled_action"] for r in records])
        last_obs = torch.stack([r["last_observation"] for r in records])
        out = {"n": n, **_feature_retrieval_summary(preds, tgts)}
        out.update({f"no_action_{k}": v for k, v in _feature_retrieval_summary(no_action, tgts).items()})
        out.update({f"shuffled_action_{k}": v for k, v in _feature_retrieval_summary(shuffled, tgts).items()})
        out.update({f"last_observation_{k}": v for k, v in _feature_retrieval_summary(last_obs, tgts).items()})
        out["action_gain_cos"] = out["mean_cos"] - out["no_action_mean_cos"]
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
                   **_feature_retrieval_summary(pred_all, tgt_all)}
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
                   **_feature_retrieval_summary(pred_all, tgt_all)}
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


def _bootstrap_primary_metrics(records, probe_type, reps=200, seed=0):
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
        summary = _summarize(sample_records, probe_type)
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


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> None:
    if not cfg.get("ckpt_path"):
        raise ValueError("Must provide ckpt_path=<path>")
    split = cfg.get("eval_split", "val")

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
    model.load_state_dict(state["state_dict"], strict=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.float().to(device).eval()

    probe_type = cfg.model.probe_type
    if probe_type not in RECORDERS:
        raise ValueError(f"Unknown probe_type {probe_type}")

    records: List[Dict[str, Any]] = []
    for i, item in enumerate(loader):
        batch = item[0] if isinstance(item, tuple) else item
        batch = {k: (v.to(device).float() if isinstance(v, torch.Tensor) and v.is_floating_point()
                     else v.to(device) if isinstance(v, torch.Tensor) else v)
                 for k, v in batch.items()}
        records.extend(RECORDERS[probe_type](model, batch))
        if (i + 1) % 20 == 0:
            log.info(f"[{i+1}/{len(loader)}] cum records: {len(records)}")

    out_dir = Path(cfg.paths.output_dir) / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / f"{split}_predictions.pt"
    sum_path = out_dir / f"{split}_summary.json"
    torch.save(records, pred_path)
    summary = _summarize(records, probe_type)
    for record in records:
        record["source"] = _source_from_path(str(record.get("scene_path", "")))
    summary["by_source"] = {
        source: _summarize([r for r in records if r["source"] == source], probe_type)
        for source in sorted({r["source"] for r in records})
    }
    summary["bootstrap_95ci"] = _bootstrap_primary_metrics(
        records, probe_type, reps=int(cfg.get("bootstrap_reps", 200)), seed=int(cfg.get("seed", 0))
    )
    summary["probe_type"] = probe_type
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
    if summary["feature_layer"] is None and summary["feat_postfix"] is not None:
        layer_match = re.search(r"layer(-?\d+)", str(summary["feat_postfix"]))
        if layer_match:
            summary["feature_layer"] = int(layer_match.group(1))
    if cfg.get("job_name"):
        summary["job_name"] = str(cfg.job_name)
    with open(sum_path, "w") as f:
        json.dump(summary, f, indent=2)
    log.info(f"Wrote {pred_path}")
    log.info(f"Wrote {sum_path}: {summary}")


if __name__ == "__main__":
    main()
