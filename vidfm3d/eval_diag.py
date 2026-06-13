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
    last_pose = batch["last_pose_enc"][valid]
    H_f, W_f = vfm_feat.shape[2], vfm_feat.shape[3]
    obj_mask_feat = _resize_mask_to_feat(per_pix, H_f, W_f)
    polar_pred = probe_module.probe(vfm_feat, obj_mask_feat, last_pose)
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
    valid_idx = torch.nonzero(valid, as_tuple=False).flatten().tolist()
    for k, b in enumerate(valid_idx):
        out.append({
            "scene_path": scene_paths[b] if isinstance(scene_paths, list) else str(scene_paths[b]),
            "action": _detach_cpu(action[k]),
            "pred": _detach_cpu(pred[k]),
            "target": _detach_cpu(target_pooled[k]),
            "valid": True,
        })
    return out


RECORDERS = {
    "view_consistency": _record_view_consistency,
    "abnormal": _record_abnormal,
    "ego_belief": _record_ego_belief,
    "ego_belief_v2": _record_ego_belief_v2,
    "action_dynamics": _record_action_dynamics,
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
        return {
            "n": n, "n_valid": len(valids),
            "mean_delta": float(sum(delta) / len(delta)),
            "pair_acc": float(sum(acc) / len(acc)),
        }
    if probe_type == "ego_belief":
        if not records:
            return {"n": 0}
        az_err = [(r["polar_pred"][0] - r["polar_gt"][0]).abs().item() * 180 / 3.14159265 for r in records]
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
        coss = []
        preds = torch.stack([r["pred"] for r in records])
        tgts = torch.stack([r["target"] for r in records])
        coss = F.cosine_similarity(preds, tgts, dim=-1)
        # global retrieval rank
        sim = F.normalize(preds, dim=-1) @ F.normalize(tgts, dim=-1).T
        rank = (sim.argsort(dim=-1, descending=True) ==
                torch.arange(n).unsqueeze(-1)).float().argmax(dim=-1)
        r1 = (rank == 0).float().mean().item()
        r5 = (rank < 5).float().mean().item()
        return {
            "n": n,
            "mean_cos": float(coss.mean().item()),
            "global_R@1": r1,
            "global_R@5": r5,
            "mean_rank": float(rank.float().mean().item()),
        }
    return {"n": n}


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> None:
    if not cfg.get("ckpt_path"):
        raise ValueError("Must provide ckpt_path=<path>")
    split = cfg.get("eval_split", "val")

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
    model.load_state_dict(state["state_dict"], strict=False)
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
    summary["probe_type"] = probe_type
    summary["ckpt_path"] = cfg.ckpt_path
    with open(sum_path, "w") as f:
        json.dump(summary, f, indent=2)
    log.info(f"Wrote {pred_path}")
    log.info(f"Wrote {sum_path}: {summary}")


if __name__ == "__main__":
    main()
