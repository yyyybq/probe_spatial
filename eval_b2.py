"""Standalone evaluation for B2 (ego_belief_v2) probes.

Reuses the Lightning module's `_step_ego_belief_v2` to get per-batch metrics,
then aggregates with sample-weighted averaging and stratifies by source.

Usage:
  python eval_b2.py \
      experiment=inscene15k_ext/ego_belief_v2_wan_v1 \
      ckpt_path=/path/to/checkpoint.ckpt
"""
from __future__ import annotations

import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import hydra
import rootutils
import torch
from omegaconf import DictConfig, OmegaConf

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from vidfm3d.utils import pylogger

log = pylogger.RankedLogger(__name__, rank_zero_only=True)


@torch.no_grad()
def _per_sample_metrics(model, batch, device) -> List[Dict[str, Any]]:
    """Compute per-sample metrics on a (validated) batch.

    Returns list of dicts with: source, ang_err_deg, top1, top3, logd_err.
    """
    valid = batch["hidden_obj_valid"]
    if int(valid.sum().item()) == 0:
        return []

    vfm_feat = batch["vfm_feat"][valid].float()
    query_feat = batch["belief_query_feat"][valid].float()
    polar_gt = batch["hidden_obj_polar"][valid].float()

    n_az = model.probe.n_az_bins
    n_el = model.probe.n_el_bins
    az_bin = ((polar_gt[:, 0] + math.pi) / (2 * math.pi) * n_az).long().clamp(0, n_az - 1)
    el_bin = ((polar_gt[:, 1] + math.pi / 2) / math.pi * n_el).long().clamp(0, n_el - 1)
    joint_bin = az_bin * n_el + el_bin

    out = model.probe(vfm_feat, query_feat)
    logits = out["logits"]
    log_dist_pred = out["log_dist"]

    flat_logits = logits.reshape(-1, n_az * n_el)
    pred_bin = flat_logits.argmax(dim=-1)
    top1 = (pred_bin == joint_bin).float()
    top3 = (flat_logits.topk(3, dim=-1).indices == joint_bin.unsqueeze(-1)).any(-1).float()

    pred_az = pred_bin // n_el
    pred_el = pred_bin % n_el
    az_center = (pred_az.float() + 0.5) * (2 * math.pi / n_az) - math.pi
    el_center = (pred_el.float() + 0.5) * (math.pi / n_el) - math.pi / 2
    cos_ang = (
        torch.sin(el_center) * torch.sin(polar_gt[:, 1])
        + torch.cos(el_center) * torch.cos(polar_gt[:, 1])
        * torch.cos(az_center - polar_gt[:, 0])
    ).clamp(-1.0, 1.0)
    ang_err_deg = torch.acos(cos_ang) * (180.0 / math.pi)
    logd_err = (log_dist_pred - polar_gt[:, 2]).abs()

    # Separate azimuth-only and elevation-only errors (for compatibility with B1 table)
    az_center_only = (pred_az.float() + 0.5) * (2 * math.pi / n_az) - math.pi
    el_center_only = (pred_el.float() + 0.5) * (math.pi / n_el) - math.pi / 2

    def _wrap(d):
        # Wrap angular error to [-pi, pi]
        return torch.atan2(torch.sin(d), torch.cos(d)).abs()

    az_err = _wrap(az_center_only - polar_gt[:, 0]) * (180.0 / math.pi)
    el_err = (el_center_only - polar_gt[:, 1]).abs() * (180.0 / math.pi)

    scene_paths = batch.get("scene_path")
    valid_idx = torch.nonzero(valid, as_tuple=False).flatten().tolist()

    records = []
    for k, b in enumerate(valid_idx):
        sp = scene_paths[b] if isinstance(scene_paths, list) else str(scene_paths)
        if "infinigen" in sp.lower():
            src = "infinigen"
        elif "scannetpp" in sp.lower():
            src = "scannetpp"
        else:
            src = "other"
        records.append({
            "source": src,
            "ang_err_deg": float(ang_err_deg[k].item()),
            "az_err_deg": float(az_err[k].item()),
            "el_err_deg": float(el_err[k].item()),
            "top1": float(top1[k].item()),
            "top3": float(top3[k].item()),
            "logd_err": float(logd_err[k].item()),
        })
    return records


def _aggregate(records: List[Dict[str, Any]], label: str) -> Dict[str, float]:
    if not records:
        return {"label": label, "n": 0}
    n = len(records)
    keys = ["ang_err_deg", "az_err_deg", "el_err_deg", "top1", "top3", "logd_err"]
    return {
        "label": label,
        "n": n,
        **{f"mean_{k}": float(sum(r[k] for r in records) / n) for k in keys},
    }


@hydra.main(version_base="1.3", config_path="configs", config_name="train.yaml")
def main(cfg: DictConfig) -> None:
    if not cfg.get("ckpt_path"):
        raise ValueError("Must provide ckpt_path=<path>")

    log.info(f"Eval ckpt: {cfg.ckpt_path}")
    log.info(f"Probe type: {cfg.model.probe_type}")

    datamodule = hydra.utils.instantiate(cfg.data.data_module)
    datamodule.setup(stage="validate")
    from types import SimpleNamespace
    datamodule.trainer = SimpleNamespace(strategy=None, world_size=1, global_rank=0, local_rank=0, num_nodes=1)
    loader = datamodule.val_dataloader()
    if isinstance(loader, list):
        loader = loader[0]

    model = hydra.utils.instantiate(cfg.model)
    state = torch.load(cfg.ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["state_dict"], strict=False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.float().to(device).eval()

    records: List[Dict[str, Any]] = []
    for i, item in enumerate(loader):
        # CombinedLoader yields (batch, batch_idx, dataloader_idx)
        batch = item[0] if isinstance(item, tuple) else item
        batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                 for k, v in batch.items()}
        records.extend(_per_sample_metrics(model, batch, device))
        if (i + 1) % 20 == 0:
            log.info(f"[{i+1}/{len(loader)}] cum records: {len(records)}")

    by_source: Dict[str, List[Dict]] = defaultdict(list)
    for r in records:
        by_source[r["source"]].append(r)

    summary = {
        "overall": _aggregate(records, "overall"),
        "by_source": {src: _aggregate(recs, src) for src, recs in by_source.items()},
        "ckpt_path": cfg.ckpt_path,
        "probe_type": cfg.model.probe_type,
    }

    out_dir = Path(cfg.paths.output_dir) / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "b2_val_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    log.info(f"Wrote {out_path}")
    log.info(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
