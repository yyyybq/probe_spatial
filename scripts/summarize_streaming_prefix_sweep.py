#!/usr/bin/env python3
"""Summarize Streaming Prefix Depth sweep runs into one CSV.

The script is intentionally tolerant: diagnostic eval summaries are merged when
present, but plain depth/camera pixalign runs are still reported from Hydra
config metadata and checkpoint presence.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
from pathlib import Path
from typing import Any


PRIMARY_METRIC = {
    "pixalign": ("val/loss", "min"),
    "camera": ("val/loss", "min"),
    "view_consistency": ("balanced_acc", "max"),
    "abnormal": ("pair_acc", "max"),
    "ego_belief": ("mean_az_err_deg", "min"),
    "ego_belief_v2": ("mean_ang_err_deg", "min"),
    "action_dynamics": ("global_R@1", "max"),
    "path_integration": ("global_R@1", "max"),
    "counterfactual": ("intervention_validity", "max"),
}


def _read_json(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml

        with path.open() as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def _get_nested(d: dict[str, Any], dotted: str, default: Any = None) -> Any:
    cur: Any = d
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def _infer_int(row: dict[str, Any], key: str, run_name: str, pattern: str) -> int | None:
    value = row.get(key)
    if value is not None:
        try:
            return int(value)
        except Exception:
            pass
    match = re.search(pattern, run_name)
    if match:
        return int(match.group(1))
    return None


def _infer_layer(row: dict[str, Any], run_name: str) -> int | None:
    value = row.get("feature_layer")
    if value is not None:
        try:
            return int(value)
        except Exception:
            pass
    for text in (row.get("feat_postfix"), run_name):
        match = re.search(r"layerneg(\d+)", str(text))
        if match:
            return -int(match.group(1))
        match = re.search(r"layer(?:neg)?(-?\d+)", str(text))
        if match:
            return int(match.group(1))
    return None


def _checkpoint_status(run_dir: Path) -> tuple[str | None, bool]:
    last = run_dir / "checkpoints" / "last.ckpt"
    if last.exists():
        return str(last), True
    ckpt_candidates = [
        p
        for p in glob.glob(str(run_dir / "**" / "checkpoints" / "*.ckpt"), recursive=True)
        if os.path.exists(p)
    ]
    ckpts = sorted(
        ckpt_candidates,
        key=lambda p: os.path.getmtime(p),
        reverse=True,
    )
    if ckpts:
        return ckpts[0], False
    return None, False


def _better(a: dict[str, Any], b: dict[str, Any], metric: str, direction: str) -> bool:
    av = _as_float(a.get(metric))
    bv = _as_float(b.get(metric))
    if av != av:
        return False
    if bv != bv:
        return True
    return av > bv if direction == "max" else av < bv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", default="logs/inscene15k_streaming/runs")
    parser.add_argument("--pattern", default="inscene15k_streaming_*")
    parser.add_argument("--split", default="val")
    parser.add_argument("--metric", default=None)
    parser.add_argument("--direction", choices=["max", "min"], default=None)
    parser.add_argument("--output", default="streaming_prefix_sweep.csv")
    args = parser.parse_args()

    run_dirs = sorted(Path(p) for p in glob.glob(os.path.join(args.runs_root, args.pattern)))
    rows: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        if not run_dir.is_dir():
            continue
        cfg = _read_yaml(run_dir / ".hydra" / "config.yaml")
        eval_path = run_dir / "eval" / f"{args.split}_summary.json"
        eval_summary = _read_json(eval_path) if eval_path.exists() else {}

        row: dict[str, Any] = {}
        row.update(eval_summary)
        run_name = run_dir.name
        row["run"] = run_name
        row["run_dir"] = str(run_dir)
        row["eval_summary_path"] = str(eval_path) if eval_path.exists() else ""
        row["has_eval_summary"] = bool(eval_path.exists())

        row.setdefault("job_name", cfg.get("job_name"))
        row.setdefault("task_name", cfg.get("task_name"))
        row.setdefault("probe_type", _get_nested(cfg, "model.probe_type"))
        row.setdefault("vfm_name", cfg.get("vfm_name"))
        row.setdefault("streaming_feat_root", cfg.get("streaming_feat_root"))
        row.setdefault("feature_layer", cfg.get("feature_layer"))
        row.setdefault("feature_timestep", cfg.get("feature_timestep"))
        row.setdefault("feat_postfix", cfg.get("feat_postfix"))
        row.setdefault("prefix_min_len", cfg.get("prefix_min_len"))
        row.setdefault("prefix_max_len", cfg.get("prefix_max_len"))
        row.setdefault("prefix_stride", cfg.get("prefix_stride"))
        row.setdefault("video_channels", cfg.get("video_channels"))

        row["feature_layer"] = _infer_layer(row, run_name)
        row["prefix_min_len"] = _infer_int(row, "prefix_min_len", run_name, r"pmin(\d+)")
        row["prefix_max_len"] = _infer_int(row, "prefix_max_len", run_name, r"pmax(\d+)")
        row["prefix_stride"] = _infer_int(row, "prefix_stride", run_name, r"_s(\d+)")
        ckpt_path, has_last = _checkpoint_status(run_dir)
        row["checkpoint_path"] = ckpt_path or ""
        row["has_checkpoint"] = ckpt_path is not None
        row["has_last_ckpt"] = has_last
        rows.append(row)

    if not rows:
        raise SystemExit("No streaming prefix runs matched.")

    rows.sort(
        key=lambda r: (
            r.get("prefix_min_len") is None,
            r.get("prefix_min_len") or 0,
            r.get("prefix_max_len") is None,
            r.get("prefix_max_len") or 0,
            r.get("prefix_stride") is None,
            r.get("prefix_stride") or 0,
            r.get("feature_layer") is None,
            r.get("feature_layer") or 0,
            r.get("run") or "",
        )
    )

    probe_type = str(rows[0].get("probe_type") or "pixalign")
    default_metric, default_direction = PRIMARY_METRIC.get(probe_type, ("val/loss", "min"))
    metric = args.metric or default_metric
    direction = args.direction or default_direction

    best = None
    if any(row.get(metric) is not None for row in rows):
        best = rows[0]
        for row in rows[1:]:
            if _better(row, best, metric, direction):
                best = row

    preferred = [
        "run",
        "probe_type",
        "vfm_name",
        "feature_layer",
        "feature_timestep",
        "prefix_min_len",
        "prefix_max_len",
        "prefix_stride",
        "streaming_feat_root",
        "has_checkpoint",
        "has_last_ckpt",
        "has_eval_summary",
        metric,
        "checkpoint_path",
        "eval_summary_path",
        "job_name",
    ]
    keys = []
    for key in preferred:
        if key not in keys:
            keys.append(key)
    for key in sorted({k for row in rows for k in row.keys()} - set(keys)):
        keys.append(key)

    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {args.output} with {len(rows)} rows")
    print(f"metric={metric} direction={direction}")
    if best is not None:
        print(
            "best="
            f"prefix({best.get('prefix_min_len')},{best.get('prefix_max_len')},{best.get('prefix_stride')}) "
            f"layer={best.get('feature_layer')} score={best.get(metric)} run={best.get('run')}"
        )
    else:
        print(f"[info] no rows contained metric '{metric}'; CSV contains run metadata only")


if __name__ == "__main__":
    main()
