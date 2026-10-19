#!/usr/bin/env python3
"""Build the official held-out eval manifest for tstep A1 / A2 / B2.

This is not a training-log harvest. Each row is one eval_a1_streaming or
eval_diag pass on the frozen ScanNet++ val split, using the trained checkpoint.
B1, mask, and C probes are excluded.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
INVENTORY = ROOT / "results/analysis/tstep_canonical_v2_seeded_v1/results.csv"
MANIFEST_DIR = ROOT / "artifacts/tstep_canonical_v2/manifests"
PROBES = ("streaming_depth", "view_consistency", "ego_belief_v2")
EVALUATOR = {
    "streaming_depth": ("a1", "vidfm3d/eval_a1_streaming.py", "mean_loss_depth"),
    "view_consistency": ("diag", "vidfm3d/eval_diag.py", "balanced_acc"),
    "ego_belief_v2": ("diag", "vidfm3d/eval_diag.py", "mean_ang_err_deg"),
}
STEP_RE = re.compile(r"step=(\d+)")
FIELDS = [
    "job_key",
    "unique_key",
    "evaluator",
    "entrypoint",
    "metric_name",
    "probe",
    "model",
    "timestep",
    "layer",
    "prefix",
    "experiment",
    "vfm_name",
    "video_channels",
    "streaming_feat_root",
    "feature_layer",
    "feature_timestep",
    "feat_postfix",
    "prefix_len",
    "ckpt_path",
    "train_output_dir",
    "scene_manifest",
    "extra_overrides_json",
]


def latest_ckpt(run_dir: Path) -> Path:
    ckpts = list(run_dir.glob("VideoProbe3D/*/checkpoints/*.ckpt"))
    if not ckpts:
        raise FileNotFoundError(f"no checkpoint under {run_dir}")
    def key(path: Path) -> tuple[int, float]:
        match = STEP_RE.search(path.name)
        step = int(match.group(1)) if match else -1
        return step, path.stat().st_mtime
    return max(ckpts, key=key)


def overrides(run_dir: Path) -> dict[str, str]:
    path = run_dir / ".hydra/overrides.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"missing hydra overrides: {path}")
    out: dict[str, str] = {}
    for line in path.read_text(errors="replace").splitlines():
        line = line.strip()
        if line.startswith("- ") and "=" in line:
            key, value = line[2:].split("=", 1)
            out[key] = value
    return out


def require(ov: dict[str, str], key: str, run_dir: Path) -> str:
    if key not in ov or not ov[key]:
        raise ValueError(f"{run_dir} missing override {key}")
    return ov[key]


def scene_manifest(prefix: int) -> Path:
    path = MANIFEST_DIR / f"inscene_c_h124_prefix{prefix}.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"missing frozen val/train manifest: {path}")
    return path


def extra_overrides(probe: str) -> dict[str, str]:
    if probe == "ego_belief_v2":
        return {"streaming_hidden_prefix_lengths": "[8,12,16,24]"}
    return {}


def build_rows() -> list[dict[str, str]]:
    with INVENTORY.open(newline="") as handle:
        inventory = [row for row in csv.DictReader(handle) if row["probe"] in PROBES]
    if len(inventory) != 216:
        raise ValueError(f"expected 216 A1/A2/B2 inventory rows, got {len(inventory)}")
    rows = []
    for item in inventory:
        run_dir = Path(item["output_dir"])
        if not (run_dir / "TRAIN_COMPLETE").is_file():
            raise FileNotFoundError(f"missing TRAIN_COMPLETE: {run_dir}")
        ov = overrides(run_dir)
        ckpt = latest_ckpt(run_dir)
        probe = item["probe"]
        prefix = int(item["prefix"])
        evaluator, entrypoint, metric = EVALUATOR[probe]
        prefix_len = require(ov, "prefix_len", run_dir)
        if int(prefix_len) != prefix:
            raise ValueError(f"prefix mismatch {run_dir}: inventory={prefix} override={prefix_len}")
        feat_ts = require(ov, "feature_timestep", run_dir)
        if int(feat_ts) != int(item["timestep"]):
            raise ValueError(f"timestep mismatch {run_dir}")
        rows.append({
            "job_key": item["unique_key"].replace(":", "__"),
            "unique_key": item["unique_key"],
            "evaluator": evaluator,
            "entrypoint": entrypoint,
            "metric_name": metric,
            "probe": probe,
            "model": item["model"],
            "timestep": item["timestep"],
            "layer": item["layer"],
            "prefix": item["prefix"],
            "experiment": require(ov, "experiment", run_dir),
            "vfm_name": require(ov, "vfm_name", run_dir),
            "video_channels": require(ov, "video_channels", run_dir),
            "streaming_feat_root": require(ov, "streaming_feat_root", run_dir),
            "feature_layer": require(ov, "feature_layer", run_dir),
            "feature_timestep": feat_ts,
            "feat_postfix": require(ov, "feat_postfix", run_dir),
            "prefix_len": prefix_len,
            "ckpt_path": str(ckpt),
            "train_output_dir": str(run_dir),
            "scene_manifest": str(scene_manifest(prefix)),
            "extra_overrides_json": json.dumps(extra_overrides(probe), sort_keys=True),
        })
    rows.sort(key=lambda row: (
        row["model"], int(row["timestep"]), int(row["layer"]), int(row["prefix"]), row["probe"],
    ))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default="tstep_eval_a1a2b2_20260923_r1")
    parser.add_argument(
        "--out-dir",
        default=str(ROOT / "artifacts/tstep_eval_a1a2b2_20260923_r1"),
    )
    parser.add_argument("--num-shards", type=int, default=3)
    args = parser.parse_args()
    out = Path(args.out_dir)
    jobs_dir = out / "jobs"
    shard_dir = jobs_dir / "shards"
    jobs_dir.mkdir(parents=True, exist_ok=True)
    shard_dir.mkdir(parents=True, exist_ok=True)
    rows = build_rows()
    tsv = jobs_dir / "eval_jobs.tsv"
    with tsv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    n = len(rows)
    size, extra = divmod(n, args.num_shards)
    start = 0
    shard_paths = []
    for index in range(args.num_shards):
        end = start + size + (1 if index < extra else 0)
        shard = shard_dir / f"shard_{index:02d}.tsv"
        with shard.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t")
            writer.writeheader()
            writer.writerows(rows[start:end])
        shard_paths.append(shard)
        start = end
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": args.run_id,
        "n_jobs": n,
        "probes": dict(Counter(row["probe"] for row in rows)),
        "evaluators": dict(Counter(row["evaluator"] for row in rows)),
        "eval_split": "val",
        "note": (
            "Official held-out evaluation, same split as the main paper. "
            "A1 uses eval_a1_streaming mean_loss_depth; "
            "A2 uses eval_diag balanced_acc; "
            "B2 uses eval_diag mean_ang_err_deg. "
            "Not training-time W&B val/* logs. B1 is excluded."
        ),
        "tsv": str(tsv),
        "shards": [str(path) for path in shard_paths],
    }
    (out / "PLAN.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
