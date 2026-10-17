#!/usr/bin/env python3
"""Build the 2026-10-16 retrain queue from the release inventories.

Two groups, derived from ``metadata/release_inventory/{canonical_v2,timestep}.csv``
after ``update_release_inventory_20261015.py``:

* ``TSTEP_MINIMAX_L20_L25_FEATURE_OVERFLOW``: every timestep MiniMax-H3 run at
  layers 20/25. The fp16 ``FEAT_STREAMING_CANONICAL_V2_SEEDED_V1`` caches at
  these layers contain inf (job pt-et0ezc14); retraining needs a re-extracted
  bf16 cache first.
* ``B2_NONFINITE_WEIGHTS_FP16_PASSTHROUGH``: every other run whose checkpoint
  tensors are non-finite. All are ``ego_belief_v2`` heads trained with fp16
  feature pass-through on high-magnitude caches (CogVideoX, Aether, MiniMax
  layer 15); the CogVideoX layer-30/35/40 reruns with ``PROBE_FP16_PASSTHROUGH=0``
  are finite, so these retrain on the existing caches with that setting.

Per-run GPU-hour estimates are single-GPU medians/p90 of the finished timestep
MiniMax runs (CogVideoX rows use the CogVideoX medians). They are planning
figures, not measurements of the future jobs.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INVENTORY = ROOT / "metadata/release_inventory"
OUT = INVENTORY / "retrain_queue_20261016.csv"

# (median_h, p90_h) single-GPU wall time, from run_manifest -> TRAIN_COMPLETE mtimes.
HOURS = {
    ("minimax_h3", "streaming_depth"): (13.70, 19.80),
    ("minimax_h3", "streaming_mask"): (17.49, 35.63),
    ("minimax_h3", "view_consistency"): (0.95, 1.48),
    ("minimax_h3", "ego_belief"): (6.94, 12.83),
    ("minimax_h3", "ego_belief_v2"): (6.97, 13.11),
    ("minimax_h3", "action_dynamics"): (1.11, 1.68),
    ("minimax_h3", "path_integration"): (1.29, 1.79),
    ("minimax_h3", "counterfactual"): (1.24, 1.83),
    ("cogvideox", "ego_belief_v2"): (8.33, 15.18),
    ("aether", "ego_belief_v2"): (8.33, 15.18),  # same backbone width as CogVideoX
}
FIELDS = [
    "queue_group", "experiment_id", "inventory", "model", "probe", "layer", "disk_layer", "prefix",
    "seed", "timestep", "current_correctness_status", "current_result_status", "checkpoint_tensor_status",
    "reason", "prerequisite", "train_setting", "experiment_config", "est_gpu_hours_median", "est_gpu_hours_p90",
    "old_output_path", "old_checkpoint_path",
]


def rows_from(path: Path):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()

    queue = []
    canonical = rows_from(INVENTORY / "canonical_v2.csv")
    timestep = rows_from(INVENTORY / "timestep.csv")

    def add(row, inventory, group, reason, prerequisite, setting):
        probe = row["probe"]
        med, p90 = HOURS.get((row["model"], probe), (None, None))
        cfg = f"inscene15k_streaming/{probe}_{row['model']}_v1"
        if probe == "ego_belief_v2" and row["model"] in {"cogvideox", "aether"}:
            cfg = f"inscene15k_streaming/ego_belief_v2_{row['model']}_v1"
        queue.append({
            "queue_group": group, "experiment_id": row["experiment_id"], "inventory": inventory,
            "model": row["model"], "probe": probe, "layer": row["layer"],
            "disk_layer": row.get("disk_layer") or row["layer"], "prefix": row["prefix"], "seed": row["seed"],
            "timestep": row["timestep"], "current_correctness_status": row["correctness_status"],
            "current_result_status": row["result_status"], "checkpoint_tensor_status": row.get("checkpoint_tensor_status", ""),
            "reason": reason, "prerequisite": prerequisite, "train_setting": setting, "experiment_config": cfg,
            "est_gpu_hours_median": "" if med is None else f"{med:.2f}",
            "est_gpu_hours_p90": "" if p90 is None else f"{p90:.2f}",
            "old_output_path": row.get("train_output_dir") or row.get("output_path", ""),
            "old_checkpoint_path": row.get("checkpoint_path", ""),
        })

    for row in timestep:
        if row["entry_level"] != "run":
            continue
        if row["model"] == "minimax_h3" and row["layer"] in {"20", "25"}:
            add(row, "timestep", "TSTEP_MINIMAX_L20_L25_FEATURE_OVERFLOW",
                "fp16 seeded-v1 cache at this layer contains inf (fp16 ceiling); weights NaN or inputs invalid",
                "re-extract FEAT_STREAMING/FEAT_TARGET seeded-v1 MiniMax-H3 t500/t1000 layers 20/25 as bf16 (new namespace)",
                "PROBE_FP16_PASSTHROUGH=0; bf16 feature namespace; same seed/protocol/manifests")
        elif row.get("checkpoint_tensor_status", "").startswith("NONFINITE"):
            add(row, "timestep", "B2_NONFINITE_WEIGHTS_FP16_PASSTHROUGH",
                "checkpoint tensors non-finite; fp16 pass-through of high-magnitude features",
                "none (existing cache is finite)",
                "PROBE_FP16_PASSTHROUGH=0; otherwise identical overrides")
    for row in canonical:
        if row["entry_level"] != "run":
            continue
        if row.get("checkpoint_tensor_status", "").startswith("NONFINITE"):
            add(row, "canonical_v2", "B2_NONFINITE_WEIGHTS_FP16_PASSTHROUGH",
                "checkpoint tensors non-finite; fp16 pass-through of high-magnitude features",
                "none (existing cache is finite)",
                "PROBE_FP16_PASSTHROUGH=0; otherwise identical overrides")

    with args.out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(queue)

    counts = Counter(q["queue_group"] for q in queue)
    by_probe = Counter((q["queue_group"], q["model"], q["probe"]) for q in queue)
    med = sum(float(q["est_gpu_hours_median"] or 0) for q in queue)
    p90 = sum(float(q["est_gpu_hours_p90"] or 0) for q in queue)
    print(f"wrote {args.out} rows={len(queue)}")
    for group, n in sorted(counts.items()):
        g_med = sum(float(q["est_gpu_hours_median"] or 0) for q in queue if q["queue_group"] == group)
        g_p90 = sum(float(q["est_gpu_hours_p90"] or 0) for q in queue if q["queue_group"] == group)
        print(f"  {group}: {n} runs, est {g_med:.0f} GPU-h (median) / {g_p90:.0f} GPU-h (p90)")
    for key, n in sorted(by_probe.items()):
        print("   ", key, n)
    print(f"total est {med:.0f} GPU-h (median) / {p90:.0f} GPU-h (p90)")


if __name__ == "__main__":
    main()
