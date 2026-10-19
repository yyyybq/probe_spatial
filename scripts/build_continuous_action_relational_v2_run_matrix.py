#!/usr/bin/env python3
"""Build the complete model/variant/seed matrix from accepted checkpoints."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

MODELS = ("dino", "f3r", "vjepa2", "wan")
STAGES = {"dino": "L0-project-final", "f3r": "L24", "vjepa2": "L23", "wan": "L29/t749"}
JOBS = {"dino": "pt-gvms1pw4", "f3r": "pt-gcbibpx9", "vjepa2": "pt-c82h4nx9", "wan": "pt-9p27k50f"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    rows = []
    for model in MODELS:
        integrity = json.loads((args.run_root / model / "training_and_integrity.json").read_text())
        if integrity["status"] != "COMPLETE_VALID" or len(integrity["training"]) != 25:
            raise ValueError(f"invalid training matrix for {model}")
        for key, value in sorted(integrity["training"].items()):
            variant, seed = key.split("|")
            rows.append({
                "model": model,
                "feature_stage": STAGES[model],
                "variant": variant,
                "seed": int(seed),
                "head": value["head"],
                "selected_epoch_dev": value["selected_epoch"],
                "parameters": value["parameters"],
                "checkpoint_sha256": value["checkpoint_sha256"],
                "validation_samples": integrity["validation_records"],
                "job_id": JOBS[model],
                "pool": "zoetrope",
                "gpu_count": 1,
                "status": "COMPLETE_VALID",
            })
    if len(rows) != 100:
        raise ValueError(len(rows))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"status": "COMPLETE_VALID", "rows": len(rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
