#!/usr/bin/env python3
"""Fail closed unless paired canary outputs are protocol-equivalent."""

from __future__ import annotations

import argparse
import json
import math
import pathlib


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", required=True, type=pathlib.Path)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    parser.add_argument("--parameter-policy", choices=["exact", "skip"], default="exact")
    args = parser.parse_args()
    results = []
    failures = []
    old_files = sorted(args.directory.glob("old__*.json"))
    for old_path in old_files:
        suffix = old_path.name[len("old__"):]
        new_path = args.directory / f"optimized__{suffix}"
        if not new_path.is_file():
            failures.append(f"missing optimized pair for {old_path.name}")
            continue
        old, new = json.loads(old_path.read_text()), json.loads(new_path.read_text())
        item = {"case": suffix[:-5], "checks": {}}
        for key in ("task", "backend", "timestep", "layer", "prefix", "split", "index",
                    "manifest_sha256", "manifest_sample_id", "effective_batch_size",
                    "num_workers", "num_workers_val", "precision", "seed"):
            ok = old[key] == new[key]
            item["checks"][key] = ok
            if not ok:
                failures.append(f"{suffix}: {key}: {old[key]!r} != {new[key]!r}")
        if "empty_mask_fixture" in old or "empty_mask_fixture" in new:
            ok = old.get("empty_mask_fixture") == new.get("empty_mask_fixture")
            item["checks"]["empty_mask_fixture"] = ok
            if not ok:
                failures.append(f"{suffix}: empty-mask fixture flags differ")
        old_fields, new_fields = old["required_fields"], new["required_fields"]
        if set(old_fields) != set(new_fields):
            failures.append(f"{suffix}: required field set differs")
        for key in sorted(set(old_fields) & set(new_fields)):
            # The optimized path deliberately reuses the same torch operations;
            # require bit identity (atol=rtol=0), which is stricter than a loose
            # floating tolerance and prevents masking semantic drift.
            ok = old_fields[key] == new_fields[key]
            item["checks"][f"field:{key}:bit_exact"] = ok
            if not ok:
                failures.append(f"{suffix}: required tensor differs: {key}")
        loss_ok = len(old["losses"]) == len(new["losses"]) and all(
            math.isclose(a, b, rel_tol=0.0, abs_tol=0.0) or (math.isnan(a) and math.isnan(b))
            for a, b in zip(old["losses"], new["losses"])
        )
        item["checks"]["loss_trajectory_bit_exact"] = loss_ok
        if not loss_ok:
            failures.append(f"{suffix}: loss trajectory differs: {old['losses']} vs {new['losses']}")
        if args.parameter_policy == "exact":
            initial_ok = old["initial_parameter_sha256"] == new["initial_parameter_sha256"]
            item["checks"]["initial_parameters_bit_exact"] = initial_ok
            if not initial_ok:
                failures.append(f"{suffix}: initial parameters differ")
            state_ok = old["final_parameter_sha256"] == new["final_parameter_sha256"]
            item["checks"]["short_train_final_parameters_bit_exact"] = state_ok
            if not state_ok:
                failures.append(f"{suffix}: short-train parameters differ")
        item["performance"] = {
            "old_sample_tensor_bytes": old["sample_tensor_bytes"],
            "new_sample_tensor_bytes": new["sample_tensor_bytes"],
            "old_data_seconds": old["data_seconds_cold_then_steady"],
            "new_data_seconds": new["data_seconds_cold_then_steady"],
            "old_h2d_ms": old["h2d_ms"],
            "new_h2d_ms": new["h2d_ms"],
            "old_gpu_compute_ms": old["gpu_compute_ms"],
            "new_gpu_compute_ms": new["gpu_compute_ms"],
            "old_max_memory": old["max_memory_allocated_bytes"],
            "new_max_memory": new["max_memory_allocated_bytes"],
        }
        results.append(item)
    report = {
        "schema": "tstep-probe-canary-comparison-v1",
        "parameter_policy": args.parameter_policy,
        "float_policy": (
            "required tensors and losses require bit identity (atol=rtol=0); parameter comparison skipped"
            if args.parameter_policy == "skip" else
            "required tensors, losses, and trainable short-train parameters require bit identity (atol=rtol=0)"
        ),
        "pairs": len(results),
        "passed": not failures,
        "failures": failures,
        "results": results,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
