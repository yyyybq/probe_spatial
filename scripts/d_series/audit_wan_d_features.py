#!/usr/bin/env python3
"""Fail-closed full audit for the Wan D-series feature cache."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from safetensors import safe_open

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.d_series.feature_store import MANIFEST_SHA256, feature_paths, sha256_file
from scripts.d_series.run_d1 import REVISIONS


MODEL = "wan21_t2v_1p3b"
ROLE_SPECS = {
    "prefix": {
        "ids": "parent",
        "tensors": {
            "context_pooled_sequence": [21, 1536],
            "context_spatial": [30, 52, 1536],
        },
    },
    "isolated_target": {
        "ids": "sample",
        "tensors": {"pooled": [1536], "spatial": [30, 52, 1536]},
    },
}
COMMON_RECEIPT = {
    "model_id": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
    "model_revision": REVISIONS[MODEL],
    "layer_zero_based": 20,
    "final_layernorm": False,
    "input_frames": 81,
    "resize": [480, 832],
    "prompt": "",
    "target_or_future_text_used": False,
    "timestep_argument": 749,
    "timestep_semantics": "scheduler_timesteps_index",
    "resolved_scheduler_timestep": 502,
    "scheduler": "UniPCMultistepScheduler",
    "flow_shift": 3.0,
    "noise_seed_scheme": "sha256(42|space_and_sample_id)_uint32_global_rng_reset_before_forward",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest_sha = hashlib.sha256(args.manifest.read_bytes()).hexdigest()
    if manifest_sha != MANIFEST_SHA256:
        raise ValueError(f"manifest hash mismatch: {manifest_sha}")
    rows = [json.loads(line) for line in args.manifest.open()]
    sample_ids = [row["sample_id"] for row in rows]
    parents = sorted({row["parent_sample_id"] for row in rows})
    by_sample = {row["sample_id"]: row for row in rows}
    by_parent = {row["parent_sample_id"]: row for row in rows}
    counts, failures, payloads = {}, [], []

    for role, spec in ROLE_SPECS.items():
        ids = parents if spec["ids"] == "parent" else sample_ids
        complete = 0
        for identifier in ids:
            payload, receipt_path = feature_paths(args.feature_root, MODEL, role, identifier)
            try:
                receipt = json.loads(receipt_path.read_text())
                expected = {
                    "manifest_sha256": MANIFEST_SHA256,
                    "model": MODEL,
                    "space": role,
                    "sample_id": identifier,
                    "model_revision": REVISIONS[MODEL],
                    "payload_bytes": payload.stat().st_size,
                } | COMMON_RECEIPT
                if role == "prefix":
                    expected["raw_frame_ids"] = by_parent[identifier]["prefix_raw_frame_ids"]
                    expected["future_rgb_access"] = False
                else:
                    row = by_sample[identifier]
                    expected["horizon"] = row["horizon"]
                    expected["target_raw_frame_id"] = row["target_raw_frame_id"]
                mismatch = {key: [receipt.get(key), value] for key, value in expected.items()
                            if receipt.get(key) != value}
                if mismatch:
                    raise ValueError(mismatch)
                with safe_open(str(payload), framework="pt", device="cpu") as handle:
                    shapes = {key: list(handle.get_slice(key).get_shape()) for key in sorted(handle.keys())}
                if shapes != spec["tensors"]:
                    raise ValueError(f"tensor header mismatch: {shapes}")
                declared = receipt.get("tensors", {})
                expected_declared = {
                    key: {"shape": shape, "dtype": "float16"} for key, shape in spec["tensors"].items()
                }
                if declared != expected_declared:
                    raise ValueError(f"receipt tensor declaration mismatch: {declared}")
                payloads.append((payload, receipt["payload_sha256"]))
                complete += 1
            except Exception as error:  # Keep scanning to report coverage, but fail closed.
                if len(failures) < 100:
                    failures.append({"role": role, "sample_id": identifier, "error": repr(error)})
        counts[role] = {"expected": len(ids), "complete": complete}

    ordered = sorted(payloads, key=lambda value: str(value[0]))
    probes = []
    if ordered:
        indices = sorted({0, len(ordered) // 4, len(ordered) // 2, 3 * len(ordered) // 4, len(ordered) - 1})
        for index in indices:
            path, expected = ordered[index]
            actual = sha256_file(path)
            probes.append({"path": str(path), "expected_sha256": expected,
                           "actual_sha256": actual, "match": actual == expected})
            if actual != expected:
                failures.append({"role": "probe", "sample_id": str(path), "error": "payload SHA mismatch"})

    status = "COMPLETE_VALID" if not failures and all(
        value["complete"] == value["expected"] for value in counts.values()
    ) else "FAILED_VALIDATION"
    result = {
        "schema_version": 1,
        "status": status,
        "model": MODEL,
        "model_revision": REVISIONS[MODEL],
        "manifest": str(args.manifest),
        "manifest_sha256": MANIFEST_SHA256,
        "expected": {"scenes": 681, "windows": 2721, "samples": 5442,
                     "train_samples": 4890, "dev_samples": 552},
        "resolved": {"scenes": len({row["scene_id"] for row in rows}), "windows": len(parents),
                     "samples": len(rows), "train_samples": sum(row["split"] == "train" for row in rows),
                     "dev_samples": sum(row["split"] == "dev" for row in rows)},
        "roles": counts,
        "protocol_contract": COMMON_RECEIPT,
        "full_check": "all expected sidecars, identity fields, extraction contract, payload sizes and safetensors headers",
        "payload_hash_policy": "five deterministic probes; per-payload SHA generated transactionally by extractor",
        "payload_hash_probes": probes,
        "failures": failures,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if status == "COMPLETE_VALID" else 2


if __name__ == "__main__":
    raise SystemExit(main())
