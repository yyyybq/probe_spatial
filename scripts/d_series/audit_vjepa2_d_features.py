#!/usr/bin/env python3
"""One-time full identity/header audit for D-series V-JEPA2 missing-role fills."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from safetensors import safe_open

from scripts.d_series.feature_store import MANIFEST_SHA256, feature_paths, sha256_file
from scripts.d_series.run_d1 import REVISIONS

MODEL = "vjepa2_vitl256"
SPACES = ("prefix", "isolated_target", "native_target", "native_pred")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if hashlib.sha256(args.manifest.read_bytes()).hexdigest() != MANIFEST_SHA256:
        raise ValueError("manifest hash mismatch")
    rows = [json.loads(line) for line in args.manifest.open()]
    sample_ids = [row["sample_id"] for row in rows]
    parent_ids = sorted({row["parent_sample_id"] for row in rows})
    counts, failures, payloads = {}, [], []
    for space in SPACES:
        ids = parent_ids if space == "prefix" else sample_ids
        complete = 0
        for sample_id in ids:
            payload, receipt_path = feature_paths(args.feature_root, MODEL, space, sample_id)
            try:
                receipt = json.loads(receipt_path.read_text())
                expected = {
                    "manifest_sha256": MANIFEST_SHA256, "model": MODEL, "space": space,
                    "sample_id": sample_id, "model_revision": REVISIONS[MODEL],
                    "payload_bytes": payload.stat().st_size,
                }
                mismatch = {key: [receipt.get(key), value] for key, value in expected.items() if receipt.get(key) != value}
                if mismatch:
                    raise ValueError(mismatch)
                with safe_open(str(payload), framework="pt", device="cpu") as handle:
                    keys = sorted(handle.keys())
                    shapes = {key: list(handle.get_slice(key).get_shape()) for key in keys}
                if shapes != {key: value["shape"] for key, value in sorted(receipt["tensors"].items())}:
                    raise ValueError(f"tensor header mismatch: {shapes}")
                payloads.append((payload, receipt["payload_sha256"]))
                complete += 1
            except Exception as error:
                if len(failures) < 100:
                    failures.append({"space": space, "sample_id": sample_id, "error": repr(error)})
        counts[space] = {"expected": len(ids), "complete": complete}
    # Full identity/header coverage above; hash deterministic boundary/quartile probes only.
    ordered = sorted(payloads, key=lambda item: str(item[0]))
    indices = sorted(set([0, len(ordered) // 4, len(ordered) // 2, 3 * len(ordered) // 4, len(ordered) - 1]))
    probes = []
    for index in indices:
        path, expected = ordered[index]
        actual = sha256_file(path)
        probes.append({"path": str(path), "expected_sha256": expected, "actual_sha256": actual, "match": actual == expected})
        if actual != expected:
            failures.append({"space": "probe", "sample_id": str(path), "error": "payload SHA mismatch"})
    status = "COMPLETE_VALID" if not failures and all(value["complete"] == value["expected"] for value in counts.values()) else "FAILED_VALIDATION"
    result = {
        "schema_version": 1, "status": status, "model": MODEL, "model_revision": REVISIONS[MODEL],
        "manifest": str(args.manifest), "manifest_sha256": MANIFEST_SHA256,
        "expected": {"scenes": 681, "windows": 2721, "samples": 5442, "train_samples": 4890, "dev_samples": 552},
        "resolved": {"scenes": len({r['scene_id'] for r in rows}), "windows": len(parent_ids), "samples": len(rows),
                     "train_samples": sum(r['split'] == 'train' for r in rows), "dev_samples": sum(r['split'] == 'dev' for r in rows)},
        "roles": counts, "full_check": "all expected sidecars, identity fields, payload sizes and safetensors headers",
        "payload_hash_policy": "five deterministic probes; per-payload SHA generated transactionally by extractor",
        "payload_hash_probes": probes, "failures": failures,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if status == "COMPLETE_VALID" else 2


if __name__ == "__main__":
    raise SystemExit(main())
