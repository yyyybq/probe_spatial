#!/usr/bin/env python3
"""Read representative remote sidecars on an AOSS-capable compute node."""

from __future__ import annotations

import json
import os
import argparse
from pathlib import Path

import numpy as np
import torch

from analyses.se3_protocol_v2.wan_vjepa2_v2_data import (
    RESULT,
    atomic_json,
    read_jsonl,
)
from vidfm3d.utils import aoss_fs


def inspect_wan_target_route(routes: list[dict]) -> None:
    route = next(
        item for item in routes
        if item["cell"] == "wan_l15"
        and all(
            target["source_kind"] == "canonical_v2_exact_gap"
            for target in item["targets"].values()
        )
    )
    target_uris = {target["uri"] for target in route["targets"].values()}
    index_uris = {target["index_uri"] for target in route["targets"].values()}
    if len(target_uris) != 1 or len(index_uris) != 1:
        raise RuntimeError("representative Wan A/B/C target rows must share one layer payload/index")
    uri = target_uris.pop()
    index_uri = index_uris.pop()
    for required in (uri, uri + ".manifest.json", index_uri, index_uri + ".manifest.json"):
        if not aoss_fs.exists(required):
            raise FileNotFoundError(required)
    sidecar = aoss_fs.load_json(uri + ".manifest.json")
    index_sidecar = aoss_fs.load_json(index_uri + ".manifest.json")
    indices = np.asarray(aoss_fs.load_npy(index_uri), dtype=np.int64).reshape(-1)
    feature = aoss_fs.load_safetensors(uri)["feat"]
    if feature.shape[0] != len(indices) or feature.shape[-1] != 1536:
        raise RuntimeError(
            f"Wan target payload/index shape mismatch: feature={tuple(feature.shape)}, "
            f"indices={tuple(indices.shape)}"
        )
    role_rows = {}
    for role, target in route["targets"].items():
        matches = np.flatnonzero(indices == int(target["target_position"]))
        if len(matches) != 1:
            raise RuntimeError(f"Wan target {role} has {len(matches)} matching rows")
        row_index = int(matches[0])
        if not torch.isfinite(feature[row_index]).all():
            raise RuntimeError(f"Wan target {role} row contains non-finite values")
        role_rows[role] = {
            "target_position": int(target["target_position"]),
            "row_index": row_index,
        }
    atomic_json(RESULT / "inventory/wan_l15_target_route_smoke.json", {
        "status": "PASS_TARGET_DELTA_ROUTE_AND_PAYLOAD_SMOKE",
        "cell": "wan_l15",
        "sample_id": route["sample_id"],
        "scene_id": route["scene_id"],
        "feature_uri": uri,
        "index_uri": index_uri,
        "feature_shape": list(feature.shape),
        "index_count": len(indices),
        "role_rows": role_rows,
        "feature_sidecar": sidecar,
        "index_sidecar": index_sidecar,
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-only", action="store_true")
    args = parser.parse_args()
    routes = read_jsonl(RESULT / "data/route_plan.jsonl")
    if args.target_only:
        inspect_wan_target_route(routes)
        return
    output = {}
    for cell in ("wan_l15", "vjepa2_l11"):
        row = next(item for item in routes if item["cell"] == cell)
        uri = row["streaming"]["uri"]
        output[cell] = {
            "sample_id": row["sample_id"], "uri": uri,
            "expected_input_positions": row["streaming"]["expected_input_positions"],
            "sidecar": aoss_fs.load_json(uri + ".manifest.json"),
        }
    legacy = next(
        item for item in routes
        if item["cell"] == "vjepa2_l11"
        and item["streaming"]["source_kind"] == "exact_legacy_reference_fresh_gate_required"
    )
    legacy_uri = legacy["streaming"]["uri"]
    output["vjepa2_l11_legacy_reference"] = {
        "sample_id": legacy["sample_id"], "uri": legacy_uri,
        "expected_input_positions": legacy["streaming"]["expected_input_positions"],
        "catalog_status": legacy["streaming"]["catalog_status"],
        "sidecar": aoss_fs.load_json(legacy_uri + ".manifest.json"),
    }
    atomic_json(RESULT / "inventory/representative_sidecars.json", output)
    for path in RESULT.rglob("*"):
        if path.is_file():
            os.chmod(path, 0o644)
        elif path.is_dir():
            os.chmod(path, 0o755)


if __name__ == "__main__":
    main()
