#!/usr/bin/env python3
"""Build official Wan/Aether streaming gate requests.

This is a new campaign. It does not modify the frozen 816 request SHA
``5e0854c11f5141780ae797da3f1368cffec270a42b9ec7ab2d4cf31908597cd5``.

Every request is keyed by ``(vfm, scene, layer, timestep, prefix, window_id)``
and points at ``FEAT_STREAMING_CANONICAL_V2_GAP_DELTA_V1``. Missing payloads
are still requested so the later gate can fail-close them.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
VFMS = ("aether", "wan")
PREFIXES = (8, 12, 16, 24)
EXPECTED_RECORDS_PER_PREFIX = 2721
EXPECTED_SPECS = {"aether": 4, "wan": 28}
EXPECTED_REQUESTS = {"aether": 10884, "wan": 76188}
GAP_NAMESPACE = "FEAT_STREAMING_CANONICAL_V2_GAP_DELTA_V1"
AOSS_ROOT = f"aoss://baiqiao/InsScene-15K/{GAP_NAMESPACE}"
CANONICAL_PARENT_SOURCE_SHA256 = (
    "ebc496eb0587cda372ccf29a7c5244ef454096151f73c8eed9511967ded1902d"
)
CANONICAL_UNIVERSE_SHA256 = (
    "4771e2859c43889bd5fb75f5876bcf493758f4187336c9bd7eeca5432acbac9b"
)
FROZEN_PROTOCOL_FILE_SHA256 = (
    "02e17c40b59ae5075a23641cb41ce5c46c2d7cda6333d40c48d988dffd106c7d"
)
FROZEN_PROTOCOL_SEMANTIC_SHA256 = (
    "215c720c8537f4dfae6243bedcc7339fbb7e3ba8a6ca2dcaf04f49cd4b738e4f"
)
INVENTORY = (
    ROOT
    / "artifacts/windowfix_v2/feature_inventory/"
    "canonical_v2_feature_specs.windowfix_v2_b34c099e010d.json"
)
MANIFEST_TEMPLATE = (
    ROOT / "artifacts/tstep_canonical_v2/manifests/inscene_c_h124_prefix{prefix}.jsonl"
)
CANONICAL_PARENT_SOURCE = (
    ROOT / "manifests/c_retrain_20260722/inscene_c_h124_prefix24.jsonl"
)
NAMESPACE_LOCK = (
    ROOT / "artifacts/windowfix_v2/canonical_namespace_lock.02e17c40b59a.json"
)
EXPERIMENTS_1096 = (
    ROOT / "experiments/task_status_1096_before_expand_20260822.csv"
)
PROMPT_CONTRACTS = {
    "aether": {"mode": "exact", "value": ""},
    "wan": {"mode": "exact", "value": "."},
}
PREPROCESS_CONTRACTS = {
    "aether": {"resize": [480, 720], "num_frames": 81, "temporal_padding": "repeat_tail"},
    "wan": {"resize": [480, 832], "num_frames": 81, "temporal_padding": "repeat_tail"},
}
EXPECTED_SHAPES = {
    "aether": [2, 11, 30, 45, 3072],
    "wan": [21, 30, 52, 1536],
}
REQUIRED_CHECKS = [
    "source_exists",
    "sidecar_exists",
    "file_and_size_match",
    "raw_ids_match",
    "model_layer_timestep_match",
    "prompt_preprocess_match_or_not_applicable",
    "shape_dtype_match",
    "safetensors_header_valid",
]


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_binding(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def load_canonical(path: Path, prefix: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, int, int]] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            value = json.loads(line)
            raw_ids = [int(item) for item in value["streaming_indices"]]
            row = {
                "scene_id": str(value["scene_id"]),
                "sample_id": str(value["sample_id"]),
                "split": str(value["split"]),
                "window_id": int(value["streaming_window_id"]),
                "tail": int(value["streaming_tail"]),
                "prefix": int(value["prefix_length"]),
                "raw_ids": raw_ids,
            }
            if row["prefix"] != prefix or row["tail"] != prefix - 1:
                raise ValueError(f"{path}:{line_number}: prefix/tail mismatch")
            if len(raw_ids) != prefix:
                raise ValueError(f"{path}:{line_number}: raw ID length mismatch")
            key = (row["scene_id"], row["window_id"], row["tail"])
            if key in seen:
                raise ValueError(f"{path}:{line_number}: duplicate key {key}")
            seen.add(key)
            rows.append(row)
    if len(rows) != EXPECTED_RECORDS_PER_PREFIX:
        raise ValueError(
            f"{path}: expected {EXPECTED_RECORDS_PER_PREFIX} rows, found {len(rows)}"
        )
    return rows


def relative_feature(spec: dict[str, Any], row: dict[str, Any]) -> str:
    return (
        f"{spec['vfm']}/scannetpp/{row['scene_id']}/"
        f"window_{int(row['window_id']):04d}/"
        f"prefix_{int(row['tail']):06d}/{spec['feature_filename']}"
    )


def build_request(spec: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
    vfm = str(spec["vfm"])
    rel = relative_feature(spec, row)
    source_uri = f"{AOSS_ROOT}/{rel}"
    return {
        "schema": "LEGACY_EXACT_REFERENCE_REMOTE_GATE_REQUEST_V1",
        "unit_id": f"{spec['spec_id']}:{row['sample_id']}",
        "source_uri": source_uri,
        "sidecar_uri": f"{source_uri}.manifest.json",
        "source_index_uri": (
            f"{AOSS_ROOT}/{vfm}/scannetpp/{row['scene_id']}/prefix_index.npy"
        ),
        "source_index_record": {
            "window_id": int(row["window_id"]),
            "tail": int(row["tail"]),
        },
        "expected": {
            "vfm": vfm,
            "model_id": spec["model_id"],
            "source": "scannetpp",
            "scene_id": row["scene_id"],
            "window_id": int(row["window_id"]),
            "prefix": int(row["prefix"]),
            "feature_layer": int(spec["feature_layer"]),
            "timestep": int(spec["timestep"]),
            "feature_filename": spec["feature_filename"],
            "mode": "streaming_prefix",
            "feature_namespace": GAP_NAMESPACE,
            "canonical_source_sha256": CANONICAL_PARENT_SOURCE_SHA256,
            "prompt_contract": PROMPT_CONTRACTS[vfm],
            "preprocess_identity": {
                "frame_index_schema": "image_mask_intersection_v1",
                **PREPROCESS_CONTRACTS[vfm],
            },
            "frame_index_schema": "image_mask_intersection_v1",
            "tensor_key": "feat",
            "expected_shape": list(EXPECTED_SHAPES[vfm]),
            "shape_contract": "sidecar_and_safetensors_header_both_equal_exact_expected_shape",
            "expected_dtype": "float16",
            "dtype_contract": "sidecar_and_safetensors_header_both_equal_expected_dtype",
            "raw_ids": list(row["raw_ids"]),
            "raw_ids_sha256": sha256_bytes(canonical_json(row["raw_ids"]).encode("utf-8")),
        },
        "required_checks": list(REQUIRED_CHECKS),
    }


def write_experiments(destination: Path) -> dict[str, Any]:
    with EXPERIMENTS_1096.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row["model"] in VFMS]
    if not rows:
        raise ValueError("no wan/aether experiments in 1096 CSV")
    fieldnames = list(rows[0].keys())
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    counts = Counter(row["model"] for row in rows)
    if counts["wan"] != 192 or counts["aether"] != 28:
        raise ValueError(f"unexpected experiment counts: {dict(counts)}")
    return {"rows": len(rows), "by_model": dict(counts), **file_binding(destination)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "artifacts/wan_aether_streaming_official_r1/requests",
    )
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise SystemExit(f"output already exists: {output_dir}")

    parent_sha = sha256_file(CANONICAL_PARENT_SOURCE)
    if parent_sha != CANONICAL_PARENT_SOURCE_SHA256:
        raise SystemExit(f"canonical parent source SHA changed: {parent_sha}")
    lock = json.loads(NAMESPACE_LOCK.read_text(encoding="utf-8"))
    if lock.get("canonical_universe_sha256") != CANONICAL_UNIVERSE_SHA256:
        raise SystemExit("namespace lock universe SHA changed")
    if lock.get("canonical_source", {}).get("sha256") != CANONICAL_PARENT_SOURCE_SHA256:
        raise SystemExit("namespace lock parent source SHA changed")
    inventory = json.loads(INVENTORY.read_text(encoding="utf-8"))
    if inventory.get("canonical_protocol", {}).get("protocol_sha256") != FROZEN_PROTOCOL_SEMANTIC_SHA256:
        raise SystemExit("inventory protocol SHA changed")
    if inventory.get("source", {}).get("protocol_file_sha256") != FROZEN_PROTOCOL_FILE_SHA256:
        raise SystemExit("inventory protocol-file SHA changed")

    specs = [
        spec
        for spec in inventory["specs"]
        if spec.get("vfm") in VFMS
    ]
    by_vfm = Counter(str(spec["vfm"]) for spec in specs)
    if dict(by_vfm) != EXPECTED_SPECS:
        raise SystemExit(f"unexpected wan/aether spec counts: {dict(by_vfm)}")

    canonical: dict[int, list[dict[str, Any]]] = {}
    for prefix in PREFIXES:
        path = Path(str(MANIFEST_TEMPLATE).format(prefix=prefix))
        meta_path = Path(f"{path}.meta.json")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("sha256") != sha256_file(path):
            raise SystemExit(f"manifest SHA mismatch: {path}")
        if meta.get("universe_sha256") != CANONICAL_UNIVERSE_SHA256:
            raise SystemExit(f"manifest universe SHA mismatch: {path}")
        canonical[prefix] = load_canonical(path, prefix)

    output_dir.mkdir(parents=True)
    experiments = write_experiments(output_dir / "experiments_wan_aether_220.csv")
    requests_path = output_dir / "remote_gate_requests_wan_aether.jsonl.gz"
    counts = Counter()
    seen_units: set[str] = set()
    seen_keys: set[tuple[Any, ...]] = set()
    with gzip.GzipFile(filename=str(requests_path), mode="wb", mtime=0) as sink:
        for spec in sorted(
            specs,
            key=lambda item: (
                str(item["vfm"]),
                int(item["feature_layer"]),
                int(item["timestep"]),
                int(item["prefix"]),
            ),
        ):
            prefix = int(spec["prefix"])
            for row in canonical[prefix]:
                request = build_request(spec, row)
                unit = request["unit_id"]
                if unit in seen_units:
                    raise SystemExit(f"duplicate unit_id: {unit}")
                seen_units.add(unit)
                expected = request["expected"]
                key = (
                    expected["vfm"],
                    expected["scene_id"],
                    expected["feature_layer"],
                    expected["timestep"],
                    expected["prefix"],
                    expected["window_id"],
                )
                if key in seen_keys:
                    raise SystemExit(f"duplicate official key: {key}")
                seen_keys.add(key)
                counts[f"{expected['vfm']}:L{expected['feature_layer']}:p{expected['prefix']}"] += 1
                sink.write((canonical_json(request) + "\n").encode("utf-8"))

    by_model = Counter(key[0] for key in seen_keys)
    if dict(by_model) != EXPECTED_REQUESTS:
        raise SystemExit(f"unexpected request counts: {dict(by_model)}")
    created_at = datetime.now(timezone.utc).isoformat()
    selection = {
        "schema": "canonical_v2_wan_aether_gate_request_v1",
        "status": "BUILT_NOT_GATED",
        "created_at": created_at,
        "frozen_816_request_sha256": "5e0854c11f5141780ae797da3f1368cffec270a42b9ec7ab2d4cf31908597cd5",
        "note": "This campaign is disjoint from the frozen 816 request set.",
        "feature_namespace": GAP_NAMESPACE,
        "canonical_parent_source_sha256": CANONICAL_PARENT_SOURCE_SHA256,
        "canonical_universe_sha256": CANONICAL_UNIVERSE_SHA256,
        "canonical_protocol_sha256": FROZEN_PROTOCOL_SEMANTIC_SHA256,
        "canonical_protocol_file_sha256": FROZEN_PROTOCOL_FILE_SHA256,
        "experiments": experiments,
        "inventory": file_binding(INVENTORY),
        "namespace_lock": file_binding(NAMESPACE_LOCK),
        "canonical_parent_source": file_binding(CANONICAL_PARENT_SOURCE),
        "output": {**file_binding(requests_path), "rows": len(seen_units)},
        "counts_by_model": dict(by_model),
        "counts_by_model_disk_layer_prefix": dict(sorted(counts.items())),
        "official_key": ["vfm", "scene_id", "feature_layer", "timestep", "prefix", "window_id"],
        "scheduler_writes_performed": False,
    }
    selection_path = output_dir / "selection_summary.json"
    selection_path.write_text(
        json.dumps(selection, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest = {
        "schema": "CANONICAL_V2_WAN_AETHER_STREAMING_GATE_REQUEST_MANIFEST_V1",
        "created_at": created_at,
        "canonical_parent_source_sha256": CANONICAL_PARENT_SOURCE_SHA256,
        "canonical_universe_sha256": CANONICAL_UNIVERSE_SHA256,
        "outputs": [
            {
                "file": requests_path.name,
                "sha256": sha256_file(requests_path),
                "size_bytes": requests_path.stat().st_size,
            },
            {
                "file": selection_path.name,
                "sha256": sha256_file(selection_path),
                "size_bytes": selection_path.stat().st_size,
            },
            {
                "file": "experiments_wan_aether_220.csv",
                "sha256": experiments["sha256"],
                "size_bytes": experiments["size_bytes"],
            },
        ],
        "note": "catalog_manifest.json is excluded from its own non-circular digest set",
    }
    (output_dir / "catalog_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({**selection, "catalog_manifest": file_binding(output_dir / "catalog_manifest.json")}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
