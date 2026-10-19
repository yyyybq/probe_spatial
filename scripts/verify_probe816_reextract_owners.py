#!/usr/bin/env python3
"""Verify exact locally re-extracted streaming owners and freeze receipts."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import struct
from pathlib import Path
from typing import Any, Iterable

import numpy as np


CANONICAL_SOURCE_SHA = "ebc496eb0587cda372ccf29a7c5244ef454096151f73c8eed9511967ded1902d"
GAP_NAMESPACE = "FEAT_STREAMING_CANONICAL_V2_GAP_DELTA_V1"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def jsonl(path: Path) -> Iterable[dict[str, Any]]:
    opener = gzip.open if path.name.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def tensor_header(path: Path) -> tuple[list[int], str, str]:
    with path.open("rb") as f:
        raw = f.read(8)
        if len(raw) != 8:
            raise ValueError(f"truncated safetensors header: {path}")
        size = struct.unpack("<Q", raw)[0]
        raw_header = f.read(size)
        header = json.loads(raw_header)
    if set(k for k in header if k != "__metadata__") != {"feat"}:
        raise ValueError(f"unexpected tensor keys: {path}")
    info = header["feat"]
    return [int(v) for v in info["shape"]], str(info["dtype"]), hashlib.sha256(raw_header).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--requests", type=Path, required=True)
    ap.add_argument("--holes", type=Path, required=True)
    ap.add_argument("--local-root", type=Path, required=True)
    ap.add_argument("--vfm", action="append", required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        raise ValueError(f"output exists: {args.output}")
    wanted = set(args.vfm)
    requests = {str(row["unit_id"]): row for row in jsonl(args.requests)}
    holes = [row for row in jsonl(args.holes) if row["key"][1] in wanted]
    if not holes:
        raise ValueError("no selected holes")

    receipts = []
    for hole in holes:
        key = hole["key"]
        _, vfm, source, scene, layer, timestep, window_id, prefix = key
        unit = str(hole["unit_id"])
        request = requests.get(unit)
        if request is None:
            raise ValueError(f"missing request for {unit}")
        expected = request["expected"]
        local_dir = args.local_root / vfm / source / scene
        feature = local_dir / f"window_{int(window_id):04d}" / f"prefix_{int(prefix)-1:06d}" / expected["feature_filename"]
        sidecar_path = Path(str(feature) + ".manifest.json")
        index_path = local_dir / "prefix_index.npy"
        index_sidecar_path = Path(str(index_path) + ".manifest.json")
        for path in (feature, sidecar_path, index_path, index_sidecar_path):
            if path.is_symlink() or not path.is_file():
                raise ValueError(f"missing exact local owner for {unit}: {path}")
        sidecar = json.loads(sidecar_path.read_text())
        index_sidecar = json.loads(index_sidecar_path.read_text())
        if sidecar.get("canonical_source_sha256") != CANONICAL_SOURCE_SHA:
            raise ValueError(f"canonical source mismatch for {unit}")
        if sidecar.get("feature_namespace") != GAP_NAMESPACE:
            raise ValueError(f"namespace mismatch for {unit}")
        if sidecar.get("gap_record_key") != {"scene": scene, "window_id": int(window_id), "prefix": int(prefix)}:
            raise ValueError(f"gap key mismatch for {unit}")
        if str(sidecar.get("vfm")) != vfm or int(sidecar.get("layer")) != int(layer):
            raise ValueError(f"model/layer mismatch for {unit}")
        if int(sidecar.get("timestep")) != int(timestep):
            raise ValueError(f"timestep mismatch for {unit}")
        shape, dtype, header_sha = tensor_header(feature)
        if shape != expected["expected_shape"] or dtype != "F16":
            raise ValueError(f"tensor contract mismatch for {unit}: {shape}/{dtype}")
        if sidecar.get("tensors", {}).get("feat", {}).get("shape") != shape:
            raise ValueError(f"sidecar shape mismatch for {unit}")
        feature_sha = sha256_file(feature)
        if feature_sha != sidecar.get("sha256") or feature.stat().st_size != int(sidecar.get("size_bytes", -1)):
            raise ValueError(f"feature SHA/size mismatch for {unit}")
        index_sha = sha256_file(index_path)
        if index_sha != index_sidecar.get("sha256"):
            raise ValueError(f"index SHA mismatch for {unit}")
        records = np.load(index_path, allow_pickle=True).tolist()
        matches = [r for r in records if int(r["window_id"]) == int(window_id) and int(r["tail"]) == int(prefix) - 1]
        if len(matches) != 1:
            raise ValueError(f"index record cardinality mismatch for {unit}: {len(matches)}")
        record = matches[0]
        if [int(v) for v in record["indices"]] != [int(v) for v in expected["raw_ids"]]:
            raise ValueError(f"raw ID mismatch for {unit}")
        uri_base = f"aoss://baiqiao/InsScene-15K/{GAP_NAMESPACE}/{vfm}/{source}/{scene}"
        rel = feature.relative_to(local_dir).as_posix()
        receipt = {
            "schema": "CANONICAL_V2_PROBE816_REEXTRACT_OWNER_RECEIPT_V1",
            "status": "PASS_LOCAL_FULL_SHA_VERIFIED",
            "unit_id": unit,
            "key": key,
            "source_uri": f"{uri_base}/{rel}",
            "sidecar_uri": f"{uri_base}/{sidecar_path.relative_to(local_dir).as_posix()}",
            "source_index_uri": f"{uri_base}/prefix_index.npy",
            "payload_sha256": feature_sha,
            "payload_size_bytes": feature.stat().st_size,
            "safetensors_header_sha256": header_sha,
            "source_index_sha256": index_sha,
            "source_index_size_bytes": index_path.stat().st_size,
            "tensor_shape": shape,
            "tensor_dtype": "float16",
            "raw_ids": record["indices"],
            "input_length": int(record["input_length"]),
            "valid_length": int(record["valid_length"]),
            "pad_mode": str(record["pad_mode"]),
            "window_id": int(window_id),
            "prefix": int(prefix),
            "local_payload_full_sha_verified": True,
        }
        receipts.append(receipt)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as raw, gzip.GzipFile(
        filename="", mode="wb", fileobj=raw, mtime=0, compresslevel=6
    ) as gz:
        for row in sorted(receipts, key=lambda x: tuple(x["key"])):
            gz.write((json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n").encode())
    summary = {
        "schema": "CANONICAL_V2_PROBE816_REEXTRACT_OWNER_SET_V1",
        "status": "READY",
        "vfms": sorted(wanted),
        "rows": len(receipts),
        "receipts": {"path": str(args.output.resolve()), "sha256": sha256_file(args.output)},
        "requests_sha256": sha256_file(args.requests),
        "holes_sha256": sha256_file(args.holes),
        "local_root": str(args.local_root.resolve()),
        "payloads_full_sha_verified": len(receipts),
    }
    summary_path = args.output.with_suffix("").with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
