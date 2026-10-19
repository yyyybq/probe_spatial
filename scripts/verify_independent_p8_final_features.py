#!/usr/bin/env python3
"""Submit-side verifier for pooled independent-P8 exact-unit stores."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

WIDTHS = {"dino": 1024, "f3r": 1024, "vjepa2": 1024, "wan": 1536}


def sha256(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def output_path(root: Path, model: str, unit: dict[str, Any]) -> Path:
    return root / model / "scannetpp" / unit["scene_id"] / f"{unit['unit_id']}.safetensors"


def atomic_create(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(value, indent=2, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
            os.fchmod(handle.fileno(), 0o644)
        os.link(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--science-tag", required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--request-manifest", type=Path, required=True)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--model", choices=tuple(WIDTHS), required=True)
    parser.add_argument("--artifact-output", type=Path, required=True)
    parser.add_argument("--smoke-balanced", action="store_true")
    args = parser.parse_args()

    import torch
    from safetensors.torch import load_file

    request = args.request_manifest.resolve()
    rows = [json.loads(line) for line in request.read_text().splitlines() if line.strip()]
    wanted = sorted((row for row in rows if row.get("model") == args.model), key=lambda row: row["unit_id"])
    if args.smoke_balanced:
        wanted = (
            [row for row in wanted if row["feature_role"] == "isolated_frame"][:2]
            + [row for row in wanted if row["feature_role"] == "streaming_context_p8"][:2]
        )
    request_sha = sha256(request)
    problems: list[dict[str, Any]] = []
    paths: Counter[str] = Counter()
    verified = fallback = mismatch = nonfinite = constant = 0
    shape_counts: Counter[str] = Counter()
    contract_shas: Counter[str] = Counter()
    model_revisions: Counter[str] = Counter()
    tree_shas: Counter[str] = Counter()
    for unit in wanted:
        feature = output_path(args.feature_root.resolve(), args.model, unit)
        sidecar = Path(f"{feature}.manifest.json")
        paths[str(feature)] += 1
        if not feature.is_file() or not sidecar.is_file() or feature.is_symlink() or sidecar.is_symlink():
            problems.append({"unit_id": unit["unit_id"], "reason": "missing_feature_or_sidecar"})
            continue
        try:
            meta = json.loads(sidecar.read_text())
        except Exception as exc:
            problems.append({"unit_id": unit["unit_id"], "reason": "sidecar_unreadable", "detail": repr(exc)})
            continue
        expected = {
            "science_tag": args.science_tag,
            "study_protocol_sha256": args.protocol_sha256,
            "unit_id": unit["unit_id"],
            "model": args.model,
            "layer": unit["layer"],
            "timestep": unit["timestep"],
            "feature_role": unit["feature_role"],
            "raw_frame_id": unit["raw_frame_id"],
            "context_raw_ids": unit["context_raw_ids"],
            "source_manifest_sha256": unit["source_manifest_sha256"],
            "request_manifest_sha256": request_sha,
            "fallback": False,
        }
        differing = [key for key, value in expected.items() if meta.get(key) != value]
        if differing:
            mismatch += 1
            problems.append({"unit_id": unit["unit_id"], "reason": "identity_mismatch", "fields": differing})
            continue
        fallback += int(meta.get("fallback") is not False)
        if feature.stat().st_size != meta.get("size_bytes") or sha256(feature) != meta.get("feature_sha256"):
            problems.append({"unit_id": unit["unit_id"], "reason": "payload_receipt_mismatch"})
            continue
        if not meta.get("extractor_sha256") or not meta.get("model_revision") or not meta.get("snapshot_tree_sha256"):
            problems.append({"unit_id": unit["unit_id"], "reason": "missing_code_or_model_provenance"})
            continue
        try:
            tensor = load_file(str(feature))["feat"]
        except Exception as exc:
            problems.append({"unit_id": unit["unit_id"], "reason": "tensor_unreadable", "detail": repr(exc)})
            continue
        shape_counts[str(list(tensor.shape))] += 1
        contract_shas[meta["extraction_contract_sha256"]] += 1
        model_revisions[meta["model_revision"]] += 1
        tree_shas[meta["snapshot_tree_sha256"]] += 1
        if list(tensor.shape) != [WIDTHS[args.model]] or str(tensor.dtype) != "torch.float32":
            problems.append({"unit_id": unit["unit_id"], "reason": "shape_or_dtype", "shape": list(tensor.shape), "dtype": str(tensor.dtype)})
            continue
        if not torch.isfinite(tensor).all():
            nonfinite += 1
            problems.append({"unit_id": unit["unit_id"], "reason": "nonfinite"})
            continue
        if float(tensor.std().item()) <= 0 or float(tensor.abs().max().item()) <= 0:
            constant += 1
            problems.append({"unit_id": unit["unit_id"], "reason": "zero_or_constant"})
            continue
        verified += 1
    duplicates = [path for path, count in paths.items() if count != 1]
    homogeneous = len(contract_shas) == len(model_revisions) == len(tree_shas) == 1 if verified else False
    status = (
        "COMPLETE_VALID"
        if verified == len(wanted) and not problems and not duplicates and homogeneous
        else "FAILED_VALIDATION"
    )
    result = {
        "schema_version": 1,
        "science_tag": args.science_tag,
        "status": status,
        "model": args.model,
        "scope": "smoke_balanced_4" if args.smoke_balanced else "full_model_request",
        "request_manifest": str(request),
        "request_manifest_sha256": request_sha,
        "study_protocol_sha256": args.protocol_sha256,
        "feature_root": str(args.feature_root.resolve()),
        "requested": len(wanted),
        "verified": verified,
        "missing_or_invalid": len(wanted) - verified,
        "duplicates": len(duplicates),
        "duplicate_feature_paths": duplicates[:200],
        "fallback": fallback,
        "raw_id_or_contract_mismatch": mismatch,
        "nonfinite": nonfinite,
        "zero_or_constant": constant,
        "shape_counts": dict(shape_counts),
        "extraction_contract_sha256": dict(contract_shas),
        "model_revision": dict(model_revisions),
        "snapshot_tree_sha256": dict(tree_shas),
        "problems": problems[:200],
        "coverage_contract": "requested == verified; missing == duplicate == fallback == raw-ID mismatch == nonfinite == zero/constant == 0",
    }
    atomic_create(args.artifact_output.resolve(), result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if status == "COMPLETE_VALID" else 2


if __name__ == "__main__":
    raise SystemExit(main())
