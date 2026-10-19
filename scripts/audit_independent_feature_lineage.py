#!/usr/bin/env python3
"""Fail-closed actual-raw-ID audit for an independent p8 feature request."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_text_create(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def atomic_json_create(path: Path, value: Any) -> None:
    atomic_text_create(path, json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def unit_key(row: dict[str, Any]) -> tuple[Any, ...]:
    identity: str | tuple[str, ...]
    if row["feature_role"] == "streaming_context_p8":
        identity = tuple(row["context_raw_ids"])
    elif row["feature_role"] == "isolated_frame":
        identity = str(row["raw_frame_id"])
    else:
        raise ValueError(f"unsupported requested role: {row['feature_role']}")
    return (row["model"], int(row["layer"]), row.get("timestep"), row["feature_role"], identity)


def sidecar_key(meta: dict[str, Any]) -> tuple[Any, ...] | None:
    model = meta.get("model")
    layer = meta.get("layer")
    role = meta.get("feature_role")
    if not isinstance(model, str) or not isinstance(role, str) or not isinstance(layer, int):
        return None
    if role == "streaming_context_p8":
        context = meta.get("context_raw_ids")
        anchor = meta.get("raw_frame_id")
        if not isinstance(context, list) or len(context) != 8 or not all(isinstance(value, str) for value in context):
            return None
        if anchor != context[-1]:
            return None
        identity: str | tuple[str, ...] = tuple(context)
    elif role == "isolated_frame":
        raw = meta.get("raw_frame_id")
        if not isinstance(raw, str) or not raw.startswith("scannetpp:"):
            return None
        identity = raw
    else:
        return None
    return (model, layer, meta.get("timestep"), role, identity)


def valid_sidecar(path: Path, meta: dict[str, Any]) -> tuple[bool, str, dict[str, Any]]:
    feature = Path(str(path)[: -len(".manifest.json")])
    required = ("extractor_sha256", "source_manifest_sha256", "feature_sha256", "size_bytes", "dtype")
    missing = [key for key in required if meta.get(key) in (None, "", {}, [])]
    if missing:
        return False, "missing_provenance:" + ",".join(missing), {}
    if feature.is_symlink() or not feature.is_file():
        return False, "feature_missing_or_symlink", {}
    if feature.stat().st_size != meta["size_bytes"]:
        return False, "size_mismatch", {}
    digest = sha256(feature)
    if digest != meta["feature_sha256"]:
        return False, "feature_sha256_mismatch", {}
    return True, "ok", {
        "feature_path": str(feature.resolve()),
        "feature_sha256": digest,
        "feature_size_bytes": feature.stat().st_size,
        "sidecar_path": str(path.resolve()),
        "extractor_sha256": meta["extractor_sha256"],
        "source_manifest_sha256": meta["source_manifest_sha256"],
        "dtype": meta["dtype"],
        "shape": meta.get("shape") or meta.get("tensors", {}).get("feat", {}).get("shape"),
    }


def iter_sidecars(roots: Iterable[Path]) -> Iterable[Path]:
    for root in roots:
        if not root.is_dir():
            continue
        try:
            yield from root.rglob("*.manifest.json")
        except OSError:
            continue


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--science-tag", required=True)
    parser.add_argument("--request-manifest", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, action="append", required=True)
    args = parser.parse_args()
    if not args.science_tag.startswith("independent_actionability_algebra_p8_"):
        raise SystemExit("unexpected science tag")
    request = args.request_manifest.resolve()
    rows = read_jsonl(request)
    if not rows:
        raise SystemExit("empty request manifest")
    wanted = {unit_key(row): row for row in rows}
    if len(wanted) != len(rows):
        raise SystemExit("request manifest has duplicate feature-unit identities")
    reusable: dict[tuple[Any, ...], dict[str, Any]] = {}
    reject = Counter()
    scanned = 0
    roots = [path.resolve() for path in args.candidate_root]
    for sidecar in iter_sidecars(roots):
        scanned += 1
        try:
            meta = json.loads(sidecar.read_text())
        except (OSError, json.JSONDecodeError):
            reject["unreadable_sidecar"] += 1
            continue
        if not isinstance(meta, dict):
            reject["nonobject_sidecar"] += 1
            continue
        key = sidecar_key(meta)
        if key is None or key not in wanted:
            reject["not_an_exact_requested_unit"] += 1
            continue
        ok, reason, evidence = valid_sidecar(sidecar, meta)
        if not ok:
            reject[reason] += 1
            continue
        if key in reusable:
            reject["duplicate_exact_candidate"] += 1
            continue
        reusable[key] = evidence
    reusable_rows = []
    missing_rows = []
    for key, row in sorted(wanted.items(), key=lambda item: item[1]["unit_id"]):
        if key in reusable:
            reusable_rows.append({**row, "reuse_evidence": reusable[key]})
        else:
            missing_rows.append(row)
    artifact = args.artifact_dir.resolve() / "feature_lineage"
    reusable_path = artifact / "reusable_exact_feature_units_v2.jsonl"
    missing_path = artifact / "missing_exact_feature_units_v2.jsonl"
    summary_path = artifact / "exact_feature_lineage_audit_v1.json"
    report_path = args.report_dir.resolve() / "FEATURE_LINEAGE_AUDIT_V2.md"
    if any(path.exists() for path in (reusable_path, missing_path, summary_path, report_path)):
        raise SystemExit("exact lineage audit output already exists; refusing overwrite")
    atomic_text_create(reusable_path, "".join(canonical(row) + "\n" for row in reusable_rows))
    atomic_text_create(missing_path, "".join(canonical(row) + "\n" for row in missing_rows))
    by_model = {
        model: {
            "requested": sum(row["model"] == model for row in rows),
            "reusable": sum(row["model"] == model for row in reusable_rows),
            "missing": sum(row["model"] == model for row in missing_rows),
        }
        for model in sorted({row["model"] for row in rows})
    }
    status = "COMPLETE_VALID" if not missing_rows else "COMPLETE_WITH_CAVEAT"
    summary = {
        "schema_version": 1,
        "science_tag": args.science_tag,
        "status": status,
        "request_manifest": str(request),
        "request_manifest_sha256": sha256(request),
        "candidate_roots": [str(path) for path in roots],
        "sidecars_scanned": scanned,
        "requested_units": len(rows),
        "reusable_units": len(reusable_rows),
        "missing_units": len(missing_rows),
        "per_model": by_model,
        "rejected_sidecars": dict(sorted(reject.items())),
        "reuse_policy": "actual raw-ID/context-ID sidecar plus verified feature SHA only",
        "fallback_policy": "forbidden",
        "reusable_manifest_sha256": sha256(reusable_path),
        "missing_manifest_sha256": sha256(missing_path),
    }
    atomic_json_create(summary_path, summary)
    atomic_text_create(
        report_path,
        f"""# Exact feature lineage audit v2\n\nStatus: `{status}`\n\n- Request units: {len(rows)}\n- Accepted exact reusable units: {len(reusable_rows)}\n- Missing units for independent extraction: {len(missing_rows)}\n- Scanned sidecars: {scanned}\n- Candidate roots: {', '.join(f'`{path}`' for path in roots)}\n- Per model: `{by_model}`\n\nOnly sidecars containing the exact independent raw ID (or the ordered p8 context), model/layer/timestep, extractor and source-manifest provenance, size, dtype, and a matching SHA256 were accepted.  Filename/window/tail similarity and unproved cache entries were rejected.\n""",
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
