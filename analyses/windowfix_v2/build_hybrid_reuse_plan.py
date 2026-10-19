#!/usr/bin/env python3
"""Build a fail-closed canonical-v2 mixed-cache import/reextract plan.

This tool consumes the output of ``classify_feature_reuse.py`` run with
``--payload-hash-policy legacy-import``.  It writes only small JSON/JSONL plan
files.  It never copies feature payloads, writes a feature namespace, calls
SCO, or starts an extractor.

The mixed namespace is deliberately distinct from ``*_SEEDED_V1``.  Every
accepted old payload remains labelled ``LEGACY_EXACT_IMPORT_V1`` and carries
its original path plus a newly computed SHA256.  Every non-exact or incomplete
record is routed to re-extraction.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Iterator, Mapping, Sequence
from urllib.parse import urlparse, urlunparse


SCHEMA_VERSION = 1
IMPORT_CONTRACT = "LEGACY_EXACT_IMPORT_V1"
EXPECTED_NAMESPACES = {
    "streaming": "FEAT_STREAMING_CANONICAL_V2_MIXED_V1",
    "target": "FEAT_TARGET_CANONICAL_V2_MIXED_V1",
}
CLASSIFIER_STATUSES = (
    "REUSE_EXACT",
    "REEXTRACT_MISMATCH",
    "REEXTRACT_MISSING",
    "UNVERIFIED",
)
HEX = frozenset("0123456789abcdef")


class PlanError(ValueError):
    """Raised when a classifier receipt cannot safely drive an import."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise PlanError(f"required input is not a regular file: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise PlanError(f"expected JSON object: {path}")
    return value


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    if path.is_symlink() or not path.is_file():
        raise PlanError(f"required input is not a regular file: {path}")
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise PlanError(f"{path}:{line_number}: invalid JSON") from exc
            if not isinstance(value, dict):
                raise PlanError(f"{path}:{line_number}: expected object")
            yield value


def _write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.tmp-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o644)
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _jsonl(rows: Iterable[Mapping[str, Any]]) -> str:
    return "".join(
        json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n"
        for row in rows
    )


def _is_hex_sha(value: Any) -> bool:
    text = str(value or "").lower()
    return len(text) == 64 and set(text) <= HEX


def _safe_component(value: Any, *, field: str) -> str:
    text = str(value or "").strip()
    if not text or text in {".", ".."} or "/" in text or "\\" in text:
        raise PlanError(f"invalid {field}: {text!r}")
    return text


def _root_namespace(root: str) -> str:
    parsed = urlparse(root)
    path = parsed.path if parsed.scheme else root
    return PurePosixPath(path.rstrip("/")).name


def _join_root(root: str, *components: str) -> str:
    parsed = urlparse(root)
    if parsed.scheme:
        path = str(PurePosixPath(parsed.path, *components))
        return urlunparse(parsed._replace(path=path))
    return str(Path(root, *components).absolute())


def _identity_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    unit_kind = str(row.get("unit_kind", "streaming_file"))
    base = (
        unit_kind,
        row.get("spec_id"),
        row.get("scene_id"),
        row.get("window_id"),
        row.get("prefix"),
        row.get("layer"),
        row.get("timestep"),
    )
    if unit_kind == "target_row":
        return base + (row.get("target_raw_id"),)
    return base


def _preprocess_identity(expected: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in sorted(expected.items())
        if key == "resize"
        or key == "frame_index_schema"
        or key == "preprocess"
        or key.startswith("preprocess.")
        or key.startswith("preprocess_")
    }


def _destination(row: Mapping[str, Any], roots: Mapping[str, str]) -> tuple[str, str]:
    unit_kind = str(row.get("unit_kind", "streaming_file"))
    mode = "target" if unit_kind == "target_row" else "streaming"
    vfm = _safe_component(row.get("vfm"), field="vfm")
    source = _safe_component(row.get("source"), field="source")
    scene = _safe_component(row.get("scene_id"), field="scene_id")
    layer = int(row["layer"])
    timestep = row.get("timestep")
    feature_name = (
        f"feature_layer{layer}.sft"
        if timestep is None
        else f"feature_t{int(timestep)}_layer{layer}.sft"
    )
    if mode == "target":
        relative = (vfm, source, scene, feature_name)
    else:
        relative = (
            vfm,
            source,
            scene,
            f"window_{int(row['window_id']):04d}",
            f"prefix_{int(row['prefix']) - 1:06d}",
            feature_name,
        )
    return mode, _join_root(roots[mode], *relative)


def _validate_exact(row: Mapping[str, Any]) -> None:
    required = (
        "spec_id",
        "vfm",
        "model_id",
        "layer",
        "scene_id",
        "source",
        "feature_path",
        "source_payload_sha256",
        "source_size_bytes",
        "raw_ids",
    )
    missing = [field for field in required if row.get(field) in (None, "")]
    if missing:
        raise PlanError(f"exact classifier row lacks fields: {missing}")
    if not _is_hex_sha(row["source_payload_sha256"]):
        raise PlanError("exact classifier row lacks a computed source SHA256")
    if int(row["source_size_bytes"]) <= 0:
        raise PlanError("exact classifier row has invalid source_size_bytes")
    source_path = Path(str(row["feature_path"]))
    if not source_path.is_absolute() or source_path.is_symlink() or not source_path.is_file():
        raise PlanError(f"exact source is not a local regular file: {source_path}")
    expected = row.get("expected_sidecar_identity")
    if not isinstance(expected, dict) or "prompt" not in expected:
        raise PlanError("exact row lacks expected prompt identity")
    if not _preprocess_identity(expected):
        raise PlanError("exact row lacks expected preprocessing identity")
    raw_ids = row.get("raw_ids")
    if not isinstance(raw_ids, list) or not raw_ids:
        raise PlanError("exact row has no raw_ids")
    if str(row.get("unit_kind", "streaming_file")) == "target_row":
        if row.get("source_row_index") is None or row.get("target_raw_id") is None:
            raise PlanError("target-row import lacks source_row_index/target_raw_id")
        if int(row["target_raw_id"]) not in {int(value) for value in raw_ids}:
            raise PlanError("target_raw_id is not bound by raw_ids")


def build(
    *,
    classification_dir: Path,
    protocol_path: Path,
    policy_path: Path,
    destination_roots: Mapping[str, str],
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    summary_path = classification_dir / "summary.json"
    summary = read_json(summary_path)
    protocol = read_json(protocol_path)
    policy = read_json(policy_path)
    if summary.get("payload_hash_policy") != "legacy-import":
        raise PlanError("classifier must run with --payload-hash-policy legacy-import")
    if summary.get("generation_contract") != IMPORT_CONTRACT:
        raise PlanError("classifier generation contract is not LEGACY_EXACT_IMPORT_V1")
    if summary.get("historical_sidecar_sha256_required") is not False:
        raise PlanError("classifier receipt has an inconsistent historical-SHA policy")
    canonical_protocol = str(protocol.get("canonical_protocol", ""))
    if not canonical_protocol or summary.get("canonical_protocol") != canonical_protocol:
        raise PlanError("classifier and temporal protocol do not match")
    if policy.get("contract") != IMPORT_CONTRACT:
        raise PlanError("invalid import policy contract")
    if policy.get("canonical_temporal_protocol") != canonical_protocol:
        raise PlanError("import policy and temporal protocol do not match")
    if policy.get("destination_namespaces") != EXPECTED_NAMESPACES:
        raise PlanError("unexpected mixed namespace names in import policy")
    for mode, namespace in EXPECTED_NAMESPACES.items():
        if _root_namespace(destination_roots[mode]) != namespace:
            raise PlanError(
                f"{mode} destination must end in {namespace}, found {destination_roots[mode]}"
            )
        if "SEEDED_V1" in destination_roots[mode]:
            raise PlanError("mixed import must never target a SEEDED_V1 namespace")

    actions: dict[str, list[dict[str, Any]]] = {
        "import_files": [],
        "import_target_rows": [],
        "reextract_records": [],
        "unverified_records": [],
    }
    seen: set[tuple[Any, ...]] = set()
    destinations: set[tuple[str, str, int | None]] = set()
    classifier_files: dict[str, dict[str, Any]] = {}
    observed_statuses: Counter[str] = Counter()
    for status in CLASSIFIER_STATUSES:
        path = classification_dir / f"{status.lower()}.jsonl"
        classifier_files[status] = {"path": str(path), "sha256": sha256_file(path)}
        for row in read_jsonl(path):
            if row.get("status") != status:
                raise PlanError(f"{path}: row status does not match its worklist")
            key = _identity_key(row)
            if key in seen:
                raise PlanError(f"duplicate canonical classifier identity: {key}")
            seen.add(key)
            observed_statuses[status] += 1
            mode, destination = _destination(row, destination_roots)
            base = {
                "schema_version": SCHEMA_VERSION,
                "contract": IMPORT_CONTRACT,
                "canonical_protocol": canonical_protocol,
                "spec_id": row.get("spec_id"),
                "unit_kind": row.get("unit_kind", "streaming_file"),
                "vfm": row.get("vfm"),
                "model_id": row.get("model_id"),
                "layer": row.get("layer"),
                "timestep": row.get("timestep"),
                "tensor_key": row.get("tensor_key", "feat"),
                "source": row.get("source"),
                "scene_id": row.get("scene_id"),
                "window_id": row.get("window_id"),
                "prefix": row.get("prefix"),
                "raw_ids": row.get("raw_ids"),
                "destination_mode": mode,
                "destination_path": destination,
                "classifier_status": status,
                "classifier_reason_codes": row.get("reason_codes", []),
            }
            if status == "REUSE_EXACT":
                _validate_exact(row)
                expected = row["expected_sidecar_identity"]
                base.update(
                    {
                        "action": (
                            "IMPORT_TARGET_ROW"
                            if row.get("unit_kind") == "target_row"
                            else "COPY_WHOLE_FILE"
                        ),
                        "source_path": row["feature_path"],
                        "source_payload_sha256": row["source_payload_sha256"],
                        "source_size_bytes": int(row["source_size_bytes"]),
                        "legacy_root_id": row.get("legacy_root_id"),
                        "prompt": expected["prompt"],
                        "preprocess_identity": _preprocess_identity(expected),
                    }
                )
                if row.get("unit_kind") == "target_row":
                    base.update(
                        {
                            "source_row_index": int(row["source_row_index"]),
                            "target_raw_id": int(row["target_raw_id"]),
                        }
                    )
                    destination_key = (destination, "target_row", int(row["target_raw_id"]))
                    actions["import_target_rows"].append(base)
                else:
                    destination_key = (destination, "file", None)
                    actions["import_files"].append(base)
                if destination_key in destinations:
                    raise PlanError(f"multiple sources target the same mixed-cache unit: {destination_key}")
                destinations.add(destination_key)
            elif status == "UNVERIFIED":
                base["action"] = "REEXTRACT_UNVERIFIED"
                actions["unverified_records"].append(base)
            else:
                base["action"] = "REEXTRACT"
                actions["reextract_records"].append(base)

    expected_counts = summary.get("status_counts")
    if not isinstance(expected_counts, dict):
        raise PlanError("classifier summary has no status_counts")
    normalized_counts = {status: observed_statuses[status] for status in CLASSIFIER_STATUSES}
    if normalized_counts != {status: int(expected_counts.get(status, 0)) for status in CLASSIFIER_STATUSES}:
        raise PlanError("classifier summary counts disagree with worklists")

    plan: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "plan_type": "canonical_v2_hybrid_feature_reuse",
        "contract": IMPORT_CONTRACT,
        "status": "DRY_RUN_PLAN_ONLY",
        "copy_performed": False,
        "sco_submission_performed": False,
        "probing_training_in_scope": False,
        "canonical_protocol": canonical_protocol,
        "canonical_protocol_file": str(protocol_path),
        "canonical_protocol_file_sha256": sha256_file(protocol_path),
        "import_policy_file": str(policy_path),
        "import_policy_file_sha256": sha256_file(policy_path),
        "classification_summary": {
            "path": str(summary_path),
            "sha256": sha256_file(summary_path),
        },
        "classification_worklists": classifier_files,
        "destination_roots": dict(destination_roots),
        "destination_namespaces": EXPECTED_NAMESPACES,
        "seeded_namespace_written": False,
        "counts": {
            "canonical_records": sum(normalized_counts.values()),
            "copy_whole_file": len(actions["import_files"]),
            "import_target_row": len(actions["import_target_rows"]),
            "reextract": len(actions["reextract_records"])
            + len(actions["unverified_records"]),
            "reextract_due_to_unverified": len(actions["unverified_records"]),
            "classifier_statuses": normalized_counts,
        },
    }
    for rows in actions.values():
        rows.sort(
            key=lambda row: json.dumps(
                _identity_key(row), separators=(",", ":"), ensure_ascii=False
            )
        )
    plan["action_worklists"] = {
        f"{name}.jsonl": {
            "records": len(rows),
            "content_sha256": hashlib.sha256(_jsonl(rows).encode()).hexdigest(),
        }
        for name, rows in sorted(actions.items())
    }
    plan["content_sha256"] = canonical_sha(
        {key: value for key, value in plan.items() if key != "content_sha256"}
    )
    return plan, actions


def write_outputs(
    output_dir: Path,
    plan: Mapping[str, Any],
    actions: Mapping[str, Sequence[Mapping[str, Any]]],
) -> None:
    _write_atomic(
        output_dir / "hybrid_plan.json",
        json.dumps(plan, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
    )
    for name, rows in actions.items():
        _write_atomic(output_dir / f"{name}.jsonl", _jsonl(rows))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--classification-dir", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument(
        "--policy",
        type=Path,
        default=Path("protocols/inscene_feature_import_legacy_exact_v1.json"),
    )
    parser.add_argument("--destination-streaming-root", required=True)
    parser.add_argument("--destination-target-root", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    plan, actions = build(
        classification_dir=args.classification_dir.resolve(),
        protocol_path=args.protocol.resolve(),
        policy_path=args.policy.resolve(),
        destination_roots={
            "streaming": args.destination_streaming_root,
            "target": args.destination_target_root,
        },
    )
    write_outputs(args.output_dir.resolve(), plan, actions)
    print(json.dumps(plan["counts"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
