#!/usr/bin/env python3
"""Build a temporal-unit-aware raw-ID exact request manifest for one study.

``feature_request_manifest.jsonl`` (v1) is intentionally left immutable.  It
was an allow-list of individual raw frames, but it did not preserve the ordered
p8 context as one extraction unit.  This tool validates that limitation and
creates v2 from the already frozen raw-pose manifests; it never edits v1.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
MODELS = (
    {"model": "dino", "layer": 0, "timestep": None},
    {"model": "f3r", "layer": 24, "timestep": None},
    {"model": "vjepa2", "layer": 23, "timestep": None},
    {"model": "wan", "layer": 20, "timestep": 749},
)


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def atomic_text_create(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    except FileExistsError:
        raise
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def atomic_json_create(path: Path, value: Any) -> None:
    atomic_text_create(path, json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n")


def source_rows(manifest_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, str]]:
    names = {
        "actionability_train": "actionability_p8_independent_v1_train.jsonl",
        "actionability_val": "actionability_p8_independent_v1_val.jsonl",
        "algebra_train": "algebra_p8_independent_v1_train.jsonl",
        "algebra_val": "algebra_p8_independent_v1_val.jsonl",
        "algebra_real_loops": "algebra_p8_independent_v1_real_loops.jsonl",
    }
    manifest_hashes = {key: sha256(manifest_dir / name) for key, name in names.items()}
    regular = []
    loops = []
    for key, name in names.items():
        rows = read_jsonl(manifest_dir / name)
        (loops if key == "algebra_real_loops" else regular).extend(rows)
    return regular, loops, manifest_hashes


def add_isolated(
    units: dict[tuple[Any, ...], dict[str, Any]],
    model: dict[str, Any], row: dict[str, Any], raw: str, role: str, hashes: dict[str, str]
) -> None:
    key = (model["model"], model["layer"], model["timestep"], "isolated_frame", raw)
    unit = units.setdefault(
        key,
        {
            "unit_id": "fu_" + hashlib.sha256(canonical(key).encode()).hexdigest()[:24],
            **model,
            "feature_role": "isolated_frame",
            "requested_roles": [],
            "scene_id": row["scene_id"],
            "raw_frame_id": raw,
            "context_raw_ids": None,
            "source_sample_ids": [],
            "source_manifest_sha256": hashes,
            "exact_raw_id_required": True,
            "fallback_forbidden": True,
        },
    )
    unit["requested_roles"].append(role)
    unit["source_sample_ids"].append(row["sample_id"])


def add_context(
    units: dict[tuple[Any, ...], dict[str, Any]], model: dict[str, Any], row: dict[str, Any], hashes: dict[str, str]
) -> None:
    context = tuple(row["context_raw_ids"])
    if len(context) != 8 or row["anchor_raw_id"] != context[-1]:
        raise ValueError(f"{row['sample_id']}: context is not an ordered p8 ending at anchor")
    key = (model["model"], model["layer"], model["timestep"], "streaming_context_p8", context)
    unit = units.setdefault(
        key,
        {
            "unit_id": "fu_" + hashlib.sha256(canonical(key).encode()).hexdigest()[:24],
            **model,
            "feature_role": "streaming_context_p8",
            "requested_roles": ["streaming_context_p8"],
            "scene_id": row["scene_id"],
            "raw_frame_id": row["anchor_raw_id"],
            "context_raw_ids": list(context),
            "source_sample_ids": [],
            "source_manifest_sha256": hashes,
            "exact_raw_id_required": True,
            "fallback_forbidden": True,
        },
    )
    unit["source_sample_ids"].append(row["sample_id"])


def build_units(rows: Iterable[dict[str, Any]], loop_rows: Iterable[dict[str, Any]], hashes: dict[str, str]) -> list[dict[str, Any]]:
    units: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in rows:
        for model in MODELS:
            add_context(units, model, row, hashes)
            add_isolated(units, model, row, row["anchor_raw_id"], "isolated_anchor", hashes)
            if "goal_raw_id" in row:
                add_isolated(units, model, row, row["goal_raw_id"], "isolated_goal", hashes)
                add_isolated(units, model, row, row["candidate_endpoint_raw_id_1"], "isolated_candidate_endpoint_1", hashes)
                add_isolated(units, model, row, row["candidate_endpoint_raw_id_2"], "isolated_candidate_endpoint_2", hashes)
            else:
                for horizon, role in (("h1", "isolated_h1"), ("h2", "isolated_h2"), ("h4", "isolated_h4")):
                    add_isolated(units, model, row, row["horizon_target_raw_ids"][horizon], role, hashes)
    for row in loop_rows:
        for model in MODELS:
            add_context(units, model, row, hashes)
            add_isolated(units, model, row, row["anchor_raw_id"], "isolated_anchor", hashes)
            add_isolated(units, model, row, row["final_raw_id"], "isolated_loop_endpoint", hashes)
    for unit in units.values():
        unit["requested_roles"] = sorted(set(unit["requested_roles"]))
        unit["source_sample_ids"] = sorted(set(unit["source_sample_ids"]))
    return sorted(units.values(), key=lambda row: row["unit_id"])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--science-tag", required=True)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    args = parser.parse_args()
    if not args.science_tag.startswith("independent_actionability_algebra_p8_"):
        raise SystemExit("unexpected science tag")
    manifest_dir = args.manifest_dir.resolve()
    summary = json.loads((manifest_dir / "manifest_summary.json").read_text())
    if summary.get("science_tag") != args.science_tag or summary.get("status") != "COMPLETE_VALID":
        raise SystemExit("raw-pose manifest is not this valid independent study")
    regular, loops, hashes = source_rows(manifest_dir)
    expected_hashes = summary["manifest_sha256"]
    mismatches = {key: {"expected": expected_hashes[key], "actual": value} for key, value in hashes.items() if expected_hashes.get(key) != value}
    if mismatches:
        raise SystemExit(f"frozen raw-pose manifest SHA mismatch: {mismatches}")
    v1 = manifest_dir / "feature_request_manifest.jsonl"
    v1_rows = read_jsonl(v1)
    v1_violations = {
        "missing_context_raw_ids": len(v1_rows),
        "streaming_rows_are_frame_allowlist_not_context_units": sum(row.get("feature_role") == "streaming_context_p8" for row in v1_rows),
    }
    units = build_units(regular, loops, hashes)
    output = manifest_dir / "feature_request_manifest_v2.jsonl"
    validation = args.artifact_dir.resolve() / "feature_lineage" / "feature_request_v1_validation.json"
    report = args.report_dir.resolve() / "FEATURE_REQUEST_V2.md"
    if output.exists() or validation.exists() or report.exists():
        raise SystemExit("v2 request outputs already exist; refusing overwrite")
    atomic_text_create(output, "".join(canonical(row) + "\n" for row in units))
    output_sha = sha256(output)
    per_model = Counter(row["model"] for row in units)
    by_kind = Counter(row["feature_role"] for row in units)
    atomic_json_create(
        validation,
        {
            "schema_version": 1,
            "science_tag": args.science_tag,
            "status": "FAILED_VALIDATION",
            "v1_path": str(v1),
            "v1_sha256": sha256(v1),
            "v1_rows": len(v1_rows),
            "reason": "v1 did not preserve ordered p8 context_raw_ids as an extraction-unit identity",
            "violations": v1_violations,
            "remediation": "v2 regenerated from frozen raw-pose manifests; v1 remains unmodified",
            "v2_path": str(output),
            "v2_sha256": output_sha,
            "v2_units": len(units),
            "v2_per_model": dict(sorted(per_model.items())),
            "v2_by_feature_role": dict(sorted(by_kind.items())),
            "source_manifest_sha256": hashes,
        },
    )
    atomic_text_create(
        report,
        f"""# Feature request v2\n\nStatus: `COMPLETE_WITH_CAVEAT`\n\nThe original frame-level request manifest is preserved but is `FAILED_VALIDATION` for temporal extraction because it lacks ordered p8 context-unit identity.  It has **not** been edited.  `feature_request_manifest_v2.jsonl` is regenerated directly from the frozen raw-pose manifests and is the only valid input to exact-feature audit/extraction.\n\n- v1 rows: {len(v1_rows)}\n- v2 units: {len(units)}\n- v2 SHA256: `{output_sha}`\n- Per model: `{dict(sorted(per_model.items()))}`\n- Unit kinds: `{dict(sorted(by_kind.items()))}`\n- Source raw-pose SHA map: `artifacts/{args.science_tag}/manifests/manifest_summary.json`\n""",
    )
    print(json.dumps({"status": "COMPLETE_WITH_CAVEAT", "v2_units": len(units), "v2_sha256": output_sha, "per_model": per_model}, default=dict))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
