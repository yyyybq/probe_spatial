#!/usr/bin/env python3
"""Audit temporal-window semantics for the frozen 1096 probing settings.

This audit is deliberately fail-closed.  A setting is exact-safe only when an
actual Dataset/DataLoader canary has proven exact raw-frame identity.  Equal
window lengths, matching path names, or provenance-only checks are not treated
as exact proof.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
TASK_STATUS = ROOT / "experiments/task_status_1096_before_expand_20260822.csv"
Q00 = ROOT / "results/analysis/q00_1096/universe_1096_reconciled.csv"
Q01 = ROOT / "results/analysis/q01_feature_semantics/universe_1096_feature_semantics.csv"
ATOMIC = ROOT / "results/analysis/q01_feature_semantics/atomic_canonical_impact.csv"
DEFAULT_OUT = ROOT / "results/analysis/main_table_1096_window_audit"
CANONICAL_MANIFEST_DIR = ROOT / "manifests/c_retrain_20260722"
SHARED_STREAM_ROOT = Path("/mnt/umm/users/yinbaiqiao/InsScene-15K/FEAT_STREAMING")
SHARED_TARGET_ROOT = Path("/mnt/umm/users/yinbaiqiao/InsScene-15K/FEAT_TARGET")

WINDOW_RE = re.compile(r"window_obs_len=(\d+), max_future=(\d+)")

J_EXACT = "EXACT_SAFE"
J_SHARED = "CONFIRMED_ERROR_SHARED24_VS_H124"
J_PREFIX_MANIFEST = "HIGH_RISK_PER_PREFIX_VS_H124"
J_DYNAMIC = "HIGH_RISK_DYNAMIC_OBS_VS_PER_PREFIX"
J_NO_CONFLICT = "UNRESOLVED_NO_OBS_CONFLICT"
J_NO_ROOT = "UNRESOLVED_FEATURE_ROOT"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def as_bool(value: str) -> bool:
    return value.strip().lower() == "true"


def last_dynamic_window(log_paths: str) -> tuple[int | None, int | None, str]:
    matches: list[tuple[int, int, str]] = []
    for raw_path in log_paths.split(";"):
        raw_path = raw_path.strip()
        if not raw_path:
            continue
        path = Path(raw_path)
        if not path.is_absolute():
            path = ROOT / path
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for obs_len, max_future in WINDOW_RE.findall(text):
            matches.append((int(obs_len), int(max_future), str(path)))
    return matches[-1] if matches else (None, None, "")


def feature_protocol(root: str, exact: bool, prefix: int) -> tuple[str, str]:
    if exact:
        return "MANIFEST_EXACT_VERSIONED", "manifest_exact"
    if not root:
        return "UNRESOLVED", ""
    if "FEAT_STREAMING_BY_PREFIX" in root:
        return "PER_PREFIX_GENERATOR_INFERRED", str(prefix)
    if "FEAT_STREAMING_PREFIX_ALIAS" in root:
        return "SHARED24_VIA_PREFIX_ALIAS", "24"
    if "FEAT_STREAMING" in root:
        return "SHARED24", "24"
    return "UNRESOLVED", ""


def evidence_grade(judgment: str, model: str) -> str:
    if judgment == J_EXACT:
        return "A_DATASET_CANARY_EXACT"
    if judgment == J_SHARED and model in {"wan", "minimax_h3"}:
        return "A_FULL_ACTUAL_INDEX_PLUS_CODE"
    if judgment == J_SHARED and model == "cogvideox":
        return "B_PARTIAL_ACTUAL_INDEX_PLUS_CODE"
    if judgment == J_SHARED:
        return "C_EXTRACTOR_CODE_AND_SHARED_PROTOCOL"
    if judgment in {J_PREFIX_MANIFEST, J_DYNAMIC}:
        return "D_PROTOCOL_CONFLICT_HISTORICAL_INDEX_MISSING"
    if judgment == J_NO_CONFLICT:
        return "D_NO_CONFLICT_HISTORICAL_INDEX_MISSING"
    return "E_UNRESOLVED"


def target_judgment(family: str, prefix: int, target_root: str, judgment: str) -> str:
    if family not in {"C1", "C2", "C3"}:
        return "NOT_USED_BY_PROBE"
    if judgment == J_EXACT:
        return "EXACT_SAFE_VERSIONED_TARGET"
    if not target_root:
        return "TARGET_ROOT_UNRESOLVED"
    if "FEAT_TARGET_BY_PREFIX" in target_root:
        return "PER_PREFIX_TARGET_RAW_IDS_UNVERIFIED"
    if "FEAT_TARGET_PREFIX_ALIAS" in target_root or "FEAT_TARGET" in target_root:
        if prefix == 12:
            return "CONFIRMED_PARTIAL_FUTURE_COVERAGE_PREFIX12"
        return "FLAT_FUTURE_UNION_COVERED_INPUT_STILL_NOT_SAFE"
    return "TARGET_PROTOCOL_UNRESOLVED"


def result_disposition(judgment: str, numeric: bool, lifecycle: str) -> str:
    if judgment == J_EXACT:
        return "KEEP_EXACT_RESULT"
    if judgment == J_SHARED:
        if numeric:
            return "INVALIDATE_OR_QUARANTINE_RESULT_AND_RERUN"
        if lifecycle == "TRAIN_COMPLETE_EVAL_NOT_STARTED":
            return "DO_NOT_EVAL_CHECKPOINT_REEXTRACT_AND_RETRAIN"
        return "NO_USABLE_RESULT_FIX_PROTOCOL_BEFORE_RERUN"
    if judgment in {J_PREFIX_MANIFEST, J_DYNAMIC}:
        if numeric:
            return "QUARANTINE_RESULT_PENDING_RAW_ID_AUDIT_OR_RERUN"
        return "DO_NOT_PROMOTE_PENDING_RAW_ID_AUDIT_OR_RERUN"
    if judgment in {J_NO_CONFLICT, J_NO_ROOT}:
        if numeric:
            return "QUARANTINE_RESULT_PENDING_EXACT_RAW_ID_AUDIT"
        return "DO_NOT_PROMOTE_PENDING_EXACT_RAW_ID_AUDIT"
    raise AssertionError(judgment)


def classify(
    q0: dict[str, str], q1: dict[str, str]
) -> tuple[str, int | None, int | None, str, str, str]:
    prefix = int(q1["prefix"])
    exact = q1["train_feature_semantic_status"] == "PROVEN_SAFE_EXACT"
    protocol, feature_obs_len = feature_protocol(
        q1["streaming_feature_root_train"], exact, prefix
    )
    if exact:
        return J_EXACT, prefix + 4, 4, "MANIFEST_H124", protocol, feature_obs_len

    if q1["train_manifest_path"]:
        dataset_obs_len = prefix + 4
        if protocol in {"SHARED24", "SHARED24_VIA_PREFIX_ALIAS"}:
            judgment = J_SHARED
        elif protocol == "PER_PREFIX_GENERATOR_INFERRED":
            judgment = J_PREFIX_MANIFEST
        else:
            judgment = J_NO_ROOT
        return judgment, dataset_obs_len, 4, "MANIFEST_H124", protocol, feature_obs_len

    obs_len, max_future, log_path = last_dynamic_window(q0["train_log_path"])
    if obs_len is None:
        return J_NO_ROOT, None, None, "DYNAMIC_LOG_MISSING", protocol, feature_obs_len
    judgment = J_DYNAMIC if obs_len != prefix else J_NO_CONFLICT
    return judgment, obs_len, max_future, f"DYNAMIC_LOG:{log_path}", protocol, feature_obs_len


def sequence_key(row: dict[str, Any]) -> tuple[str, str]:
    started = row["train_started_at"] or "9999-99-99 99:99:99"
    return started, row["experiment_id"]


def optional_shared_index_audit() -> dict[str, Any]:
    try:
        import numpy as np  # type: ignore
    except Exception as exc:  # pragma: no cover - environment dependent
        return {"status": "SKIPPED_NUMPY_UNAVAILABLE", "detail": str(exc)}

    models = ["wan", "minimax_h3", "cogvideox", "vjepa2", "lingbot"]
    output: dict[str, Any] = {"status": "COMPLETE", "models": {}}
    manifest_cache: dict[int, list[dict[str, Any]]] = {}
    for prefix in (8, 12, 16, 24):
        path = CANONICAL_MANIFEST_DIR / f"inscene_c_h124_prefix{prefix}.jsonl"
        manifest_cache[prefix] = [json.loads(line) for line in path.open(encoding="utf-8")]

    for model in models:
        model_result: dict[str, Any] = {}
        for prefix, manifest_rows in manifest_cache.items():
            exact = mismatch = missing = ambiguous = 0
            for row in manifest_rows:
                index_path = (
                    SHARED_STREAM_ROOT
                    / model
                    / "scannetpp"
                    / row["scene_id"]
                    / "prefix_index.npy"
                )
                if not index_path.is_file():
                    missing += 1
                    continue
                records = np.load(index_path, allow_pickle=True).tolist()
                hits = [
                    record
                    for record in records
                    if int(record.get("tail", -1)) == prefix - 1
                    and int(record.get("window_id", -1))
                    == int(row["streaming_window_id"])
                ]
                if len(hits) != 1:
                    ambiguous += 1
                    continue
                actual = [int(value) for value in hits[0]["indices"]]
                expected = [int(value) for value in row["streaming_indices"]]
                if actual == expected:
                    exact += 1
                else:
                    mismatch += 1
            checked = exact + mismatch
            model_result[str(prefix)] = {
                "manifest_rows": len(manifest_rows),
                "checked": checked,
                "exact": exact,
                "mismatch": mismatch,
                "exact_rate": (exact / checked) if checked else None,
                "missing_index": missing,
                "ambiguous_record": ambiguous,
            }
        output["models"][model] = model_result

    target_output: dict[str, Any] = {}
    for model in ("wan", "minimax_h3"):
        model_result = {}
        for prefix, manifest_rows in manifest_cache.items():
            horizon_coverage = {1: 0, 2: 0, 4: 0}
            all_horizons = missing = 0
            for row in manifest_rows:
                index_path = (
                    SHARED_TARGET_ROOT
                    / model
                    / "scannetpp"
                    / row["scene_id"]
                    / "target_indices.npy"
                )
                if not index_path.is_file():
                    missing += 1
                    continue
                available = {int(value) for value in np.load(index_path).tolist()}
                targets = []
                tail = int(row["streaming_tail"])
                for horizon in (1, 2, 4):
                    target = int(row["streaming_window_indices"][tail + horizon])
                    targets.append(target)
                    horizon_coverage[horizon] += int(target in available)
                all_horizons += int(all(target in available for target in targets))
            model_result[str(prefix)] = {
                "manifest_rows": len(manifest_rows),
                "h1": horizon_coverage[1],
                "h2": horizon_coverage[2],
                "h4": horizon_coverage[4],
                "all_horizons": all_horizons,
                "missing_index": missing,
            }
        target_output[model] = model_result
    output["target_future_coverage"] = target_output
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--skip-shared-index-audit", action="store_true")
    args = parser.parse_args()

    task_rows = read_csv(TASK_STATUS)
    q00_rows = read_csv(Q00)
    q01_rows = read_csv(Q01)
    atomic_rows = read_csv(ATOMIC)
    task_by_id = {row["experiment_id"]: row for row in task_rows}
    q00_by_id = {row["universe_id"]: row for row in q00_rows}
    q01_by_id = {row["universe_id"]: row for row in q01_rows}
    ids = set(task_by_id)
    assert len(task_rows) == len(q00_rows) == len(q01_rows) == 1096
    assert ids == set(q00_by_id) == set(q01_by_id)

    chronological = sorted(task_rows, key=sequence_key)
    rank = {row["experiment_id"]: idx + 1 for idx, row in enumerate(chronological)}
    audited: list[dict[str, Any]] = []
    for uid in sorted(ids):
        task = task_by_id[uid]
        q0 = q00_by_id[uid]
        q1 = q01_by_id[uid]
        judgment, dataset_obs, max_future, dataset_source, protocol, feature_obs = classify(q0, q1)
        numeric = as_bool(q0["numeric_result_present"])
        row: dict[str, Any] = {
            "chronological_rank": rank[uid],
            "experiment_id": uid,
            "campaign": q1["campaign"],
            "model": q1["model"],
            "model_variant": q1["model_variant"],
            "family": q1["family"],
            "probe": q1["probe"],
            "layer": q1["layer"],
            "prefix": q1["prefix"],
            "horizon": q1["horizon"],
            "seed": q1["seed"],
            "train_started_at": task["train_started_at"],
            "train_completed_at": task["train_completed_at"],
            "lifecycle_status": task["lifecycle_status"],
            "q00_status": q0["status"],
            "numeric_result_present": q0["numeric_result_present"],
            "canonical_valid_before_window_audit": q0["canonical_valid"],
            "canonical_ids": q1["canonical_ids"],
            "train_manifest_path": q1["train_manifest_path"],
            "train_manifest_resolution": q1["train_manifest_resolution"],
            "streaming_feature_root_train": q1["streaming_feature_root_train"],
            "target_feature_root_train": q1["target_feature_root_train"],
            "dataset_window_source": dataset_source,
            "dataset_obs_len": "" if dataset_obs is None else dataset_obs,
            "dataset_max_future": "" if max_future is None else max_future,
            "feature_window_protocol": protocol,
            "feature_obs_len": feature_obs,
            "window_judgment": judgment,
            "error_proven": str(judgment == J_SHARED).lower(),
            "exact_safe": str(judgment == J_EXACT).lower(),
            "safe_for_main_table": str(judgment == J_EXACT).lower(),
            "evidence_grade": evidence_grade(judgment, q1["model"]),
            "target_judgment": target_judgment(
                q1["family"], int(q1["prefix"]), q1["target_feature_root_train"], judgment
            ),
            "result_disposition": result_disposition(
                judgment, numeric, task["lifecycle_status"]
            ),
            "q01_previous_semantic_status": q1["train_feature_semantic_status"],
            "evidence_paths": q1["evidence_paths"],
        }
        audited.append(row)

    fields = list(audited[0])
    audited.sort(key=lambda row: int(row["chronological_rank"]))
    write_csv(args.out_dir / "window_audit_1096.csv", audited, fields)

    sequence_fields = [
        "chronological_rank",
        "experiment_id",
        "campaign",
        "probe",
        "family",
        "model",
        "model_variant",
        "layer",
        "prefix",
        "train_started_at",
        "train_completed_at",
        "lifecycle_status",
        "window_judgment",
    ]
    write_csv(args.out_dir / "probing_sequence_1096.csv", audited, sequence_fields)

    canonical = [row for row in audited if as_bool(row["canonical_valid_before_window_audit"])]
    assert len(canonical) == 642
    write_csv(args.out_dir / "canonical_parent_642_window_impact.csv", canonical, fields)

    audit_by_id = {row["experiment_id"]: row for row in audited}
    atomic_output = []
    for atomic in atomic_rows:
        parent = audit_by_id[atomic["parent_universe_id"]]
        atomic_output.append(
            {
                **atomic,
                "window_judgment": parent["window_judgment"],
                "safe_for_main_table": parent["safe_for_main_table"],
                "evidence_grade": parent["evidence_grade"],
                "target_judgment": parent["target_judgment"],
                "result_disposition": parent["result_disposition"],
            }
        )
    atomic_fields = list(atomic_output[0])
    write_csv(args.out_dir / "atomic_canonical_734_window_impact.csv", atomic_output, atomic_fields)

    followup = [
        row
        for row in audited
        if row["window_judgment"] in {J_PREFIX_MANIFEST, J_DYNAMIC, J_NO_CONFLICT, J_NO_ROOT}
    ]
    write_csv(args.out_dir / "raw_id_followup_queue_228.csv", followup, fields)
    assert len(followup) == 228

    judgment_counts = Counter(row["window_judgment"] for row in audited)
    campaign_counts: dict[str, Counter[str]] = defaultdict(Counter)
    family_counts: dict[str, Counter[str]] = defaultdict(Counter)
    lifecycle_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in audited:
        campaign_counts[row["campaign"]][row["window_judgment"]] += 1
        family_counts[row["family"]][row["window_judgment"]] += 1
        lifecycle_counts[row["lifecycle_status"]][row["window_judgment"]] += 1

    campaign_sequence = []
    for campaign in sorted({row["campaign"] for row in audited}):
        rows = [row for row in audited if row["campaign"] == campaign]
        starts = sorted(row["train_started_at"] for row in rows if row["train_started_at"])
        completions = sorted(
            row["train_completed_at"] for row in rows if row["train_completed_at"]
        )
        probe_first: dict[str, str] = {}
        for row in rows:
            if row["train_started_at"]:
                old = probe_first.get(row["probe"])
                if old is None or row["train_started_at"] < old:
                    probe_first[row["probe"]] = row["train_started_at"]
        campaign_sequence.append(
            {
                "campaign": campaign,
                "rows": len(rows),
                "first_started_at": starts[0] if starts else None,
                "last_completed_at": completions[-1] if completions else None,
                "probe_first_start_order": [
                    {"probe": probe, "first_started_at": started}
                    for probe, started in sorted(probe_first.items(), key=lambda item: (item[1], item[0]))
                ],
            }
        )
    campaign_sequence.sort(key=lambda row: row["first_started_at"] or "9999")

    shared_index_audit = (
        {"status": "SKIPPED_BY_FLAG"}
        if args.skip_shared_index_audit
        else optional_shared_index_audit()
    )
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "audit_policy": {
            "exact_safe_requires_dataset_canary": True,
            "equal_obs_len_is_not_exact_proof": True,
            "shared24_vs_h124_is_confirmed_protocol_error": True,
            "historical_per_prefix_cache_without_index_is_high_risk_not_proven_error": True,
        },
        "input_sha256": {
            str(TASK_STATUS.relative_to(ROOT)): sha256(TASK_STATUS),
            str(Q00.relative_to(ROOT)): sha256(Q00),
            str(Q01.relative_to(ROOT)): sha256(Q01),
            str(ATOMIC.relative_to(ROOT)): sha256(ATOMIC),
        },
        "counts": {
            "universe": len(audited),
            "canonical_parents": len(canonical),
            "atomic_canonical": len(atomic_output),
            "raw_id_followup_queue": len(followup),
            "by_judgment": dict(sorted(judgment_counts.items())),
            "canonical_by_judgment": dict(
                sorted(Counter(row["window_judgment"] for row in canonical).items())
            ),
            "atomic_by_judgment": dict(
                sorted(Counter(row["window_judgment"] for row in atomic_output).items())
            ),
        },
        "by_campaign": {key: dict(sorted(value.items())) for key, value in sorted(campaign_counts.items())},
        "by_family": {key: dict(sorted(value.items())) for key, value in sorted(family_counts.items())},
        "by_lifecycle": {key: dict(sorted(value.items())) for key, value in sorted(lifecycle_counts.items())},
        "campaign_sequence": campaign_sequence,
        "shared_index_audit": shared_index_audit,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary["counts"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
