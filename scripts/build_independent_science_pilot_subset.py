#!/usr/bin/env python3
"""Deterministically select one shared, four-model science pilot from frozen manifests.

The script never changes a protocol manifest.  It writes a selection receipt and
an exact feature-request-v2 subset whose rows are copied verbatim from v2.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable


TAG = "independent_actionability_algebra_p8_20260831T091224Z_28bda217"
ROOT = Path(__file__).resolve().parents[1]
MANIFESTS = ROOT / "artifacts" / TAG / "manifests"
REPORTS = ROOT / "reports" / TAG
SEED = "independent-p8-science-pilot-v1-20260831"

SOURCES = {
    "actionability_train": MANIFESTS / "actionability_p8_independent_v1_train.jsonl",
    "actionability_val": MANIFESTS / "actionability_p8_independent_v1_val.jsonl",
    "algebra_train": MANIFESTS / "algebra_p8_independent_v1_train.jsonl",
    "algebra_val": MANIFESTS / "algebra_p8_independent_v1_val.jsonl",
    "algebra_real_loops": MANIFESTS / "algebra_p8_independent_v1_real_loops.jsonl",
}
FEATURE_V2 = MANIFESTS / "feature_request_manifest_v2.jsonl"
PREFIX = "science_pilot_subset_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def digest_key(value: str) -> str:
    return hashlib.sha256(f"{SEED}:{value}".encode()).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def atomic_create(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    atomic_create(path, "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows))


def motion_bins(rows: list[dict[str, Any]]) -> dict[str, int]:
    ordered = sorted(rows, key=lambda row: (float(row["cumulative_motion"]), row["sample_id"]))
    n = len(ordered)
    return {row["sample_id"]: min(2, (index * 3) // n) for index, row in enumerate(ordered)}


def balanced_scene_selection(
    rows: list[dict[str, Any]], target: int, stratum: Callable[[dict[str, Any]], str]
) -> list[dict[str, Any]]:
    """Cover every available scene first, then round-robin strata deterministically."""
    if target > len(rows):
        raise ValueError(f"target {target} exceeds {len(rows)} candidates")
    by_scene: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_stratum: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_scene[row["scene_id"]].append(row)
        by_stratum[stratum(row)].append(row)
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    counts: Counter[str] = Counter()
    scene_counts: Counter[str] = Counter()
    # This phase guarantees maximum scene coverage because all requested sizes
    # exceed the number of frozen scenes in the split.
    for scene in sorted(by_scene):
        candidate = min(
            by_scene[scene],
            key=lambda row: (counts[stratum(row)], digest_key(row["sample_id"])),
        )
        selected.append(candidate)
        selected_ids.add(candidate["sample_id"])
        counts[stratum(candidate)] += 1
        scene_counts[scene] += 1
        if len(selected) == target:
            return selected
    while len(selected) < target:
        available = [name for name, values in by_stratum.items() if any(row["sample_id"] not in selected_ids for row in values)]
        if not available:
            raise RuntimeError("candidate exhaustion during balanced selection")
        chosen_stratum = min(available, key=lambda name: (counts[name], len(by_stratum[name]), name))
        candidates = [row for row in by_stratum[chosen_stratum] if row["sample_id"] not in selected_ids]
        candidate = min(candidates, key=lambda row: (scene_counts[row["scene_id"]], digest_key(row["sample_id"])))
        selected.append(candidate)
        selected_ids.add(candidate["sample_id"])
        counts[chosen_stratum] += 1
        scene_counts[candidate["scene_id"]] += 1
    return selected


def action_stratum(row: dict[str, Any]) -> str:
    motion = "translation" if row["translation_dominant"] else "rotation"
    return f"label={row['best_candidate']}|goal={row['goal_temporal_direction']}|motion={motion}"


def algebra_stratum_factory(rows: list[dict[str, Any]]) -> Callable[[dict[str, Any]], str]:
    bins = motion_bins(rows)
    return lambda row: f"path={row['path_length']}|motion_tertile={bins[row['sample_id']]}"


def record_stats(rows: list[dict[str, Any]], stratum: Callable[[dict[str, Any]], str]) -> dict[str, Any]:
    return {
        "records": len(rows),
        "scenes": len({row["scene_id"] for row in rows}),
        "strata": dict(sorted(Counter(stratum(row) for row in rows).items())),
        "sample_id_sha256": hashlib.sha256("\n".join(sorted(row["sample_id"] for row in rows)).encode()).hexdigest(),
    }


def main() -> int:
    source_rows = {name: read_jsonl(path) for name, path in SOURCES.items()}
    action_train = balanced_scene_selection(source_rows["actionability_train"], 512, action_stratum)
    action_val = balanced_scene_selection(source_rows["actionability_val"], 128, action_stratum)
    algebra_train_stratum = algebra_stratum_factory(source_rows["algebra_train"])
    algebra_val_stratum = algebra_stratum_factory(source_rows["algebra_val"])
    algebra_train = balanced_scene_selection(source_rows["algebra_train"], 512, algebra_train_stratum)
    algebra_val = balanced_scene_selection(source_rows["algebra_val"], 128, algebra_val_stratum)

    loops = {row["sample_id"]: row for row in source_rows["algebra_real_loops"]}
    loop_pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for row in loops.values():
        if row["real_loop_flag"]:
            mate = loops.get(row["matched_non_loop_id"])
            if mate is None or mate["real_loop_flag"] or mate.get("matched_non_loop_id") != row["sample_id"]:
                raise RuntimeError(f"invalid real-loop pair for {row['sample_id']}")
            loop_pairs.append((row, mate))
    loop_pairs.sort(key=lambda pair: digest_key(pair[0]["sample_id"]))
    selected_pairs = loop_pairs[:64]
    if len(selected_pairs) != 64:
        raise RuntimeError(f"expected >=64 real-loop pairs, found {len(loop_pairs)}")
    selected_loops = [row for pair in selected_pairs for row in pair]

    selected = {
        "actionability_train": action_train,
        "actionability_val": action_val,
        "algebra_train": algebra_train,
        "algebra_val": algebra_val,
        "algebra_real_loops": selected_loops,
    }
    output_paths = {name: MANIFESTS / f"{PREFIX}_{name}.jsonl" for name in selected}
    for name, rows in selected.items():
        write_jsonl(output_paths[name], rows)

    selected_ids = {row["sample_id"] for rows in selected.values() for row in rows}
    feature_rows = read_jsonl(FEATURE_V2)
    universal = [row for row in feature_rows if selected_ids.intersection(row["source_sample_ids"])]
    dino = [row for row in universal if row["model"] == "dino" and row["layer"] == 0 and row["timestep"] is None]
    universal_coverage: dict[str, list[str]] = {}
    for model in ("dino", "f3r", "vjepa2", "wan"):
        found = {sample_id for row in universal if row["model"] == model for sample_id in row["source_sample_ids"]}
        missing = sorted(selected_ids - found)
        if missing:
            universal_coverage[model] = missing
    if universal_coverage:
        raise RuntimeError(f"feature v2 lacks selected sample coverage: {universal_coverage}")
    if any(row.get("exact_raw_id_required") is not True or row.get("fallback_forbidden") is not True for row in universal):
        raise RuntimeError("pilot request contains a non-exact unit")
    universal_path = MANIFESTS / f"{PREFIX}_feature_request_v2_all_models.jsonl"
    dino_path = MANIFESTS / f"{PREFIX}_feature_request_v2_dino_l0.jsonl"
    write_jsonl(universal_path, universal)
    write_jsonl(dino_path, dino)

    summary = {
        "schema_version": 1,
        "science_tag": TAG,
        "status": "COMPLETE_VALID",
        "operation": "deterministic_immutable_subset_selection",
        "seed": SEED,
        "source_manifest_sha256": {name: sha256(path) for name, path in SOURCES.items()},
        "source_feature_request_v2": str(FEATURE_V2),
        "source_feature_request_v2_sha256": sha256(FEATURE_V2),
        "selected_manifests": {name: {"path": str(output_paths[name]), "sha256": sha256(output_paths[name])} for name in selected},
        "selection": {
            "actionability_train": record_stats(action_train, action_stratum),
            "actionability_val": record_stats(action_val, action_stratum),
            "algebra_train": record_stats(algebra_train, algebra_train_stratum),
            "algebra_val": record_stats(algebra_val, algebra_val_stratum),
            "algebra_real_loop_pairs": {"pairs": len(selected_pairs), "records": len(selected_loops), "scenes": len({row['scene_id'] for row in selected_loops})},
            "shared_sample_id_count": len(selected_ids),
        },
        "feature_request": {
            "all_models_path": str(universal_path),
            "all_models_sha256": sha256(universal_path),
            "all_models_units": len(universal),
            "units_by_model": dict(sorted(Counter(row["model"] for row in universal).items())),
            "dino_l0_path": str(dino_path),
            "dino_l0_sha256": sha256(dino_path),
            "dino_l0_units": len(dino),
            "identity_contract": "all feature units copied verbatim from feature-request v2; raw ID exact and fallback forbidden",
        },
    }
    summary_path = MANIFESTS / f"{PREFIX}.summary.json"
    atomic_create(summary_path, json.dumps(summary, indent=2, sort_keys=True) + "\n")
    report = [
        "# Four-model shared science pilot (v1)",
        "",
        "This is a deterministic selection from frozen manifests and a verbatim subset of feature-request v2; no protocol or source manifest was changed.",
        "",
        f"- Seed: `{SEED}`",
        f"- Actionability: train {len(action_train)}, val {len(action_val)}; all available scenes are represented before balance fill.",
        f"- Algebra: train {len(algebra_train)}, val {len(algebra_val)}; 64 paired real-loop/non-loop examples are retained for frozen-dynamics analysis.",
        f"- Shared selected sample IDs: {len(selected_ids)}",
        f"- Exact DINO L0 units needed now: {len(dino)}",
        f"- Corresponding all-model request units: {len(universal)}",
        "",
        "## Strata counts",
        "",
        "```json",
        json.dumps(summary["selection"], indent=2, sort_keys=True),
        "```",
        "",
        "## Immutable request receipts",
        "",
        f"- all models SHA256: `{summary['feature_request']['all_models_sha256']}`",
        f"- DINO L0 SHA256: `{summary['feature_request']['dino_l0_sha256']}`",
    ]
    report_path = REPORTS / "SCIENCE_PILOT_SUBSET_V1.md"
    atomic_create(report_path, "\n".join(report) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
