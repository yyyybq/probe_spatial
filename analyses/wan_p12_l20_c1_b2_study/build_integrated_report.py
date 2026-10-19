#!/usr/bin/env python3
"""Build the integrated Wan P12/L20 C1/B2 report from structured outputs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from string import Template
from typing import Dict, Iterable, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = PROJECT_ROOT / "reports/wan_p12_l20_c1_b2_integrated_analysis.md"


def read_csv(path: Path) -> list[Dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def select_one(rows: Iterable[Mapping[str, str]], **criteria: str) -> Mapping[str, str]:
    matches = [
        row
        for row in rows
        if all(str(row.get(key, "")) == str(value) for key, value in criteria.items())
    ]
    if len(matches) != 1:
        raise RuntimeError(f"expected one row for {criteria}, found {len(matches)}")
    return matches[0]


def number(row: Mapping[str, str], key: str) -> float:
    value = row.get(key)
    if value in (None, ""):
        raise RuntimeError(f"missing numeric field {key} in selected row")
    return float(value)


def condition_map(rows: Iterable[Mapping[str, str]]) -> Dict[str, Mapping[str, str]]:
    result = {str(row["condition"]): row for row in rows}
    required = {
        "correct_action",
        "shuffled_action",
        "eval_null_action",
        "copy_last_streaming_state",
        "copy_prefix_pooled_state",
    }
    missing = sorted(required - set(result))
    if missing:
        raise RuntimeError(f"missing P12 conditions: {missing}")
    return result


def b2_record(rows: list[Dict[str, str]], baseline: str) -> Dict[str, float | int]:
    # The integrated report is pinned to the original Wan P12/L20 paired study.
    # Later paper-strengthening rows share the CSV but are a separate analysis.
    row = select_one(
        rows,
        model="wan",
        prefix="12",
        analysis_type="phase2_paired_method",
        baseline_or_readout=baseline,
    )
    return {
        "n": int(number(row, "n")),
        "scenes": int(number(row, "n_scenes")),
        "error": number(row, "mean_ang_err_deg"),
        "delta": number(row, "delta_error_vs_full"),
        "ci_low": number(row, "ci_low"),
        "ci_high": number(row, "ci_high"),
    }


def geometry_evidence(project_root: Path) -> Dict[str, float]:
    rows = read_csv(
        project_root
        / "results/paper_validity_20260821T200224Z_2075f3e/c_baselines_summary.csv"
    )
    wan_full = select_one(
        rows,
        task="C1",
        model="wan",
        prefix="12",
        variant="full",
        sample_set="common_models_and_prefixes_legacy_pair",
    )
    wan_copy = select_one(
        rows,
        task="C1",
        model="wan",
        prefix="12",
        variant="copy_last_context",
        sample_set="common_models_and_prefixes_legacy_pair",
    )
    dino_copy = select_one(
        rows,
        task="C1",
        model="dino",
        prefix="12",
        variant="copy_last_context",
        sample_set="common_models_and_prefixes_legacy_pair",
    )
    return {
        "wan_copy_raw_cosine": number(wan_copy, "correct_target_cosine"),
        "wan_full_raw_cosine": number(wan_full, "correct_target_cosine"),
        "wan_temporal_displacement": 1.0 - number(wan_copy, "correct_target_cosine"),
        "wan_copy_centered_cosine": number(wan_copy, "centered_cosine"),
        "wan_full_centered_cosine": number(wan_full, "centered_cosine"),
        "dino_copy_centered_cosine": number(dino_copy, "centered_cosine"),
        "dino_copy_r1": number(dino_copy, "global_R@1"),
    }


def collect_evidence(project_root: Path = PROJECT_ROOT) -> dict:
    strict_root = project_root / "results/analysis/wan_p12_l20_action_validity"
    strict_summary = json.loads((strict_root / "summary.json").read_text())
    conditions = condition_map(read_csv(strict_root / "tables/condition_metrics.csv"))
    c1_rows = read_csv(project_root / "analyses/probe_validity/c1_validity.csv")
    b2_rows = read_csv(project_root / "analyses/probe_validity/b2_validity.csv")

    historical = select_one(
        c1_rows,
        model="wan",
        analysis_type="phase2_paired_candidate_pool",
        candidate_pool="HISTORICAL_GLOBAL",
        action_condition="true",
    )
    same_scene = select_one(
        c1_rows,
        model="wan",
        analysis_type="phase2_paired_candidate_pool",
        candidate_pool="SAME_SCENE",
        action_condition="true",
    )
    contract = {
        "model": str(strict_summary["model"]),
        "prefix": int(strict_summary["prefix"]),
        "layer": int(strict_summary["layer"]),
    }
    if contract != {"model": "wan", "prefix": 12, "layer": 20}:
        raise RuntimeError(f"unexpected frozen contract: {contract}")

    gates = strict_summary["phase1_gate_detail"]
    strict = {
        "interpretation": str(strict_summary["final_interpretation"]),
        "phase1_gate": str(strict_summary["phase1_gate"]),
        "correct_r1": number(conditions["correct_action"], "r1"),
        "formal_shuffle_r1": number(conditions["shuffled_action"], "r1"),
        "null_r1": number(conditions["eval_null_action"], "r1"),
        "last_copy_r1": number(conditions["copy_last_streaming_state"], "r1"),
        "prefix_copy_r1": number(conditions["copy_prefix_pooled_state"], "r1"),
        "formal_gain": float(strict_summary["correct_vs_shuffled"]["estimate"]),
        "formal_ci_low": float(strict_summary["correct_vs_shuffled"]["ci_low"]),
        "formal_ci_high": float(strict_summary["correct_vs_shuffled"]["ci_high"]),
        "permutation_p": float(gates["permutation_p_r1"]),
        "null_gain": float(strict_summary["correct_vs_no_condition"]["estimate"]),
        "null_ci_low": float(strict_summary["correct_vs_no_condition"]["ci_low"]),
        "null_ci_high": float(strict_summary["correct_vs_no_condition"]["ci_high"]),
        "wrong_gain": float(strict_summary["correct_vs_all_wrong_cos"]["estimate"]),
        "wrong_ci_low": float(strict_summary["correct_vs_all_wrong_cos"]["ci_low"]),
        "wrong_ci_high": float(strict_summary["correct_vs_all_wrong_cos"]["ci_high"]),
        "copy_gain": float(strict_summary["correct_vs_copy"]["estimate"]),
        "copy_ci_low": float(strict_summary["correct_vs_copy"]["ci_low"]),
        "copy_ci_high": float(strict_summary["correct_vs_copy"]["ci_high"]),
        "positive_classes": int(gates["positive_action_classes"]),
        "total_classes": int(gates["total_action_classes"]),
        "positive_scenes": int(gates["positive_scenes"]),
        "total_scenes": int(gates["total_scenes"]),
    }
    c1 = {
        "n": int(number(historical, "n")),
        "scenes": int(number(historical, "n_scenes")),
        "legacy_shuffle_gain": number(historical, "delta_vs_shuffle"),
        "matched_gain": number(historical, "true_minus_matched"),
        "matched_ci_low": number(historical, "ci_low"),
        "matched_ci_high": number(historical, "ci_high"),
        "same_scene_gain": number(same_scene, "true_minus_matched"),
        "same_scene_ci_low": number(same_scene, "ci_low"),
        "same_scene_ci_high": number(same_scene, "ci_high"),
    }
    b2 = {
        "full": b2_record(b2_rows, "FULL_HISTORY_PLUS_QUERY"),
        "current_frame": b2_record(b2_rows, "CURRENT_FRAME_PLUS_QUERY"),
        "remove_visible": b2_record(b2_rows, "REMOVE_TARGET_VISIBLE_FRAMES"),
        "query_only": b2_record(b2_rows, "QUERY_FEATURE_ONLY"),
        "query_mask": b2_record(b2_rows, "HISTORY_WITHOUT_QUERY"),
        "query_swap": b2_record(b2_rows, "SAME_SCENE_QUERY_SWAP"),
        "shuffle": b2_record(b2_rows, "FRAME_ORDER_SHUFFLE"),
        "reverse": b2_record(b2_rows, "REVERSED_HISTORY"),
        "global_prior": b2_record(b2_rows, "GLOBAL_CONSTANT"),
    }
    return {
        "contract": contract,
        "strict": strict,
        "c1": c1,
        "b2": b2,
        "geometry": geometry_evidence(project_root),
    }


def signed(value: float, digits: int = 4) -> str:
    return f"{value:+.{digits}f}"


def plain(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}"


def report_values(evidence: dict) -> Dict[str, str]:
    strict = evidence["strict"]
    c1 = evidence["c1"]
    b2 = evidence["b2"]
    geometry = evidence["geometry"]
    values: Dict[str, str] = {
        "prefix": str(evidence["contract"]["prefix"]),
        "layer": str(evidence["contract"]["layer"]),
        "c1_n": str(c1["n"]),
        "c1_scenes": str(c1["scenes"]),
        "interpretation": strict["interpretation"],
        "phase1_gate": strict["phase1_gate"],
        "correct_r1": plain(strict["correct_r1"]),
        "formal_shuffle_r1": plain(strict["formal_shuffle_r1"]),
        "null_r1": plain(strict["null_r1"]),
        "last_copy_r1": plain(strict["last_copy_r1"]),
        "prefix_copy_r1": plain(strict["prefix_copy_r1"]),
        "formal_gain": signed(strict["formal_gain"]),
        "formal_ci_low": signed(strict["formal_ci_low"]),
        "formal_ci_high": signed(strict["formal_ci_high"]),
        "permutation_p": plain(strict["permutation_p"]),
        "null_gain": signed(strict["null_gain"]),
        "null_ci_low": signed(strict["null_ci_low"]),
        "null_ci_high": signed(strict["null_ci_high"]),
        "wrong_gain": signed(strict["wrong_gain"], 6),
        "wrong_ci_low": signed(strict["wrong_ci_low"], 6),
        "wrong_ci_high": signed(strict["wrong_ci_high"], 6),
        "copy_gain": signed(strict["copy_gain"]),
        "copy_ci_low": signed(strict["copy_ci_low"]),
        "copy_ci_high": signed(strict["copy_ci_high"]),
        "positive_classes": str(strict["positive_classes"]),
        "total_classes": str(strict["total_classes"]),
        "positive_scenes": str(strict["positive_scenes"]),
        "total_scenes": str(strict["total_scenes"]),
        "legacy_shuffle_gain": signed(c1["legacy_shuffle_gain"]),
        "matched_gain": signed(c1["matched_gain"]),
        "matched_ci_low": signed(c1["matched_ci_low"]),
        "matched_ci_high": signed(c1["matched_ci_high"]),
        "same_scene_gain": signed(c1["same_scene_gain"]),
        "same_scene_ci_low": signed(c1["same_scene_ci_low"]),
        "same_scene_ci_high": signed(c1["same_scene_ci_high"]),
    }
    for name, record in b2.items():
        values[f"{name}_n"] = str(record["n"])
        values[f"{name}_scenes"] = str(record["scenes"])
        values[f"{name}_error"] = plain(record["error"], 2)
        values[f"{name}_delta"] = signed(record["delta"], 2)
        values[f"{name}_ci_low"] = signed(record["ci_low"], 2)
        values[f"{name}_ci_high"] = signed(record["ci_high"], 2)
    for name, value in geometry.items():
        values[name] = plain(value)
    return values


def render_report(evidence: dict, package_dir: Path = PACKAGE_DIR) -> str:
    template = Template((package_dir / "integrated_report_template.md").read_text())
    return template.substitute(report_values(evidence)).rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    root = args.project_root.resolve()
    output = args.output if args.output.is_absolute() else root / args.output
    evidence = collect_evidence(root)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_report(evidence, PACKAGE_DIR), encoding="utf-8")
    print(json.dumps({"ok": True, "output": str(output), "evidence": evidence}, indent=2))


if __name__ == "__main__":
    main()
