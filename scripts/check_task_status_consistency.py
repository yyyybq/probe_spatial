#!/usr/bin/env python3
"""Consistency checks for the 1260-task lifecycle ledger (expanded from 1096)."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TASKS = ROOT / "experiments/task_status_1096.csv"
ATOMIC = ROOT / "results/atomic_probe_coverage.csv"
PAPER = ROOT / "results/paper_ready_results.csv"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--require-final", action="store_true",
        help="Also require all rows in the historical paper table to be final-claim-ready.",
    )
    args = parser.parse_args()
    violations: list[str] = []
    tasks = read_csv(TASKS)
    atomic = read_csv(ATOMIC)
    paper = read_csv(PAPER)

    if len(tasks) != 1260:
        violations.append(f"task_status_1096 row count is {len(tasks)}, expected 1260")
    ids = [row["experiment_id"] for row in tasks]
    if len(set(ids)) != len(ids):
        violations.append("task_status_1096 experiment_id values are not unique")
    for row in tasks:
        exp = row["experiment_id"]
        for field in ["training_status", "evaluation_status", "lifecycle_status", "paper_table_eligible", "status_reason"]:
            if not row.get(field):
                violations.append(f"{exp}: missing {field}")
        if row["training_status"] == "COMPLETE" and not row.get("train_completion_evidence"):
            violations.append(f"{exp}: training COMPLETE without completion evidence")
        if row["training_status"] == "COMPLETE" and not row.get("checkpoint_path"):
            violations.append(f"{exp}: training COMPLETE without checkpoint_path")
        if row["evaluation_status"] in {"COMPLETE_RAW", "COMPLETE_CANONICAL"} and not row.get("metric_evidence"):
            violations.append(f"{exp}: evaluation complete without metric_evidence")
        if row["paper_table_eligible"] == "YES" and not row.get("canonical_metric_path"):
            violations.append(f"{exp}: paper eligible without canonical metric")
        if row["lifecycle_status"] == "TRAIN_AND_EVAL_COMPLETE" and not (row.get("checkpoint_path") and (row.get("eval_output_path") or row.get("raw_metric_path") or row.get("canonical_metric_path"))):
            violations.append(f"{exp}: train+eval complete without checkpoint/eval output")
    atomic_ids = {row["experiment_id"] for row in atomic}
    missing_atomic = sorted(set(ids) - atomic_ids)
    if missing_atomic:
        violations.append(f"{len(missing_atomic)} tasks have no atomic coverage row")
    for row in paper:
        if row.get("paper_table_eligible") != "YES":
            violations.append(f"{row.get('experiment_id')}: paper table row is not eligible")
        if not row.get("canonical_metric_path"):
            violations.append(f"{row.get('experiment_id')}: paper table row lacks canonical metric path")
        if row.get("selection_policy") != "FULL_SWEEP":
            violations.append(f"{row.get('experiment_id')}: paper table row uses unexpected selection policy")
    bad_words = {"FAILED", "PARTIAL", "PROVISIONAL", "SUPERSEDED"}
    task_by_id = {row["experiment_id"]: row for row in tasks}
    eligible_ids = {row["experiment_id"] for row in tasks if row.get("paper_table_eligible") == "YES"}
    missing_paper = eligible_ids - {row["experiment_id"] for row in paper}
    if missing_paper:
        violations.append(f"{len(missing_paper)} eligible tasks have no exported paper-table row")
    for row in paper:
        task = task_by_id.get(row["experiment_id"])
        if task is None:
            violations.append(f"{row['experiment_id']}: paper row has no lifecycle task")
            continue
        if task.get("paper_table_eligible") != "YES":
            violations.append(f"{row['experiment_id']}: lifecycle task is not paper eligible")
        status_text = " ".join([task.get("training_status", ""), task.get("evaluation_status", ""), task.get("lifecycle_status", "")])
        if any(word in status_text for word in bad_words):
            violations.append(f"{row.get('experiment_id')}: paper table includes invalid status {status_text}")

    if args.require_final:
        nonfinal = sum(row.get("final_claim_ready") != "YES" for row in paper)
        if not paper or nonfinal:
            violations.append(f"final-result gate: {nonfinal}/{len(paper)} paper rows are not final-claim-ready")

    if violations:
        print("FAIL: task status consistency violations")
        for item in violations[:200]:
            print(f"- {item}")
        if len(violations) > 200:
            print(f"... {len(violations) - 200} more")
        raise SystemExit(1)
    print("PASS: task_status_1096, atomic coverage, and paper-ready tables are internally consistent")
    if not args.require_final:
        print("This checks historical table consistency, not final-result readiness; use --require-final for that gate.")


if __name__ == "__main__":
    main()
