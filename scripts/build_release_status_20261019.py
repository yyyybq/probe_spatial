#!/usr/bin/env python3
"""Join the 2026-10-16 run inventories with the accepted v2 evaluation tables.

The inventories in ``metadata/release_inventory/`` (canonical_v2.csv, timestep.csv)
record training completion and checkpoint validity as of 2026-10-16. The accepted
evaluations of 2026-10-17/18 live in three result tables. This script does not
modify either source. It writes one row per inventory run with its evaluation
status and a single release class, plus a summary with input hashes:

    metadata/release_inventory/release_status_20261019.csv
    metadata/release_inventory/release_status_20261019_summary.json

Units: one row is one trained run (model, probe, layer, prefix, timestep, seed).
A run replaced by a 2026-10-16 fp32 re-train keeps its inventory identity; the
accepted row then points to the new checkpoint (``checkpoint_replaced``).
C3-from-C1 is an evaluation of an existing C1 run, not a separate training, so it
is attached to the C1 row instead of adding rows.

Usage:  python scripts/build_release_status_20261019.py [--check]
``--check`` recomputes and fails if the written files differ.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INV = ROOT / "metadata/release_inventory"
INPUTS = {
    "canonical_inventory": INV / "canonical_v2.csv",
    "timestep_inventory": INV / "timestep.csv",
    "retrain_queue": INV / "retrain_queue_20261016.csv",
    "canonical_accepted": ROOT / "results/canonical_v2_eval_v2/final/production_metrics.csv",
    "timestep_accepted": ROOT / "results/tstep_eval_v2/t1/production_metrics.csv",
    "c3_from_c1_accepted": ROOT / "results/c3_from_c1_eval/c3/production_metrics.csv",
}
OUT_CSV = INV / "release_status_20261019.csv"
OUT_JSON = INV / "release_status_20261019_summary.json"

# Training-code identities recorded by the evaluation tables. The commits behind
# these tags were not pushed with the branch (checked 2026-10-19: none of the 17
# commits in artifacts/release_audit_20261017/code_alignment/pack_commits.tsv is
# in the GitHub repository); e4e4886 is.
SNAPSHOT_TAGS = {"code/tstep_probe_opt_v3", "code/tstep_probe_pack_v4_1"}
# Date each evaluation revision was run (UTC), from docs/EVALUATION_PLAN_20261017.md
# and docs/TSTEP_EVALUATION_20261018.md.
EVAL_DATES = {"r2": "2026-10-17", "r3": "2026-10-18", "r4": "2026-10-18", "t1": "2026-10-18"}
PUBLISHED_TRAINING_CODE = {"commit e4e4886"}

# Release class -> remaining-work category used in the reports.
CLASSES = {
    "ACCEPTED_VAL_RESULT": "export_only",
    "AWAITING_STAGE5_RETRAIN": "retrain_needed",
    "WAN_A2_CANARY_CONFLICT": "needs_diagnosis",
    "UNRECOVERABLE_TRAINING_CODE_20260907": "user_decision",
    "LEGACY_TRAINED_C3": "not_in_release",
    "NONCANONICAL_PROBE": "optional_extension",
}

FIELDS = [
    "group", "experiment_id", "scope", "model", "probe", "layer", "prefix", "timestep",
    "inventory_training_status", "inventory_correctness_status", "inventory_result_status",
    "inventory_checkpoint_tensor_status", "inventory_observed_at", "release_class", "remaining_work_category",
    "eval_status", "eval_table", "eval_revision", "eval_date", "eval_scope", "training_code_tag",
    "eval_pack", "primary_metric", "primary_value", "primary_ci_low", "primary_ci_high",
    "n_loader", "n_scored", "checkpoint_replaced", "accepted_checkpoint_sha256",
    "caveats", "c3_variant", "c3_intervention_validity", "c3_ci_low", "c3_ci_high",
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def timestep_key(inventory_id: str) -> str:
    """'wan:scannetpp:499:15:8:streaming_depth:seed42' -> table unique_key format."""
    model, dataset, timestep, layer, prefix, probe, seed = inventory_id.split(":")
    return f"{model}__{dataset}__t{timestep}__l{layer}__p{prefix}__{probe}__{seed}"


def unique(rows: list[dict[str, str]], key: str, name: str) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for row in rows:
        if row[key] in out:
            raise SystemExit(f"{name}: duplicate {key} {row[key]!r}")
        out[row[key]] = row
    return out


def classify(group: str, inv: dict[str, str], accepted: dict | None, stage5: set[str]) -> str:
    if accepted is not None:
        return "ACCEPTED_VAL_RESULT"
    if group == "canonical_v2":
        if inv["scope"].endswith("_noncanonical_probe"):
            return "NONCANONICAL_PROBE"
        if inv["model"] == "wan" and inv["probe"] == "view_consistency":
            return "WAN_A2_CANARY_CONFLICT"
    else:
        if inv["experiment_id"] in stage5:
            return "AWAITING_STAGE5_RETRAIN"
        if "20260907" in inv["evidence_path"]:
            return "UNRECOVERABLE_TRAINING_CODE_20260907"
        if inv["probe"] == "counterfactual":
            return "LEGACY_TRAINED_C3"
    raise SystemExit(f"{group}: unclassified run without accepted evaluation: {inv['experiment_id']}")


def caveats(group: str, acc: dict[str, str], replaced: bool) -> list[str]:
    out = ["VAL_SPLIT_SEED42_ONLY"]
    if acc["scope"] == "canonical_v2_minimax_all_layers_20261011":
        out.append("VAL_171_SAMPLES_NOT_COMPARABLE_TO_276")
    if acc["training_code_tag"] in SNAPSHOT_TAGS:
        out.append("TRAINING_CODE_SNAPSHOT_IDENTITY_20261017")
    if acc["training_code_tag"] not in PUBLISHED_TRAINING_CODE:
        out.append("TRAINING_AND_EVAL_PACK_COMMITS_NOT_ON_GITHUB")
    if replaced:
        out.append("FP32_RETRAIN_REPLACES_NONFINITE_CHECKPOINT")
    if acc["probe"] in {"ego_belief", "ego_belief_v2"} and acc["n_scored"] != acc["n_loader"]:
        out.append(f"B2_SCORED_{acc['n_scored']}_OF_{acc['n_loader']}")
    if acc["probe"] in {"action_dynamics", "path_integration"}:
        out.append("REPORT_WITH_NO_ACTION_SHUFFLED_COPY_LAST_CONTROLS")
    return out


def build() -> tuple[str, str]:
    for name, path in INPUTS.items():
        if not path.is_file():
            raise SystemExit(f"missing input {name}: {path}")
    canon_inv = [r for r in read(INPUTS["canonical_inventory"]) if r["entry_level"] == "run"]
    tstep_inv = [r for r in read(INPUTS["timestep_inventory"]) if r["entry_level"] == "run"]
    queue = read(INPUTS["retrain_queue"])
    stage5 = {r["experiment_id"] for r in queue if r["queue_group"] == "TSTEP_MINIMAX_L20_L25_FEATURE_OVERFLOW"}
    tables = {
        "canonical_v2": unique(read(INPUTS["canonical_accepted"]), "unique_key", "canonical table"),
        "timestep": unique(read(INPUTS["timestep_accepted"]), "unique_key", "timestep table"),
    }
    c3 = unique(read(INPUTS["c3_from_c1_accepted"]), "unique_key", "C3 table")
    for name, table in (*tables.items(), ("c3_from_c1", c3)):
        bad = Counter(r["status"] for r in table.values() if r["status"] != "ACCEPTED")
        if bad:
            raise SystemExit(f"{name}: non-ACCEPTED rows {dict(bad)}")

    rows, seen = [], {g: set() for g in tables}
    for group, inventory in (("canonical_v2", canon_inv), ("timestep", tstep_inv)):
        table_name = "results/canonical_v2_eval_v2/final" if group == "canonical_v2" else "results/tstep_eval_v2/t1"
        for inv in inventory:
            key = inv["experiment_id"] if group == "canonical_v2" else timestep_key(inv["experiment_id"])
            acc = tables[group].get(key)
            release_class = classify(group, inv, acc, stage5)
            row = {
                "group": group, "experiment_id": inv["experiment_id"],
                "scope": inv.get("scope") or inv.get("campaign", ""),
                "model": inv["model"], "probe": inv["probe"], "layer": inv["layer"],
                "prefix": inv["prefix"], "timestep": inv["timestep"],
                "inventory_training_status": inv["training_status"],
                "inventory_correctness_status": inv["correctness_status"],
                "inventory_result_status": inv["result_status"],
                "inventory_checkpoint_tensor_status": inv.get("checkpoint_tensor_status", ""),
                "inventory_observed_at": inv.get("observed_at", ""),
                "release_class": release_class,
                "remaining_work_category": CLASSES[release_class],
                "eval_status": "NOT_EVALUATED", "checkpoint_replaced": "",
            }
            if acc is not None:
                seen[group].add(key)
                replaced = Path(acc["checkpoint_path"]).parts[-6:] != Path(inv["checkpoint_path"]).parts[-6:]
                replaced = replaced and acc["scope"].startswith(("canonical_v2_b2_fp32", "tstep_b2_fp32"))
                row.update({
                    "eval_status": "ACCEPTED_V2", "eval_table": table_name,
                    "eval_revision": acc.get("eval_revision", "t1"),
                    "eval_date": EVAL_DATES[acc.get("eval_revision", "t1")], "eval_scope": acc["scope"],
                    **{k: acc[k] for k in ("training_code_tag", "eval_pack", "primary_metric", "primary_value",
                                           "primary_ci_low", "primary_ci_high", "n_loader", "n_scored")},
                    "checkpoint_replaced": str(replaced).lower(),
                    "accepted_checkpoint_sha256": acc["checkpoint_sha256"],
                    "caveats": ";".join(caveats(group, acc, replaced)),
                })
                if key in c3:
                    row.update({"c3_variant": c3[key]["variant"],
                                "c3_intervention_validity": c3[key]["primary_value"],
                                "c3_ci_low": c3[key]["primary_ci_low"],
                                "c3_ci_high": c3[key]["primary_ci_high"]})
            rows.append(row)

    for group, table in tables.items():
        missing = sorted(set(table) - seen[group])
        if missing:
            raise SystemExit(f"{group}: {len(missing)} accepted rows have no inventory run, e.g. {missing[:3]}")
    attached = sum(1 for r in rows if r.get("c3_variant"))
    if attached != len(c3):
        raise SystemExit(f"C3-from-C1: attached {attached} of {len(c3)} rows to C1 runs")

    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=FIELDS, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({k: row.get(k, "") for k in FIELDS})

    def count(group: str, field: str) -> dict[str, int]:
        return dict(sorted(Counter(r[field] for r in rows if r["group"] == group).items()))

    summary = {
        "schema": "PROBE_SPATIAL_RELEASE_STATUS_V1",
        "unit": "trained run; C3-from-C1 is an evaluation attached to its C1 run",
        "inventory_observed": "2026-10-16 (training completion, checkpoint tensor validity)",
        "evaluations_accepted": "2026-10-17/18 (v2 evaluator, val split, seed 42)",
        "c3_from_c1_evaluated": "2026-10-18 (main tree 206c5de)",
        "inputs": {name: {"path": str(path.relative_to(ROOT)), "sha256": sha256(path)}
                   for name, path in INPUTS.items()},
        "groups": {
            group: {
                "runs": sum(1 for r in rows if r["group"] == group),
                "release_class": count(group, "release_class"),
                "accepted_by_eval_scope": dict(sorted(Counter(
                    r["eval_scope"] for r in rows if r["group"] == group and r.get("eval_scope")).items())),
                "checkpoint_replaced_by_fp32_retrain": sum(
                    1 for r in rows if r["group"] == group and r.get("checkpoint_replaced") == "true"),
            }
            for group in ("canonical_v2", "timestep")
        },
        "c3_from_c1": {
            "evaluations": len(c3),
            "variant": dict(sorted(Counter(r["variant"] for r in c3.values()).items())),
            "ci_low_above_one_third": dict(sorted(Counter(
                r["variant"] for r in c3.values() if float(r["primary_ci_low"]) > 1 / 3).items())),
        },
        "release_class_to_remaining_work": CLASSES,
    }
    return buffer.getvalue(), json.dumps(summary, indent=2, ensure_ascii=False) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="fail if the written outputs are stale")
    args = parser.parse_args()
    table, summary = build()
    if args.check:
        stale = [p for p, text in ((OUT_CSV, table), (OUT_JSON, summary))
                 if not p.is_file() or p.read_text(encoding="utf-8") != text]
        if stale:
            print("stale:", ", ".join(str(p.relative_to(ROOT)) for p in stale), file=sys.stderr)
            return 1
        print("release status is up to date")
        return 0
    OUT_CSV.write_text(table, encoding="utf-8")
    OUT_JSON.write_text(summary, encoding="utf-8")
    print(f"wrote {OUT_CSV.relative_to(ROOT)} and {OUT_JSON.relative_to(ROOT)}")
    print(json.dumps(json.loads(summary)["groups"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
