#!/usr/bin/env python3
"""Expand the frozen inventory into independent, exactly-once replacement jobs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pathlib
import shlex
from datetime import datetime, timezone


ROOT = pathlib.Path(__file__).resolve().parents[2]
PROTECTED_OLD_JOBS = {
    "pt-7zwspsz8", "pt-8tvccn5g", "pt-ckbz2bc2",
    "pt-gvbjbcsb", "pt-mnlx6ea7", "pt-ucbtgkss",
}
BACKEND = {"wan": "wan", "cogvideox": "cgx", "minimax_h3": "mmh3"}
TASK = {
    "streaming_depth": "sd", "streaming_mask": "sm", "view_consistency": "a2",
    "ego_belief": "b1", "ego_belief_v2": "b2", "action_dynamics": "c1",
    "path_integration": "c2", "counterfactual": "c3",
}


def sha(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", required=True, type=pathlib.Path)
    parser.add_argument("--output-dir", required=True, type=pathlib.Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--optimized-source-root", required=True, type=pathlib.Path)
    parser.add_argument("--output-namespace", required=True)
    parser.add_argument("--job-prefix", required=True)
    args = parser.parse_args()
    opt_root = args.optimized_source_root.resolve()
    if not (opt_root / "OPTIMIZED_SHA256SUMS").is_file():
        raise FileNotFoundError(f"missing immutable source manifest under {opt_root}")
    inv = json.loads(args.inventory.read_text())
    args.output_dir.mkdir(parents=True, exist_ok=False)
    jobs = {row["job_id"]: row for row in inv["jobs"]}
    records = []
    categories = {"already_complete": [], "protected_old_job": [], "replacement": []}
    for exp in inv["experiments"]:
        old_job = jobs[exp["old_job_id"]]
        evidence = exp["output_evidence"]
        complete_path = pathlib.Path(evidence["train_complete_path"])
        if evidence["train_complete"] or complete_path.is_file():
            categories["already_complete"].append(exp["unique_key"])
            continue
        if exp["old_job_id"] in PROTECTED_OLD_JOBS:
            categories["protected_old_job"].append(exp["unique_key"])
            continue
        if old_job["scheduler"]["state"] not in {
            "STARTING", "QUEUEING", "PENDING", "SUSPENDED"
        }:
            raise RuntimeError(
                f"refusing replacement for {exp['unique_key']} from state "
                f"{old_job['scheduler']['state']}"
            )
        campaign_dir = pathlib.Path(next(
            c["directory"] for c in inv["campaigns"]
            if c["campaign_id"] == exp["campaign_id"]
        ))
        groups = campaign_dir / "matrix/groups.tsv"
        audit = exp["feature_identity"]["strict_audit_path"]
        job_name = (
            f"{args.job_prefix}_{BACKEND[exp['backend']]}_t{exp['timestep']}_l{exp['layer']}"
            f"_p{exp['prefix']}_{TASK[exp['probe']]}"
        )
        command_env = {
            "MAIN_ROOT": str(ROOT),
            "GROUPS_TSV": str(groups),
            "GROUP_ID": exp["group_id"],
            "PROBE_TASK": exp["probe"],
            "RUN_ID": args.run_id,
            "OLD_CAMPAIGN_ID": exp["campaign_id"],
            "INSCENE_STRICT_AUDIT": audit,
        }
        command = "cd " + shlex.quote(str(opt_root)) + " && env " + " ".join(
            f"{key}={shlex.quote(value)}" for key, value in command_env.items()
        ) + " bash scripts/run_tstep_probe_single.sh"
        record = {
            "unique_key": exp["unique_key"],
            "old_job_id": exp["old_job_id"],
            "old_campaign_id": exp["campaign_id"],
            "group_id": exp["group_id"],
            "backend": exp["backend"],
            "timestep": exp["timestep"],
            "layer": exp["layer"],
            "prefix": exp["prefix"],
            "probe": exp["probe"],
            "seed": exp["seed"],
            "job_name": job_name,
            "command": command,
            "worker_spec": "N4lS.Iq.I80.1",
            "cpu": 14,
            "memory_gib": 240,
            "gpus": 1,
            "source_root": str(opt_root),
            "resolved_old_config": exp["resolved_config"],
            "manifest_identity": exp["manifest_identity"],
            "feature_identity": exp["feature_identity"],
            "new_output_dir": str(
                ROOT / "logs/inscene15k_streaming/runs" / args.output_namespace /
                args.run_id /
                f"tp_{exp['backend']}_t{exp['timestep']}_l{exp['layer']}_p{exp['prefix']}_{exp['probe']}"
            ),
        }
        records.append(record)
        categories["replacement"].append(exp["unique_key"])
    keys = [r["unique_key"] for r in records]
    if len(keys) != len(set(keys)):
        raise RuntimeError("duplicate independent experiment key in replacement plan")
    names = [r["job_name"] for r in records]
    if len(names) != len(set(names)):
        raise RuntimeError("duplicate scheduler job_name in replacement plan")
    plan = {
        "schema": "tstep-probe-independent-replacement-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": args.run_id,
        "inventory": str(args.inventory.resolve()),
        "inventory_sha256": sha(args.inventory),
        "optimized_source_root": str(opt_root),
        "optimized_source_manifest": str(opt_root / "OPTIMIZED_SHA256SUMS"),
        "output_namespace": args.output_namespace,
        "protected_old_jobs": sorted(PROTECTED_OLD_JOBS),
        "counts": {key: len(value) for key, value in categories.items()},
        "categories": categories,
        "replacements": records,
    }
    (args.output_dir / "replacement_plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    columns = [
        "unique_key", "old_job_id", "group_id", "backend", "timestep", "layer",
        "prefix", "probe", "seed", "job_name", "worker_spec", "new_output_dir", "command",
    ]
    with (args.output_dir / "replacement_plan.tsv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    print(json.dumps(plan["counts"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
