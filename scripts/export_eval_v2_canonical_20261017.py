#!/usr/bin/env python3
"""Export canonical-V2 v2-evaluator results into one provenance-bound table.

Reads every EVAL_COMPLETE.json / EVAL_FAILED.json under
logs/inscene15k_streaming/runs/main_v2_eval_v2_canonical_20261017_r1/<mode>/<key>/
plus the evaluator's val_summary.json, and writes

  results/canonical_v2_eval_v2/<mode>_metrics.csv
  results/canonical_v2_eval_v2/<mode>_metrics.json   (same rows + counts)

A row is ACCEPTED only if: the evaluator reports a v2 evaluator_version, the
primary metric is finite, no prediction was non-finite, at least one record
was scored, and the summary's SHA-256 still matches the receipt.  Everything
else is REJECTED with a reason.  Acceptance here is about evaluation validity;
it does not replace the training-identity verification in the inventory.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN_ID = "main_v2_eval_v2_canonical_20261017_r1"
RUNS = ROOT / "logs/inscene15k_streaming/runs" / RUN_ID
PLAN = ROOT / "artifacts/main_v2_probe_v2_816_20260917/eval_v2_canonical_20261017_r1/plan.json"
OUT = ROOT / "results/canonical_v2_eval_v2"
SECONDARY = {
    "streaming_depth": ["mean_Auc_30", "mean_Auc_5"],
    "streaming_mask": [],
    "view_consistency": ["overlap_mae", "pos_acc", "neg_acc"],
    "ego_belief_v2": ["mean_az_err_deg", "mean_el_err_deg", "mean_log_dist_err", "top1", "top3"],
    "ego_belief": ["mean_el_err_deg", "mean_log_dist_err"],
    "action_dynamics": ["global_R@5", "mean_rank", "mean_cos", "no_action_global_R@1", "shuffled_action_global_R@1"],
    "path_integration": ["global_R@5", "mean_rank", "mean_cos", "final_pose_error"],
    "c3_from_c1": ["intervention_chance", "intervention_validity_chance_adjusted", "exact_candidate_validity", "intervention_margin", "counterfactual_consistency",
                   "action_sensitivity_cosine", "target_sensitivity_cosine",
                   "target_to_prediction_sensitivity_pearson", "sensitivity_calibration_mae",
                   "n_valid_action_sets", "n_valid_candidates"],
}
FIELDS = ["unique_key", "model", "probe", "layer", "prefix", "timestep", "scope", "variant", "status", "reject_reason",
          "primary_metric", "primary_value", "primary_ci_low", "primary_ci_high", "secondary_json",
          "n_loader", "n_scored", "n_nonfinite_pred", "evaluator_version", "eval_pack", "training_code_tag",
          "checkpoint_path", "checkpoint_sha256", "summary_path", "summary_sha256"]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def finite(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=("canary", "production"), required=True)
    parser.add_argument("--revision", choices=("r1", "r2", "r3", "r4", "t1", "c3"), default="r1")
    args = parser.parse_args()
    global RUN_ID, RUNS, PLAN, OUT
    if args.revision == "c3":
        RUN_ID = "c3_from_c1_eval_20261018_c3"
        PLAN = ROOT / "artifacts/c3_from_c1_eval_20261018_c3/plan.json"
        OUT = ROOT / "results/c3_from_c1_eval/c3"
    elif args.revision.startswith("t"):
        RUN_ID = f"tstep_eval_v2_20261018_{args.revision}"
        PLAN = ROOT / f"artifacts/tstep_eval_v2_20261018_{args.revision}/plan.json"
        OUT = ROOT / "results/tstep_eval_v2" / args.revision
    else:
        RUN_ID = f"main_v2_eval_v2_canonical_20261017_{args.revision}"
        PLAN = ROOT / f"artifacts/main_v2_probe_v2_816_20260917/eval_v2_canonical_20261017_{args.revision}/plan.json"
        OUT = ROOT / "results/canonical_v2_eval_v2" / args.revision
    RUNS = ROOT / "logs/inscene15k_streaming/runs" / RUN_ID
    plan = json.loads(PLAN.read_text())["items"]
    rows = []
    for key, entry in sorted(plan.items()):
        item = json.loads(Path(entry["path"]).read_text())
        out = RUNS / args.mode / key
        receipt_path = out / "EVAL_COMPLETE.json"
        row = {f: "" for f in FIELDS}
        row.update({k: item[k] for k in ("unique_key", "model", "probe", "layer", "prefix", "scope", "primary_metric")})
        row["timestep"] = item.get("timestep", "")
        row["variant"] = item.get("c3_variant", "")
        row.update({"eval_pack": item["eval_pack"]["name"], "training_code_tag": item["training_code_tag"],
                    "checkpoint_path": item["checkpoint"]["path"]})
        if not receipt_path.is_file():
            row.update(status="NOT_EVALUATED" if not (out / "EVAL_FAILED.json").is_file() else "FAILED",
                       reject_reason="no EVAL_COMPLETE.json")
            rows.append(row)
            continue
        receipt = json.loads(receipt_path.read_text())
        summary_path = Path(receipt["summary_paths"][0])
        summary = json.loads(summary_path.read_text())
        value = summary.get(item["primary_metric"])
        # bootstrap_95ci is {metric: {"low": x, "high": y}}; not every evaluator
        # bootstraps every metric (A1 none, A2 none, C1/C2 only some).
        ci_entry = (summary.get("bootstrap_95ci") or {}).get(item["primary_metric"])
        # eval_diag stores {low, high}; its C3 bootstrap stores [low, high].
        ci = ([ci_entry.get("low"), ci_entry.get("high")] if isinstance(ci_entry, dict)
              else list(ci_entry) if isinstance(ci_entry, (list, tuple)) and len(ci_entry) == 2 else ["", ""])
        # Field names differ by evaluator/probe: eval_diag A2/B2/C1 n_nonfinite_pred,
        # eval_diag C2 n_nonfinite_pred_records, eval_a1_streaming n_nonfinite_prediction.
        n_nonfinite = next((summary[k] for k in ("n_nonfinite_pred", "n_nonfinite_pred_records",
                                                  "n_nonfinite_prediction") if k in summary), None)
        n_scored = next((summary[k] for k in ("n_scored", "n_valid_prediction", "n_valid_records",
                                              "n_valid_action_sets")
                         if k in summary), None)
        row.update({
            "primary_value": value, "primary_ci_low": ci[0] if isinstance(ci, list) else "",
            "primary_ci_high": ci[1] if isinstance(ci, list) else "",
            "secondary_json": json.dumps({k: summary.get(k) for k in SECONDARY[item["probe"]]}, sort_keys=True),
            "n_loader": summary.get("loader_n_samples", summary.get("n_samples")),
            "n_scored": n_scored, "n_nonfinite_pred": n_nonfinite,
            "evaluator_version": summary.get("evaluator_version"),
            "checkpoint_sha256": receipt["checkpoint"].get("sha256"),
            "summary_path": str(summary_path), "summary_sha256": sha256(summary_path),
        })
        reasons = []
        if "_v2_" not in str(summary.get("evaluator_version", "")):
            reasons.append("not a v2 evaluator")
        if not finite(value):
            reasons.append("primary metric not finite")
        if n_nonfinite is None:
            reasons.append("summary has no non-finite prediction count")
        elif n_nonfinite != 0:
            reasons.append(f"{n_nonfinite} non-finite predictions")
        if not n_scored:
            reasons.append("no scored records")
        if row["summary_sha256"] != receipt.get("summary_sha256"):
            reasons.append("summary changed after receipt")
        row.update(status="ACCEPTED" if not reasons else "REJECTED", reject_reason="; ".join(reasons))
        rows.append(row)
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / f"{args.mode}_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    counts = Counter(r["status"] for r in rows)
    (OUT / f"{args.mode}_metrics.json").write_text(
        json.dumps({"run_id": RUN_ID, "mode": args.mode, "counts": dict(counts), "rows": rows}, indent=1, sort_keys=True) + "\n",
        encoding="utf-8")
    print(json.dumps(dict(counts)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
