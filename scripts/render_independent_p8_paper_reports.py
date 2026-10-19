#!/usr/bin/env python3
"""Render result-dependent paper prose and final integrity audit."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

MODELS = ("dino", "f3r", "vjepa2", "wan")
DISPLAY = {"dino": "DINO", "f3r": "F3R", "vjepa2": "V-JEPA2", "wan": "Wan"}


def interval(row, metric):
    sd = pd.to_numeric(pd.Series([row.get(f"{metric}_seed_sd")]), errors="coerce").iloc[0]
    spread = "NA" if pd.isna(sd) else f"{sd:.3f}"
    return f"{row[metric]:.3f} ± {spread} [{row[f'{metric}_ci_low']:.3f}, {row[f'{metric}_ci_high']:.3f}]"


def effect_text(row) -> str:
    relation = "favored the first method" if row.ci_low > 1e-12 else "favored the second method" if row.ci_high < -1e-12 else "was inconclusive"
    return f"{row.estimate:+.3f} (95% CI {row.ci_low:+.3f} to {row.ci_high:+.3f}; {relation})"


def verdict(row, reverse: bool = False) -> str:
    if reverse:
        return "supported" if row.ci_high < -1e-12 else "not supported" if row.ci_low > 1e-12 else "inconclusive"
    return "supported" if row.ci_low > 1e-12 else "not supported" if row.ci_high < -1e-12 else "inconclusive"


def find(frame: pd.DataFrame, **values):
    result = frame
    for key, value in values.items(): result = result[result[key] == value]
    if len(result) != 1: raise ValueError(f"expected one row for {values}, found {len(result)}")
    return result.iloc[0]


def write_new(path: Path, text: str) -> None:
    if path.exists(): raise FileExistsError(path)
    path.write_text(text.rstrip() + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--paper-dir", type=Path, required=True)
    parser.add_argument("--package-root", type=Path, required=True); args = parser.parse_args()
    tables = args.paper_dir / "tables"
    action = pd.read_csv(tables / "TABLE_1_ACTIONABILITY.csv", keep_default_na=False)
    ae = pd.read_csv(tables / "actionability_paired_effects.csv")
    dyn = pd.read_csv(tables / "dynamics_main.csv", keep_default_na=False)
    de = pd.read_csv(tables / "dynamics_paired_effects.csv")
    comp = pd.read_csv(tables / "algebra_composition.csv")
    cycles = pd.read_csv(tables / "algebra_identity_inverse.csv")
    loops = pd.read_csv(tables / "algebra_observed_loops.csv")
    alge = pd.read_csv(tables / "algebra_paired_effects.csv")

    lines = ["# Main Results", "", "All numbers below use the frozen validation set; it is not a blind test. Accuracy and regret are seed means with scene-cluster 95% CIs.", "",
             "| Model | Direct acc. | Goal+action/no-current acc. | Dynamics acc. | Endpoint reference acc. | Direct regret | Dynamics regret |", "|---|---:|---:|---:|---:|---:|---:|"]
    for model in MODELS:
        direct = find(action, model=model, variant="direct", transform="raw")
        no_current = find(action, model=model, variant="goal_action_no_current", transform="raw")
        dynamics = find(action, model=model, variant="clean_dynamics_selection", transform="raw")
        endpoint = find(action, model=model, variant="observed_endpoint_feature_reference", transform="raw")
        lines.append(f"| {DISPLAY[model]} | {interval(direct, 'correct')} | {interval(no_current, 'correct')} | {interval(dynamics, 'correct')} | {interval(endpoint, 'correct')} | {interval(direct, 'regret')} | {interval(dynamics, 'regret')} |")
    dino_random = find(ae, model="dino", effect="direct_minus_random", metric="correct")
    f3r_dyn = find(ae, model="f3r", effect="dynamics_minus_direct", metric="correct")
    vj_dyn = find(ae, model="vjepa2", effect="dynamics_minus_direct", metric="correct")
    wan_action = find(ae, model="wan", effect="direct_minus_action_only", metric="correct")
    wan_nogoal = find(ae, model="wan", effect="direct_minus_current_action_no_goal", metric="correct")
    current_effects = [find(ae, model=model, effect="direct_minus_goal_action_no_current", metric="correct")
                       for model in MODELS]
    lines += ["", "Pre-registered questions:", "",
              f"- DINO direct readout over random: **{verdict(dino_random)}** ({effect_text(dino_random)}).",
              f"- F3R direct over dynamics-based selection: **{verdict(f3r_dyn, reverse=True)}** ({effect_text(f3r_dyn)} for dynamics minus direct).",
              f"- V-JEPA2 dynamics-based decision advantage over direct: **{verdict(vj_dyn)}** ({effect_text(vj_dyn)}). Algebra evidence remains decomposed in Table 2.",
              f"- Wan direct utility beyond action-only and no-goal shortcuts: **{'supported' if wan_action.ci_low > 1e-12 and wan_nogoal.ci_low > 1e-12 else 'not jointly supported'}** (effects {effect_text(wan_action)} and {effect_text(wan_nogoal)}).",
              "", "Current-state contribution beyond goal and action alone:", ""]
    lines += [f"- {DISPLAY[model]} direct minus goal+action/no-current: **{verdict(effect)}** ({effect_text(effect)})."
              for model, effect in zip(MODELS, current_effects)]
    lines += ["", "The direct scorer does not significantly outperform the goal+action/no-current control for any representation. The experiment therefore supports goal-conditioned candidate-action selection, but does not establish an additional contribution from the current visual state. The endpoint column is privileged post-action reference information, not a theoretical upper bound. Structural diagnostics are intentionally not collapsed into an algebra pass score."]
    write_new(args.paper_dir / "MAIN_RESULTS.md", "\n".join(lines))

    prose = ["# Paper Results", "", "## Actionability", "",
             "We evaluated 227 queries from 77 validation scenes. The earlier pilot had already touched every validation scene, so these estimates characterize the frozen validation protocol rather than a fresh confirmatory test.", ""]
    for model in MODELS:
        direct = find(action, model=model, variant="direct", transform="raw")
        dynamics = find(action, model=model, variant="clean_dynamics_selection", transform="raw")
        endpoint = find(action, model=model, variant="observed_endpoint_feature_reference", transform="raw")
        random_eff = find(ae, model=model, effect="direct_minus_random", metric="correct")
        shortcut_eff = find(ae, model=model, effect="direct_minus_action_only", metric="correct")
        no_current_eff = find(ae, model=model, effect="direct_minus_goal_action_no_current", metric="correct")
        dyn_eff = find(ae, model=model, effect="dynamics_minus_direct", metric="correct")
        goal_eff = find(ae, model=model, effect="direct_minus_shuffled_goal", metric="correct")
        prose.append(f"**{DISPLAY[model]}.** Direct accuracy was {interval(direct, 'correct')} with regret {interval(direct, 'regret')}; dynamics selection reached {interval(dynamics, 'correct')}, and the observed-endpoint reference reached {interval(endpoint, 'correct')}. The paired direct-over-random accuracy effect was {effect_text(random_eff)}, direct-over-action-only was {effect_text(shortcut_eff)}, direct-over-goal+action/no-current was {effect_text(no_current_eff)}, dynamics-over-direct was {effect_text(dyn_eff)}, and the intact-over-shuffled-goal effect was {effect_text(goal_eff)}.")
        prose.append("")

    prose += ["Across all four representations, the direct scorer did not significantly outperform the goal+action/no-current control. Together with the large loss under goal shuffling, this supports goal-conditioned candidate-action selection while leaving the incremental contribution of the current visual state unresolved.", ""]

    prose += ["## Forward dynamics", ""]
    for model in MODELS:
        bits = []
        for horizon in (1, 2, 4):
            full = find(dyn, model=model, variant="full", horizon=horizon)
            gain = find(de, model=model, effect="full_gain_over_copy", metric="cosine_distance", horizon=horizon)
            visual = find(de, model=model, effect="full_gain_over_visual_only", metric="cosine_distance", horizon=horizon)
            bits.append(f"h{horizon} error {interval(full, 'cosine_distance')}, copy gain {effect_text(gain)}, visual-only gain {effect_text(visual)}")
        prose.append(f"**{DISPLAY[model]}.** " + "; ".join(bits) + ".")
        prose.append("")

    prose += ["## Action algebra diagnostics", "",
              "We report prediction correctness, action dependence and structural consistency separately; low cycle drift alone is not treated as evidence because exact copy is a degenerate zero-drift control.", ""]
    for model in MODELS:
        one = find(comp, model=model, variant="one_composed")
        inverse = find(cycles, model=model, variant="inverse_cycle")
        inv_wrong = find(alge, model=model, effect="inverse_gain_over_wrong_inverse", metric="decoded_to_anchor_cosine")
        inv_copy = find(alge, model=model, effect="inverse_gain_over_copy", metric="decoded_to_anchor_cosine")
        real = find(loops, model=model, variant="real_loop")
        nonloop = find(loops, model=model, variant="matched_non_loop")
        loop_eff = find(alge, model=model, effect="matched_non_loop_minus_real_loop", metric="prediction_to_actual_endpoint_cosine")
        prose.append(f"**{DISPLAY[model]}.** One-composed endpoint error was {interval(one, 'prediction_to_endpoint_cosine')}, consistency-to-four-primitives was {interval(one, 'consistency_to_four_primitive_cosine')}, and wrong-action separation was {interval(one, 'action_dependence_wrong_action_separation')}. Synthetic inverse decoded-to-anchor error was {interval(inverse, 'decoded_to_anchor_cosine')}; its gain over wrong inverse was {effect_text(inv_wrong)} and over exact copy was {effect_text(inv_copy)}. On observed paths, prediction-to-endpoint error was {interval(real, 'prediction_to_actual_endpoint_cosine')} for near-return loops and {interval(nonloop, 'prediction_to_actual_endpoint_cosine')} for matched non-loops, with paired difference {effect_text(loop_eff)}.")
        prose.append("")
    prose += ["These results concern frozen representations plus task-trained probes at the four registered settings. They do not identify a causal effect of pretraining objective or establish a complete world model."]
    write_new(args.paper_dir / "PAPER_RESULTS.md", "\n".join(prose))
    write_new(args.paper_dir / "LIMITATIONS.md", (args.package_root / "LIMITATIONS.md").read_text())

    checks = []
    request = args.package_root.parent.parent / "artifacts/independent_actionability_algebra_p8_20260831T091224Z_28bda217/manifests/feature_request_manifest_v2.jsonl"
    checks.append(("frozen request SHA256", hashlib.sha256(request.read_bytes()).hexdigest() == "fb1dbd2d71878edae7eae9034a16ee9809fc0c0cfb67352e0a301ad0edbed369"))
    receipt_paths = {
        "dino": args.package_root / "execution/full_dino_v1/coverage_receipt.json",
        "f3r": args.package_root / "coverage/f3r_coverage_receipt.json",
        "vjepa2": args.package_root / "coverage/vjepa2_coverage_receipt.json",
        "wan": args.package_root / "execution/full_wan_v1/coverage_receipt.json",
    }
    join_audit = json.loads((args.package_root / "coverage/FEATURE_LABEL_JOIN_AUDIT.json").read_text())
    checks.append(("feature-label join and actual-load audit", join_audit["status"] == "COMPLETE_VALID"))
    for model in MODELS:
        receipt = receipt_paths[model]
        value = json.loads(receipt.read_text())
        checks.append((f"{DISPLAY[model]} exact feature coverage 12,358/12,358", value["status"] == "COMPLETE_VALID" and value["requested"] == value["verified"] == 12358))
        integrity = json.loads((args.package_root / "raw_results" / model / "training_and_integrity.json").read_text())
        checks.append((f"{DISPLAY[model]} final evaluator complete", integrity["status"] == "COMPLETE_VALID"))
    for name in ("actionability_per_sample.csv", "dynamics_per_sample.csv", "algebra_per_sample.csv"):
        checks.append((f"nonempty {name}", (args.paper_dir / name).stat().st_size > 100))
    controls = pd.read_csv(tables / "integrity_controls.csv")
    checks.append(("candidate-swap equivariance (all direct scorers)",
                   int(controls.candidate_swap_failures_direct.sum()) == 0))
    checks.append(("three full-dynamics checkpoints per model",
                   bool((controls.full_dynamics_checkpoints == 3).all())))
    run_matrix = (args.package_root / "RUN_MATRIX.csv").read_text()
    checks.append(("RUN_MATRIX has no pending or running branch",
                   not any(token in run_matrix for token in ("RUNNING", "STARTING", "IN_PROGRESS", "PENDING"))))
    for name in ("FIGURE_1_ACTIONABILITY.pdf", "FIGURE_2_DYNAMICS_ALGEBRA.pdf", "FIGURE_3_DIFFICULTY_LOOPS.pdf"):
        checks.append((f"nonempty {name}", (args.paper_dir / "figures" / name).stat().st_size > 1000))
    status = "COMPLETE_VALID" if all(value for _, value in checks) else "FAILED_INTEGRITY"
    audit = ["# Final Integrity Audit", "", f"Status: **{status}**", "", "| Check | Result |", "|---|---|"]
    audit += [f"| {name} | {'PASS' if value else 'FAIL'} |" for name, value in checks]
    audit += ["", "Validation was used during pilot development and is not represented as a blind test. No old C-series checkpoint or canonical-v2 feature store is part of these final result inputs.",
              "", "Rebuild all aggregate tables, figures and result prose without retraining:", "",
              "```bash", "PAPER_OUTPUT_DIR=/tmp/independent_p8_paper_rebuild bash scripts/build_independent_p8_final_paper.sh", "```"]
    write_new(args.paper_dir / "FINAL_INTEGRITY_AUDIT.md", "\n".join(audit))
    if status != "COMPLETE_VALID": raise SystemExit(2)
    print(json.dumps({"status": status, "checks": len(checks)}, indent=2)); return 0


if __name__ == "__main__": raise SystemExit(main())
