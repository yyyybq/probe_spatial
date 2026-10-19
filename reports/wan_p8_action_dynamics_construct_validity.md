# Wan p8 Action Dynamics Construct Validity

## Scope And Frozen Inputs

This audit uses the completed canonical-clean Wan p8 run, its six frozen checkpoints, the exact canonical manifest, and the same versioned streaming/target roots. It performs no feature extraction and does not modify registry or canonical result files.

- Manifest: `manifests/c_retrain_20260722/inscene_c_h124_prefix8.jsonl` (`sha256=772194f077c45cc85a942f17f46f50103125ae8f5b1c983c08881c1759607e0c`)
- Val candidate pool: 276 sample-ID-paired targets from 69 scenes, four samples per scene.
- Actions: continuous 9-D relative poses; there are no native discrete action labels.
- Formal shuffled control: 100 fixed-seed, exact-marginal, 10-frequency-block derangements; donors are never the same sample or scene.
- Wrong-action control: all other 275 observed val actions for every sample.
- `eval_null_action`: zero action applied only at evaluation; it is not a trained no-condition baseline.
- Rank in canonical summaries is zero-based. This report's `rank1` and MRR use conventional one-based rank.

## Execution Audit Trail

The audit failed closed three times before the final six-layer inventory was accepted. `pt-th75liy1` stopped all layers on a CPU-vs-CUDA target-pooling ULP gate; `pt-55bzlziu` preserved a valid layer-5 result but its parent failed after five children exceeded an overly tight batch-kernel max-abs gate; `pt-c5bqtpxi` preserved valid layers 0/15/25 but its parent failed when layers 10/29 changed near-tie ranks under batch 64. The final retry `pt-it5ygrcr` reproduced layers 10/29 with canonical batch 4 and succeeded. Final layer wrappers are rc=0 for all six layers; all failed-attempt logs are preserved under `results/analysis/wan_p8_action_validity/failed_attempts/`.

## Phase 0 Data-Flow Audit

All four executable contract tests passed. The probe output changes for a fixed state under different actions; the training step passes only `input_feat` and `action` into the probe; sample IDs, paths, target indices, and window IDs remain metadata; candidate reordering with remapped positives leaves retrieval metrics unchanged. Independent recomputation matched the frozen canonical R@1/R@5/cos/rank within `1e-6` for all six layers.

## Phase 1 Results

| Layer | Correct R@1 | Shuffled R@1 | Eval-null R@1 | Last-state copy R@1 | Prefix copy R@1 | Correct-wrong cosine margin |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.1123 | 0.0731 | 0.0362 | 0.1812 | 0.0181 | 0.0007 |
| 5 | 0.1739 | 0.1141 | 0.0543 | 0.1667 | 0.0145 | 0.0019 |
| 10 | 0.1957 | 0.1320 | 0.0326 | 0.2428 | 0.0145 | 0.0018 |
| 15 | 0.1812 | 0.1384 | 0.0109 | 0.2862 | 0.0109 | 0.0012 |
| 25 | 0.1812 | 0.1122 | 0.0507 | 0.2283 | 0.0217 | 0.0011 |
| 29 | 0.1739 | 0.1021 | 0.0652 | 0.1739 | 0.0109 | 0.0007 |

The action-conditioned checkpoints are action-sensitive and generally outperform zero-action and frequency-block permutations. However, persistence is a stronger retrieval baseline at layers 10 and 15: the final streaming-state copy exceeds the trained correct-action probe in R@1. Therefore the middle-layer peak cannot be attributed specifically to action-conditioned dynamics under the pre-declared gate.

## Gate

Each of layers 10 and 15 had to pass all five pre-declared checks: significant correct-over-shuffled R@1 by paired scene bootstrap and permutation test; wrong-action cosine-margin CI above zero; correct R@1 significantly above the last-state copy; positive signal in at least 8/12 deterministic action strata; and positive signal in at least 35/69 scenes with no scene contributing 10% of absolute signal.

| Layer | Overall | Correct > shuffled | Wrong margin | Beats copy | Class breadth | Scene breadth |
| ---: | --- | --- | --- | --- | --- | --- |
| 10 | FAIL | True | True | False | True (12/12) | True (67/69) |
| 15 | FAIL | True | True | False | True (12/12) | True (66/69) |

Because Phase 1 did not pass, the protocol requires stopping before capacity-matched Phase 2 training. No Phase 2 run was launched.

## Qualitative Audit

Twelve deterministic examples spanning action-specific successes, wrong-action failures, persistence-explained failures, and near-zero ambiguous cases are recorded in `results/analysis/wan_p8_action_validity/tables/qualitative_examples.csv`. Scene- and action-class-level tables and the wrong-action class confusion are in the same directory.

## Artifacts

- `results/analysis/wan_p8_action_validity/summary.json`
- `results/analysis/wan_p8_action_validity/per_sample_results.csv`
- `results/analysis/wan_p8_action_validity/per_scene_bootstrap.csv`
- `results/analysis/wan_p8_action_validity/tables/`
- `results/analysis/wan_p8_action_validity/figures/`

## Verification

All six intervention files pass shape/hash coverage: 100 x 276 permutation outcomes and exactly 75,900 finite wrong-action outcomes per layer. The combined validity and paper-contract suite passed 13 tests. `experiments/registry.csv` and every `results/canonical/*.csv` file retained its pre-audit SHA256; no canonical file was updated.

## Required Conclusion

```text
METRIC_IMPLEMENTATION_VALID: true
ACTION_INPUT_REACHES_PROBE: true
CANDIDATE_LEAKAGE: NONE_FOUND
PHASE1_GATE: WEAK_OR_NON_ACTION_SPECIFIC_SIGNAL
PHASE2_EXECUTED: false
CORRECT_VS_SHUFFLED: {"0": {"bootstrap_p_le_zero": 0.006299370062993701, "ci_high": 0.07376902173913041, "ci_low": 0.007826086956521736, "control": "shuffled_action", "direction": "correct_minus_control; positive means correct is better", "estimate": 0.03920289855072464, "layer": 0, "metric": "r1", "n_scenes": 69, "reps": 10000}, "10": {"bootstrap_p_le_zero": 9.999000099990002e-05, "ci_high": 0.09898641304347824, "ci_low": 0.0293106884057971, "control": "shuffled_action", "direction": "correct_minus_control; positive means correct is better", "estimate": 0.06369565217391304, "layer": 10, "metric": "r1", "n_scenes": 69, "reps": 10000}, "15": {"bootstrap_p_le_zero": 0.0044995500449955, "ci_high": 0.07634148550724637, "ci_low": 0.010434782608695656, "control": "shuffled_action", "direction": "correct_minus_control; positive means correct is better", "estimate": 0.0427536231884058, "layer": 15, "metric": "r1", "n_scenes": 69, "reps": 10000}, "25": {"bootstrap_p_le_zero": 9.999000099990002e-05, "ci_high": 0.1018478260869565, "ci_low": 0.03843840579710144, "control": "shuffled_action", "direction": "correct_minus_control; positive means correct is better", "estimate": 0.06891304347826088, "layer": 25, "metric": "r1", "n_scenes": 69, "reps": 10000}, "29": {"bootstrap_p_le_zero": 9.999000099990002e-05, "ci_high": 0.10420289855072462, "ci_low": 0.04097826086956523, "control": "shuffled_action", "direction": "correct_minus_control; positive means correct is better", "estimate": 0.07181159420289855, "layer": 29, "metric": "r1", "n_scenes": 69, "reps": 10000}, "5": {"bootstrap_p_le_zero": 0.00039996000399960006, "ci_high": 0.09510960144927535, "ci_low": 0.026774456521739126, "control": "shuffled_action", "direction": "correct_minus_control; positive means correct is better", "estimate": 0.059782608695652176, "layer": 5, "metric": "r1", "n_scenes": 69, "reps": 10000}}
CORRECT_VS_NO_CONDITION: NOT_RUN; eval_null_action controls={"0": {"bootstrap_p_le_zero": 0.0023997600239976003, "ci_high": 0.13043478260869565, "ci_low": 0.021739130434782608, "control": "eval_null_action", "direction": "correct_minus_control; positive means correct is better", "estimate": 0.07608695652173914, "layer": 0, "metric": "r1", "n_scenes": 69, "reps": 10000}, "10": {"bootstrap_p_le_zero": 9.999000099990002e-05, "ci_high": 0.21014492753623187, "ci_low": 0.11594202898550725, "control": "eval_null_action", "direction": "correct_minus_control; positive means correct is better", "estimate": 0.16304347826086957, "layer": 10, "metric": "r1", "n_scenes": 69, "reps": 10000}, "15": {"bootstrap_p_le_zero": 9.999000099990002e-05, "ci_high": 0.2210144927536232, "ci_low": 0.12318840579710146, "control": "eval_null_action", "direction": "correct_minus_control; positive means correct is better", "estimate": 0.17028985507246377, "layer": 15, "metric": "r1", "n_scenes": 69, "reps": 10000}, "25": {"bootstrap_p_le_zero": 9.999000099990002e-05, "ci_high": 0.17753623188405798, "ci_low": 0.08333333333333333, "control": "eval_null_action", "direction": "correct_minus_control; positive means correct is better", "estimate": 0.13043478260869565, "layer": 25, "metric": "r1", "n_scenes": 69, "reps": 10000}, "29": {"bootstrap_p_le_zero": 9.999000099990002e-05, "ci_high": 0.15579710144927536, "ci_low": 0.06159420289855073, "control": "eval_null_action", "direction": "correct_minus_control; positive means correct is better", "estimate": 0.10869565217391304, "layer": 29, "metric": "r1", "n_scenes": 69, "reps": 10000}, "5": {"bootstrap_p_le_zero": 9.999000099990002e-05, "ci_high": 0.17391304347826086, "ci_low": 0.06521739130434782, "control": "eval_null_action", "direction": "correct_minus_control; positive means correct is better", "estimate": 0.11956521739130435, "layer": 5, "metric": "r1", "n_scenes": 69, "reps": 10000}}
CORRECT_VS_COPY: {"0": {"bootstrap_p_le_zero": 0.9956004399560044, "ci_high": -0.018115942028985508, "ci_low": -0.11956521739130435, "control": "copy_last_streaming_state", "direction": "correct_minus_control; positive means correct is better", "estimate": -0.06884057971014493, "layer": 0, "metric": "r1", "n_scenes": 69, "reps": 10000}, "10": {"bootstrap_p_le_zero": 0.9488051194880512, "ci_high": 0.010869565217391304, "ci_low": -0.10869565217391304, "control": "copy_last_streaming_state", "direction": "correct_minus_control; positive means correct is better", "estimate": -0.04710144927536232, "layer": 10, "metric": "r1", "n_scenes": 69, "reps": 10000}, "15": {"bootstrap_p_le_zero": 0.9999000099990001, "ci_high": -0.05434782608695652, "ci_low": -0.15942028985507245, "control": "copy_last_streaming_state", "direction": "correct_minus_control; positive means correct is better", "estimate": -0.10507246376811594, "layer": 15, "metric": "r1", "n_scenes": 69, "reps": 10000}, "25": {"bootstrap_p_le_zero": 0.9526047395260474, "ci_high": 0.010869565217391304, "ci_low": -0.10507246376811594, "control": "copy_last_streaming_state", "direction": "correct_minus_control; positive means correct is better", "estimate": -0.04710144927536232, "layer": 25, "metric": "r1", "n_scenes": 69, "reps": 10000}, "29": {"bootstrap_p_le_zero": 0.5223477652234777, "ci_high": 0.057971014492753624, "ci_low": -0.057971014492753624, "control": "copy_last_streaming_state", "direction": "correct_minus_control; positive means correct is better", "estimate": 0.0, "layer": 29, "metric": "r1", "n_scenes": 69, "reps": 10000}, "5": {"bootstrap_p_le_zero": 0.43025697430256976, "ci_high": 0.06521739130434782, "ci_low": -0.050724637681159424, "control": "copy_last_streaming_state", "direction": "correct_minus_control; positive means correct is better", "estimate": 0.007246376811594203, "layer": 5, "metric": "r1", "n_scenes": 69, "reps": 10000}}
BEST_LAYER_AFTER_CONTROLS: 29
MIDDLE_LAYER_PEAK_SURVIVES: false
FINAL_INTERPRETATION: WEAK_ACTION_SIGNAL
```

The evidence supports a real but weak action signal. It does not support the stronger claim that the layer 10-15 peak is a valid, specifically action-conditioned dynamics construct, because a same-domain persistence copy explains more retrieval performance at those layers.
