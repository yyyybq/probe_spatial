# Selection Leakage Audit

This audit distinguishes three different facts:

1. A dataset split can be named `val`.
2. Evaluation can happen after training.
3. The same validation metric may or may not have been used for checkpoint/layer/prefix/config/model selection.

Only item 3 is selection leakage. The current evidence does not justify treating every `eval_split=val` result as confirmed leakage.

## Checkpoint Selection

Status: `UNKNOWN` for most default A/B/C rows; `ANALYSIS_ONLY`/`UNKNOWN` for Wan/Cog rows depending on use.

Evidence:

- Registry rows record checkpoints where available, especially C rows from the consolidated C CSV.
- Wan/Cog historical artifacts contain checkpoint reconciliation labels such as exact/repaired/provisional, but those are not lifecycle states.
- The current registry accepts non-provisional Wan/Cog rows as `COMPLETE` and marks provisional rows `INCOMPLETE`.

Conclusion:

There is not yet a single machine-readable policy proving whether every checkpoint was selected without using validation metrics. This is an audit gap, not confirmed leakage.

## Layer Selection

Status: `ANALYSIS_ONLY` for layer-sweep studies; `UNKNOWN` if a single layer is reported as a final model score without a frozen rule.

Evidence:

- Wan/Cog rows are explicitly a layer sweep. They should be interpreted as layer analysis unless a downstream report selects one layer as final model score.
- `results/canonical/b2_results.csv` contains all `COMPLETE` formal Wan/Cog B2 layer rows, not a hand-labeled best row.

Conclusion:

Layer sweeps are valid as scientific objects of study. Leakage risk appears only if best layer from the same split is promoted to final model score without a held-out test or predeclared selection rule.

## Prefix Selection

Status: `ANALYSIS_ONLY` for prefix sweeps; `UNKNOWN` for final best-prefix claims.

Evidence:

- A/B/C canonical rows preserve all prefixes rather than storing `best_prefix` as state.
- C1/C2/C3 are 8 models x 4 prefixes. Prefix robustness should be reported across prefixes or selected by a frozen rule.

Conclusion:

Prefix-specific tables are not leakage by themselves. Best-prefix headline claims need a frozen selection rule or a held-out test confirmation.

## Probe Architecture / Hyperparameters

Status: `UNKNOWN`.

Evidence:

- Current registry records config paths where inferable.
- It does not yet encode LR/hidden dim/pooling/loss/normalization/decoder changes as separate searchable hyperparameter fields.

Conclusion:

No confirmed leakage is recorded, but the audit is incomplete for hyperparameter tuning. Future registry expansions should add hyperparameter columns or config hashes.

## Model Selection

Status: `UNKNOWN`.

Evidence:

- Current canonical A/C tables include all default models with available `COMPLETE` rows.
- New LingBot/MiniMax-H3 backends are not canonical because final metrics are missing, not because their performance was filtered out.

Conclusion:

No evidence currently confirms model-selection leakage. Reports should avoid dropping weak models unless status is explicitly `FAILED`, `INCOMPLETE`, `INVALID`, or outside the declared scope.

## Need For Frozen Held-out Test

Current conclusion: a frozen held-out test is recommended, but not because `eval_split=val` alone proves leakage.

Reason:

- Current results are useful validation/evaluation evidence.
- Several analyses involve layer/prefix/model comparisons.
- A final paper-level claim should freeze selection rules from validation and then evaluate once on a held-out test manifest, ideally over multiple seeds.
