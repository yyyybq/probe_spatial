# Goal-conditioned one-step actionability report

## Status

`BLOCKED_MISSING_ARTIFACT` (DINO smoke n=0; all model-method cells n=0). The smoke correctly stopped before loading a quarantined checkpoint. Consequently F3R, V-JEPA2, and Wan expansion was not started.

## Coverage accounting

- Native eligible n: DINO 0, F3R 0, V-JEPA2 0, Wan L20 0.
- Four-model common n: 0.
- Dropped n cannot be computed from an unfrozen candidate universe without pretending missing feature roots were audited; the gate-level drop reason is `BLOCKED_MISSING_EXACT_STREAMING_PREFIX_OR_TARGET` for DINO/F3R/V-JEPA2 and `NO_FIXED_L20_EXACT_C1_C2` for Wan.
- The manifest files contain a non-sample blocker record by design. Result files contain blocker rows with blank estimates, not zeros.

## Methods and baselines

GT pose oracle, endpoint-feature oracle, C1 predictor ranking, direct shallow ranking, random, action-only, current-only, goal-only, identical-score, and temporal-direction prior were all `NOT_RUN` after the prerequisite gate. No scene-bootstrap CI was computed because n=0.

## Required questions

No requested empirical comparison is identifiable: there is no valid answer for the best goal metric, shallow-probe representation, C1 decision utility, F3R geometry-vs-predictor split, DINO retrieval transfer, Wan utility over random/action-only, or temporal-direction advantage. Existing C1 R@1 values are not imported because their lineage is not valid for this pilot. The prior Wan construct audit remains only a weak-signal caveat and is not upgraded.

## What would unblock this pilot

A manifest-exact p8 anchor/endpoint/goal cache at each fixed layer, a common model-independent sample-ID set with two real endpoints per anchor, clean fixed-setting C1 checkpoints, and a GPU for scene-split direct-probe training. This run did not authorize extraction or retraining.
