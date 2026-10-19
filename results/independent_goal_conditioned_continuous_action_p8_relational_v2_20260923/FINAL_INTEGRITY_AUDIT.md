# Final Integrity Audit

Status: **COMPLETE_VALID**.

## Frozen scope

- 1,280 trajectories: 803 fit, 221 dev, 256 validation.
- 552 disjoint scenes: 354 fit, 90 dev, 108 validation; all pairwise scene
  intersections are empty.
- Input is exact current P8 plus exact isolated h4.  Target is observed a1.
- Validation retains its original name and was not used for configuration or
  checkpoint selection.

## Feature integrity

- DINO, Fast3R, and V-JEPA2 reuse accepted 12,358/12,358 independent stores.
- Wan L29 is 2,560/2,560 verified with zero missing, duplicate, fallback,
  identity/contract mismatch, nonfinite, or constant units.
- Wan payloads share one revision, snapshot-tree hash, and extraction-contract
  hash.  No incompatible mainline feature was mixed into the store.
- Feature access uses exact raw-ID identities.  Full ordered tuples are required
  for current P8; exact single raw IDs are required for isolated h4.

## Label and leakage integrity

- All a1 labels were recomputed from anchor and h1 poses; maximum absolute error
  is 2.981e-08 under a 1e-07 tolerance.
- The registered SE(3) decoder/application convention reproduces h1 for all
  examples.
- Pose, goal pose, action labels, future actions, frame indices, scene IDs, and
  trajectory direction are excluded from learned inputs.
- Normalization is fit-only; epoch selection is dev-only.

## Training and statistical integrity

- Four frozen representations, five learned variants, and five seeds produce
  100 accepted checkpoints.  All four zoetrope training jobs succeeded.
- The merged file contains 32,768 validation rows: 8,192 per model, including
  trained controls, the same-checkpoint shuffled-goal intervention, motion
  prior, and a1 reference.
- All registered metrics are finite.  The summary has 32 condition rows and the
  paired-effects table has 140 model/comparison/metric rows.
- Paired intervals use 5,000 scene-cluster bootstrap replicates; seed repeats
  are averaged within sample and are not counted as independent scenes.
- The fixed shuffled-goal mapping is different-scene and matched by fit-derived
  goal-distance and first-action-magnitude tertiles.  Its mapping hash is
  `a8f7d19ed75de7f0d4a5f9d9a3afe06dab56c6611101571435d6e28115222193`.

## Rebuild without retraining

From the repository root, use a fresh destination:

```bash
OUTPUT_DIR=/tmp/continuous_action_relational_v2_rebuild \
  bash scripts/build_independent_continuous_action_relational_v2_results.sh
```

The command reads the accepted per-model outputs and rebuilds the merged
per-sample file, summaries, strata, paired effects, intervention audit, tables,
paper text, and aggregate integrity receipt.
