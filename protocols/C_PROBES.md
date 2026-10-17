# C Probes Protocol

Source of truth for current C1/C2/C3 definitions. Code evidence is in
`vidfm3d/eval_diag.py`. C3 is now an evaluation protocol over a frozen C1
one-step predictor; it is not a separately trained probe. Historical trained-C3
results remain in `results/canonical/c3_results.csv` for provenance only and
must be labelled `legacy_trained_c3` when cited.

## Shared Setup

C probes use the shared streaming-prefix protocol over ScanNet++ streaming features. Inputs are prefix/context features and target-isolated future features. The current canonical registry uses `train_split=train`, `eval_split=val`, seed `42`, and the C retrain manifests under `manifests/c_retrain_20260722/`.

Historical canonical C rows cover 8 models x 4 prefixes for each formerly
trained C probe: Aether, CogVideoX, DINO, F3R, V-JEPA2, Wan, Qwen2.5-VL, and
VideoMAE. New runs train C1 (and C2 where requested); C3 reuses the matched C1
checkpoint at evaluation time.

## C1: Action-conditioned Prediction

Scientific question: given a prefix feature and a relative camera action, can the probe predict the isolated future target feature better than shortcut controls?

Actual input and target:

- Input: `input_feat` from the streaming prefix.
- Action: 9D relative camera action from the batch.
- Target: pooled `target_feat`, computed as the spatial mean of the isolated target feature.
- Probe call: `probe(input_feat, action)`.

Metrics and controls:

- True-action retrieval: `global_R@1`, `global_R@5`, `mean_cos`, `mean_rank`.
- No-action baseline: `probe(input_feat, zeros_like(action))`.
- Shuffled-action baseline: `probe(input_feat, rolled_action)`.
- Last-observation/copy baseline: pooled last prefix observation.
- Canonical derived metrics:
  - `action_gain_vs_no_action = global_R@1 - no_action_R@1`
  - `action_gain_vs_shuffle = global_R@1 - shuffled_action_R@1`
  - `gain_over_copy = global_R@1 - last_observation_R@1`

Interpretation rule:

Absolute C1 `global_R@1` is not sufficient evidence for action reasoning. A row is stronger evidence only if true-action performance beats no-action, shuffled-action, and copy baselines under the same manifest/prefix/model/layer. The current audit found that only a minority of C1 rows beat all three R@1 controls, so C1 should be reported with baseline-normalized metrics.

## C2: Path Integration

Scientific question: can the probe recurrently roll forward through an action sequence and recover waypoint target features/poses?

Actual input and target:

- Input: `input_feat_seq`.
- Action sequence: `path_actions`.
- Horizons: `action_horizons`.
- Target: pooled `target_feat_seq`.
- Pose target: `target_extrinsics`.
- Probe call: `probe(input_feat_seq, path_actions)`.

Metrics:

- Aggregate retrieval: `global_R@1`, `global_R@5`, `mean_cos`, `mean_rank`.
- Pose metrics: `final_pose_error`, `drift_rate`, `mean_step_pose_error`, `loop_closure_error`.
- Horizon metrics: `pose_error_h1`, `pose_error_h2`, `pose_error_h4` when present.

Canonical result structure:

- `results/canonical/c2_results.csv` contains one aggregate `horizon=all` row per experiment and separate `horizon=1`, `horizon=2`, and `horizon=4` rows when the corresponding horizon-specific metric exists.

Open analysis:

The current C2 canonical table has horizon-specific pose errors, but action-shuffle or trajectory-control C2 baselines are not yet canonicalized. Do not claim C2 shortcut controls are solved unless those controls are added to the registry and canonical tables.

## C3: C1 Action-Sensitivity Evaluation (No C3 Training)

Scientific question: with the representation, anchor state, and trained C1
readout fixed, does changing the candidate action change the one-step prediction
in the way required by the matched candidate target?

Required eval-set unit:

- One `action_set_id` identifies a single frozen anchor/context.
- The set contains at least two candidate actions and one matched isolated
  target feature per action.
- Optional `action_equivalence_group` labels actions that should be equivalent
  under the eval-set definition.
- Every candidate is evaluated independently with the C1 call
  `c1_probe(shared_context, candidate_action)`.
- No candidate target, candidate ID, equivalence label, or other branch result
  is an input to C1.

The preferred batched keys are `input_feat`, `candidate_actions`,
`candidate_target_feat`, `candidate_valid`, `action_set_id`, and optional
`action_equivalence_group`. The evaluator also accepts the historical data-only
names `input_feat_seq`, `counterfactual_actions`, `target_feat_seq`, and
`counterfactual_valid`, but still requires an `action_dynamics` checkpoint.

Metrics:

- `intervention_validity`: within-set matching accuracy; when equivalence labels
  exist, retrieving any target in the action's equivalence class is correct.
- `exact_candidate_validity`: stricter diagonal-only matching, retained as a
  diagnostic when equivalent candidates have near-identical targets.
- `intervention_margin`: best same-equivalence-class cosine minus the strongest
  different-class cosine (or diagonal minus off-diagonal without equivalence
  labels).
- `counterfactual_consistency`: mean matched prediction-target cosine. The name
  is retained for table compatibility; it is causal only when the eval targets
  come from restored and executed environment branches.
- `action_sensitivity_cosine`: mean pairwise cosine distance between predictions
  under different candidate actions from the same anchor.
- `target_sensitivity_cosine`: corresponding mean pairwise target distance.
- `target_to_prediction_sensitivity_pearson` and
  `sensitivity_calibration_mae`: whether predicted changes track target changes,
  rather than merely changing by an arbitrary amount.
- When equivalence labels exist: `equivalent_action_prediction_distance`,
  `distinct_action_prediction_distance`, and `equivalence_separation_gap`.

Interpretation rule:

C3 is a held-out capability analysis of C1, not an additional learnable
capacity. Report it together with the C1 checkpoint identity and C1 shortcut
controls. High output change alone is not success: it must align with the
matched target change. For equivalent actions, lower within-class prediction
distance is desirable; for distinct actions, matching accuracy/margin and
between-class separation are desirable.

Do not call same-trajectory h1/h2/h4 endpoints counterfactual branches. They may
be used as a multi-action sensitivity diagnostic, but causal language requires
an eval set produced by restoring the exact anchor state and executing each
candidate action. The old `CounterfactualProbe` head, old counterfactual configs,
and old C3 tables are compatibility artifacts, not the current protocol.

Run with a C1 experiment/checkpoint and the evaluation-only switch:

```bash
python vidfm3d/eval_diag.py \
  experiment=inscene15k_streaming/action_dynamics_wan_h124_v1 \
  ckpt_path=/path/to/c1.ckpt \
  +evaluation_protocol=c3_action_sensitivity \
  +eval_split=val train=false test=false
```

The ordinary mixed-horizon C1 loader groups rows by `manifest_sample_id`. A
dedicated alternative-action eval dataset should instead emit an explicit
`action_set_id` and the preferred candidate batch keys above.

For a frozen `.pt` artifact, store either a list of action-set dictionaries or
`{"records": [...]}` and use the checked loader directly:

```bash
python vidfm3d/eval_diag.py \
  experiment=inscene15k_streaming/action_dynamics_wan_h124_v1 \
  ckpt_path=/path/to/c1.ckpt \
  +evaluation_protocol=c3_action_sensitivity \
  "data.data_module.validation_datasets=[\"C1ActionSetDataset(path='/path/to/action_sets.pt',split='val')\"]" \
  data.data_module.batch_size_per_device_val=4 \
  data.data_module.num_workers_val=0 \
  +eval_split=val train=false test=false
```

Each artifact record contains unbatched `input_feat (T,H,W,C)`,
`candidate_actions (K,9)`, `candidate_target_feat (K,H,W,C)` or `(K,C)`,
`candidate_valid (K)`, `candidate_ids (K)`, and the lineage strings
`action_set_id`, `manifest_sample_id`, and `scene_path`. Optional
`action_equivalence_group (K)` must use integer class IDs. The loader rejects
duplicate sets/candidates, fewer than two valid candidates, shape/channel
mismatches, and non-finite tensors before inference.

Protocol-specific outputs are written as
`eval/<split>_c3_action_sensitivity_predictions.pt` and
`eval/<split>_c3_action_sensitivity_summary.json`, so running C3 analysis cannot
overwrite the ordinary C1 evaluation dump.

For the frozen V1 multi-model sweep, run the storage preflight before any GPU
submission. The sweep gate requires the exact historical V1 feature namespaces
to be present and deliberately rejects Canonical V2 namespaces, including names
that end in `SEEDED_V1`:

```bash
python scripts/c3_action_sensitivity_v1.py preflight \
  --manifest /path/to/sweep_manifest.csv
python scripts/c3_action_sensitivity_v1.py gate \
  --manifest /path/to/sweep_manifest.csv
```
