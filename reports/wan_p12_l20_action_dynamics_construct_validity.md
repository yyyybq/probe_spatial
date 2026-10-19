# Wan P12/L20 Action Dynamics Construct Validity

Status: `COMPLETE`

## Scope And Frozen Inputs

This audit is the paper-primary Wan action-dynamics validity result. It uses the frozen Prefix=12, layer-20 checkpoint selected before this audit, the exact P12 manifest, and the historical evaluator's frozen target/prediction inventory. Wan P8 is retained only as supplementary prefix-robustness evidence. No layer or prefix search, feature extraction, probe retraining, registry update, or canonical-result update was performed.

- Manifest: `manifests/c_retrain_20260722/inscene_c_h124_prefix12.jsonl` (`sha256=0ae3f4dbe26ddc2ba6b936b7a5bde5353a8d912da304287533cff6182d93851b`)
- Checkpoint: `logs/inscene15k_streaming/runs/c_retrain_safe_20260722_1435/c_retrain_safe_20260722_1435_action_dynamics_wan_p12_layer20/VideoProbe3D/6zf0wf3f/checkpoints/epoch=49-step=30600.ckpt` (`sha256=87523ab72c43a3527eaaf45370e6f070e29c99d0122079cb56e443254d243567`)
- Validation inventory: 276 sample-ID-paired targets from 69 scenes, four windows per scene.
- Actions: continuous 9-D relative poses; action strata are deterministic diagnostics, not native labels.
- Formal shuffled control: 100 fixed-seed, exact-marginal, 10-frequency-block scene-independent derangements.
- Wrong-action control: all other 275 observed P12 validation actions per sample.
- `eval_null_action` is an evaluation-time zero action, not a trained no-condition probe.

## Replay And Data-Flow Gates

- Execution endpoint: `ssh-root@10.119.27.217`; device: `NVIDIA H800`.
- Frozen correct prediction max absolute difference: `0`.
- Frozen action max absolute difference: `1.1920929e-07`.
- Frozen target max absolute difference: `1.90734863e-06`.
- Frozen last-state-copy max absolute difference: `1.90734863e-06`.
- Executable action-validity and paper-contract tests passed: `True`.
- Probe output changes with action; training passes only `input_feat` and `action` to the probe; metadata is not a probe input; candidate ordering is metric-invariant.

## Phase 1 Results

| Prefix | Layer | Correct R@1 | Shuffled R@1 | Eval-null R@1 | Last-state copy R@1 | Prefix copy R@1 | Correct-wrong cosine margin |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 12 | 20 | 0.1522 | 0.0995 | 0.0543 | 0.1196 | 0.0109 | 0.000864 |

- Correct minus shuffled R@1: `+0.0526`, scene-bootstrap 95% CI `[+0.0259, +0.0789]`, permutation p=`0.0099`.
- Correct minus last-state copy R@1: `+0.0326`, 95% CI `[-0.0109, +0.0761]`.
- Correct minus eval-null R@1: `+0.0978`, 95% CI `[+0.0543, +0.1377]`.
- Correct minus all-wrong cosine: `+0.000864`, 95% CI `[+0.000625, +0.001129]`.

## Predeclared Gate

| Overall | Correct > shuffled | Wrong margin | Beats copy | Class breadth | Scene breadth |
| --- | --- | --- | --- | --- | --- |
| FAIL | True | True | False | True (12/12) | True (59/69) |

Phase 1 failed, so the predeclared protocol stops before capacity-matched Phase 2 training.

## Representation Geometry Context

A read-only P12 geometry bridge reuses the historical 276-sample, 69-scene C1 common-set summary from the same Wan P12/L20 checkpoint. Wan has last-state-to-target raw cosine 0.9789 (temporal displacement 0.0211), so raw cosine objectives sit in a high-persistence regime. After target-mean centering, Wan copy cosine remains 0.3127 and full prediction cosine is 0.6708; this is not solely a global-mean artifact.

This changes neither the strict gate nor the conclusion. The full-minus-copy point estimate remains +0.0326 R@1, but its interval crosses zero. The most defensible mechanism label is therefore `HIGH_TEMPORAL_SMOOTHNESS + WEAK_UNRESOLVED_INCREMENTAL_ACTION_SIGNAL`, not either `NO_DYNAMICS` or established non-copy dynamics. DINO is the centered-copy/retrieval maximum in the historical comparison, whereas Wan is the raw temporal-persistence maximum. Details and explicit unavailable quantities are in `analyses/probe_validity/REPRESENTATION_GEOMETRY_BRIDGE.md`.

## Paper-Ready Result

At the predeclared Wan P12/L20 setting, correct action conditioning passed the formal-shuffle, exhaustive-wrong-action, action-class-breadth, and scene-breadth gates, but its R@1 advantage over last-state copy was not statistically reliable. Wan therefore exhibits a reproducible but weak action-sensitive signal; the result does not establish a specifically action-conditioned dynamics representation.

Required evidence labels:

- `METRIC_IMPLEMENTATION_VALID = true`
- `ACTION_INPUT_REACHES_PROBE = true`
- `CANDIDATE_LEAKAGE = NONE_FOUND`
- `PHASE1_GATE = WEAK_OR_NON_ACTION_SPECIFIC_SIGNAL`
- `PHASE2_EXECUTED = false`
- `FINAL_INTERPRETATION = WEAK_ACTION_SIGNAL`
- `PAPER_ROLE = PRIMARY_WAN_ACTION_DYNAMICS_RESULT`
- `WAN_P8_ROLE = SUPPLEMENTARY_PREFIX_ROBUSTNESS_ONLY`

## Artifacts And Safety

- `results/analysis/wan_p12_l20_action_validity/summary.json`
- `results/analysis/wan_p12_l20_action_validity/per_sample_results.csv`
- `results/analysis/wan_p12_l20_action_validity/per_scene_bootstrap.csv`
- `results/analysis/wan_p12_l20_action_validity/tables/`
- `results/analysis/wan_p12_l20_action_validity/figures/`
- Protected files byte-identical: `True`.
