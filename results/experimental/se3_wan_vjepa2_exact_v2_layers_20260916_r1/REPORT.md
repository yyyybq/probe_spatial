# Wan / V-JEPA2 exact-v2 multilayer structural analysis

## Scope

This directory is independent of prior legacy-development results. It fixes p12, Wan diffusion condition, the common sample set, layers, seeds 42–46, head budget, validation rule, controls, and 2,000-resample paired scene bootstrap before evaluating endpoints. DINO is outside scope.

## Interface decision

The original C1 `ActionDynamicsProbe` is not a reusable state updater: it accepts a 5-D streaming context and emits an isolated-target-space vector. Feeding that vector back as streaming context would conflate roles despite matching channel dimensions.

The C2 `PathIntegrationProbe` has therefore been exposed as a closed hidden-state interface:

1. `initialize_state(streaming_context_A)`;
2. `update_state(state, encoded_SE3_action)`;
3. `decode_state(state)`.

This experiment trains a separate C2-shaped structural diagnostic head only on real A→B supervision. It does not alter or rename original C1. The same validation-selected checkpoint supplies its A→B calibration, sequential rollout, exact-composite direct rollout, and swapped-order rollout. No true B feature is injected into recursion and no composition/order/cycle loss is used.

## Outputs

- `inventory/`: live payload/sidecar receipts or isolated cell failure.
- `cells/<cell>/seeds/seed<seed>/checkpoints/`: four trained heads (`full`, `without_condition`, `action_only`, `commutative_additive`). Copy and matched shuffle require no separate checkpoint.
- `raw_predictions.npz`: one-step, sequential, direct, and swapped predictions for every baseline.
- `per_sample_metrics.csv`, `seed_metrics.csv`: auditable sample and seed results.
- `layer_main_table.csv`, `all_baselines_layer_table.csv`, `paired_scene_bootstrap.csv`, `order_vs_noncommutativity.csv`, and `layer_curves.*`: incrementally regenerated whenever a cell completes.
- `EXECUTION_STATUS.csv`: distinguishes completed, running, failed-isolated, and not-run cells.

Composition uses `encode_9d(T2 @ T1)`, never addition of 9-D actions or log vectors. The commutative additive sequential and swapped paths execute independently; their residual difference is recorded as floating-point validation rather than erased by tensor assignment.

Current numerical status is intentionally reported by `EXECUTION_STATUS.csv` and `FINDINGS.md`; unavailable cells are not represented as completed experiments.

## Completed execution

V-JEPA2 L0/L5/L11/L17/L23 each completed seeds 42--46. Every seed has four validation-selected checkpoints, raw predictions for all six controls, per-sample metrics, and per-seed metrics. This is 100 checkpoints and 25 seed-level prediction archives in total. The representative L11 cell passed first; the other cells were then trained and evaluated independently, with aggregation refreshed after each cell.

| Layer | A→B error | Sequential C error | Direct C error | Sequential--direct distance | Normalized order margin | Tie-aware accuracy |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.00581 | 0.00744 | 0.00745 | 0.000956 | -0.00150 | 0.4790 |
| 5 | 0.00202 | 0.00247 | 0.00252 | 0.000140 | 0.00147 | 0.5239 |
| 11 | 0.00251 | 0.00280 | 0.00280 | 0.0000847 | -0.000677 | 0.4851 |
| 17 | 0.00440 | 0.00493 | 0.00498 | 0.000155 | -0.000468 | 0.4826 |
| 23 | 0.12751 | 0.14245 | 0.14319 | 0.005018 | 0.000024 | 0.5138 |

All V-JEPA2 layers have a positive paired two-step gain over copy and a positive incremental gain over without-condition; the exact CIs are in `paired_scene_bootstrap.csv`. No layer has a normalized order-margin CI excluding zero. Full-minus-matched-shuffle and full-minus-commutative-additive order comparisons are likewise inconclusive. L11 is worse than action-only on this order comparison; the other full-minus-action-only CIs cross zero.

Wan L0/L5/L10/L15/L20/L25/L29 also completed seeds 42--46 after the exact scoped repair and full live gate. This adds 140 validation-selected checkpoints and 35 seed-level raw-prediction archives; combined with V-JEPA2, the directory now contains 240 checkpoints and 60 prediction archives.

| Layer | A→B error | Sequential C error | Direct C error | Sequential--direct distance | Normalized order margin | Tie-aware accuracy |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.00294 | 0.00415 | 0.00397 | 0.000357 | -0.00252 | 0.4692 |
| 5 | 0.00765 | 0.00966 | 0.00982 | 0.000638 | -0.000366 | 0.4931 |
| 10 | 0.00561 | 0.00687 | 0.00705 | 0.000430 | 0.000212 | 0.5101 |
| 15 | 0.00445 | 0.00531 | 0.00539 | 0.000252 | -0.000074 | 0.5000 |
| 20 | 0.00575 | 0.00728 | 0.00743 | 0.000483 | 0.000049 | 0.5011 |
| 25 | 0.00488 | 0.00646 | 0.00632 | 0.000505 | 0.00107 | 0.5025 |
| 29 | 0.00278 | 0.00361 | 0.00365 | 0.000222 | 0.000749 | 0.4996 |

All seven Wan layers have a positive paired two-step gain over copy with 95% scene-bootstrap CIs excluding zero. Six layers also have a positive incremental gain over without-condition; L0 is inconclusive. No Wan layer has a normalized order-margin CI excluding zero. L10 has a positive full-minus-action-only order contrast and L20 a positive full-minus-matched-shuffle contrast, while the other corresponding CIs cross zero; these isolated contrasts are not generalized to model-level order capability.

Across Wan layers, one-step error correlates descriptively with sequential endpoint error (`r=0.993`) and sequential--direct distance (`r=0.850`), but weakly with normalized order margin (`r=0.210`). V-JEPA2 shows the same qualitative separation (`r=1.000`, `0.989`, and `0.108`). These small-layer-count correlations are descriptive, not causal evidence.

The exact route repair, target smoke, final 10,913-path live gate, and job lineage are recorded in `AVAILABILITY.md`; the numerical source of truth remains the CSV/NPZ/checkpoint artifacts listed above.
