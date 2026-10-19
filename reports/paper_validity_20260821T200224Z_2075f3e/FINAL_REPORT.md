# FINAL REPORT

Run tag: `paper_validity_20260821T200224Z_2075f3e`

## 1. Actual completion across the seven goals

1. **B2 strict-zero-hidden:** implementation and 4 synthetic invariance tests completed; real manifest, strict-eval, and strict-train blocked by unavailable data/AOSS and CUDA.
2. **C1/C2 controls:** matched-parameter query/visual/action-only heads and isolated-anchor output path implemented. Legacy copy/no-action/batch-roll results were re-aggregated. New retrained and matched-shuffle numeric results are blocked.
3. **B2 trivial baselines:** train-only direction-prior, no-update metadata, analytic past-only geometry, and query-only architecture are implemented/audited; real strict numbers are blocked.
4. **Common intersections:** real C1/C2 model intersections and cross-prefix legacy-paired manifests generated. All audited C sets have n=276 with zero drops. B2 strict intersection blocked. Legacy pairing caveat is explicit. Files are under this report directory because repository `manifests/` was permission-locked.
5. **Capacity:** parameter/information-access feasibility audit completed; GPU pilot NOT_RUN.
6. **Prefix AUC:** C1/C2 common-set AUC tables/curves generated with actual spacing. No best-prefix ranking is used. Strict B2 AUC is absent.
7. **Correlation/difficulty:** raw task correlation and C1/C2 difficulty curves generated from real artifacts. Fair gain correlation and B2 curves are unavailable and left empty/blocked.

## 2. Which results are trustworthy

- Existing **standard B2** is not strict and cannot answer the zero-hidden claim.
- There are no numeric **strict-eval** or **strict-train** rows.
- Existing C controls are **legacy evaluation interventions** or contextual copy, not the requested fair retrained baselines.
- Analytic dead reckoning is an **oracle** by definition and has no real result here.
- C numeric tables use a common intersection, but cross-prefix pairing uses reconstructed legacy IDs; native/common deltas are reported.

## 3. Do C1/C2 really use action?

The current evidence cannot establish this. Legacy full/copy/no-action/batch-roll values are in `c_baselines_summary.csv`, but visual-only, action-only and magnitude-matched shuffle checkpoints are absent. Full-vs-strongest-fair-baseline gain is therefore not reported.

## 4. Does B2 beat a simple object belief?

Unknown. Strict direction prior, no-update, query-only and full checkpoints were not evaluated. Existing standard B2 aggregates cannot be substituted because target residual pixels and target-derived centroid points were allowed.

## 5. Sample-set sensitivity

C native/common differences are in `native_vs_common_metrics.csv`. Because the reconstructed common set is usually the full 276-record legacy set, changes can be small; this does not validate B2, whose IDs are missing.

## 6. Prefix conclusion

Use `paper_prefix_auc_table.csv`, p24-p8 and full curves. The AUC table is restricted to configurations present at all four prefixes on the paired common set; best prefix is descriptive only.

## 7. Task redundancy

`task_correlation_raw.csv` reports Spearman, Kendall and paired n at model-level prefix-AUC and model×prefix descriptive granularity. With only up to eight models, non-independent prefix points, and missing strict/gain columns, correlations do not support causal or redundancy claims.

## 8. Unfinished items and minimum repair

- **Real strict manifests/results:** restore a working ScanNet++ root plus streaming extractor, run `build_b2_strict_manifest.py`, extract fixed-target suffix caches, then pass DINO p8 strict/query smoke gates before the 64 B2 jobs.
- **Fair C controls:** ensure target caches include isolated anchors, train visual-only/action-only with unique `autoresume=false` folders, then run global matched-bin intervention.
- **Canonical IDs:** rerun eval-only with the updated dataset/evaluator; do not rely on legacy scene-order pairing in the paper.
- **Capacity pilot:** run only DINO/F3R/V-JEPA2 p8 after CUDA/data are available.

No background job was launched. Status matrix: 160 `COMPLETE_WITH_CAVEAT`, 832 `BLOCKED_MISSING_ARTIFACT`.
