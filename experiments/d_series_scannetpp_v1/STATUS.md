# Execution status

The frozen dataset now has a complete real V-JEPA2 **development-only / three-seed** D1/D3 result. This is not the final Wan+V-JEPA2 paper table: V-JEPA2 D2 is running, Wan remains incomplete, and there is no held-out test split. This file is intentionally explicit so a diagnostic or submitted job cannot be mistaken for completed formal evidence.

Canary scheduler observation (2026-09-16 21:03:19 UTC): V-JEPA2-only canary `pt-404n3gnv` (`dseries_vjepa2_canary_20260916_r6`) is `SUCCEEDED`. It executed submit-side snapshot SHA256 `c2b75656a4966a469c24e0238e9d429437d1f6f5ea2a1980d326d819f75ea222` from a worker-local temporary directory at base commit `dbbbce70b7ea394998de6c8ac704b9e114ef4403`. Both V2 pooled and native predictor return codes are zero. r4 and r5 are preserved as failures with their exact first failure points. Machine-readable events are in `job_ledger.jsonl`.

The missing-role extraction job `pt-a7v0cmx6` completed all 2,721 window shards but its in-job audit invocation failed only because the frozen temporary root was absent from `PYTHONPATH`. The original failure receipt is preserved. Audit-only recovery job `pt-96mlsdaa` then ran from the same frozen source snapshot on one zoetrope H800 and succeeded without re-extraction. Its full audit is `COMPLETE_VALID`: 681 scenes, 2,721 windows, 5,442 samples, and exact role counts `prefix=2721`, `isolated_target=5442`, `native_target=5442`, `native_pred=5442`, with zero failures and five matching payload-hash probes.

Seed0 worker `pt-ca3z0nnl` completed its local frozen-source chain at 2026-09-17 00:37:08 UTC with `d1_failed=0`, `d3_ab_failed=0`, and `d3_cd_failed=0`. The 92-row diagnostic table and 12-example qualitative index are under `diagnostics/dseries_vjepa2_seed0_20260916_h800_r1/`; checkpoints, curves, receipts, and per-sample rows are under `diagnostic_runs/dseries_vjepa2_seed0_20260916_h800_r1/`.

V-JEPA2 seed1 job `pt-hfosgdj8` and seed2 job `pt-8ykywqfl` completed successfully at 2026-09-17 08:53 and 08:52 UTC on zoetrope with exactly one H800 each. Both have `failed=0`, eight complete checkpoints/receipts/metric files, and all four D1 cells plus D3 A/B and A/B/C/D for spatial and pooled branches. The fail-closed three-seed build under `diagnostics/dseries_vjepa2_d1_d3_three_seed_20260917_zoetrope_r1/` contains 276 per-seed rows, 92 across-seed rows, and 16 paired D3 comparison rows with 10,000-draw 69-scene cluster bootstrap intervals.

V-JEPA2 D2 is parallelized safely by seed. Jobs `pt-iilm18ac` (seed0), `pt-l9r28mzi` (seed1), and `pt-e8yjd5wh` (seed2) started on zoetrope at 2026-09-17 10:37--10:38 UTC with `HIGHEST` priority and one H800 each. Their output directories are disjoint and each cites its seed-matched isolated/pooled D1 checkpoint and the complete V-JEPA2 feature audit. They are `RUNNING` and are not counted as completed.

Seed0 scene-macro normalized-depth MAE highlights (h2 / h4):

| experiment | branch/reference | h2 | h4 |
|---|---|---:|---:|
| D1 target readout | isolated pooled | 0.16834 | 0.16515 |
| D1 target readout | isolated spatial | 0.18185 | 0.17838 |
| D1 target readout | native pooled | 0.19228 | 0.18192 |
| D1 target readout | native spatial | 0.18218 | 0.18293 |
| D1 prior | shared train-layout prior | 0.24063 | 0.23767 |
| D3 A/B/C/D | pooled | 0.19228 / 0.23507 / 0.23055 / 0.22721 | 0.18192 / 0.23613 / 0.22965 / 0.22822 |
| D3 A/B/C/D | spatial | 0.18218 / 0.24561 / 0.25642 / 0.22315 | 0.18293 / 0.24429 / 0.25985 / 0.22765 |

Completed gates:

- canonical manifest derivation and hash verification;
- historical scene-use audit (no defensible held-out test scenes);
- actual frame identity, SE(3), motion magnitude, and directed-overlap materialisation;
- spatial/pooled readout, weak-head, external/context-only predictor, and tokenwise adapter implementations;
- V-JEPA2 distinct-tubelet mapping and prefix-before-attention implementation;
- depth/pointmap metrics and scene-cluster paired bootstrap;
- CPU contract tests.
- independent V-JEPA2 V2 full coverage receipt: 12,358/12,358 exact units;
- D-series V2 reuse audit: 425/5,442 isolated pooled targets adapted, 0/2,721 exact contexts, native roles absent;
- real V-JEPA2 canary: V2 tensor read/overfit/evaluator identity plus native predictor future-invariance.
- full V-JEPA2 missing-role cache and one-time recovered audit, without re-extracting completed shards;
- full-train/dev V-JEPA2 seed0 depth-only D1 and D3 A/B/C/D for spatial and pooled branches;
- h2/h4 scene-macro MAE/AbsRel/Delta1, 10,000-draw paired scene bootstrap intervals, per-sample rows, and fixed-scale qualitative panels.
- full-train/dev V-JEPA2 seeds 1/2 depth-only D1 and D3 A/B/C/D for spatial and pooled branches;
- V-JEPA2 D1/D3 three-seed aggregation with across-seed variation and paired B-A, C-B, D-B, and D-prior intervals.

Remaining evidence gates:

- formal matched Wan caches, especially layer20/index749;
- completion of the three running single-GPU V-JEPA2 D2 jobs, plus the remaining active Wan D1/D2 matrix;
- formal three-seed aggregation, plots, paper section, and slides from real records.

Any table builder must fail closed while required records are absent. Historical pilot values must never be copied into the formal tables.
