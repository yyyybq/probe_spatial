# Execution status

The dataset and protocol are frozen, but formal feature extraction/training/evaluation has not yet produced a complete result matrix. No main-table number exists in this directory yet. This file is intentionally explicit so submitted/running work cannot be mistaken for completed evidence.

Latest scheduler observation (2026-09-16 21:03:19 UTC): V-JEPA2-only canary `pt-404n3gnv` (`dseries_vjepa2_canary_20260916_r6`) is `SUCCEEDED`. It executed submit-side snapshot SHA256 `c2b75656a4966a469c24e0238e9d429437d1f6f5ea2a1980d326d819f75ea222` from a worker-local temporary directory at base commit `dbbbce70b7ea394998de6c8ac704b9e114ef4403`. Both V2 pooled and native predictor return codes are zero. r4 and r5 are preserved as failures with their exact first failure points. Machine-readable events are in `job_ledger.jsonl`.

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

Remaining evidence gates:

- full D-series V-JEPA2 missing-role cache fill and its one-time coverage audit;
- formal matched Wan caches, especially layer20/index749;
- full-train/dev seed0 V-JEPA2 D1/D3 diagnostic, then remaining seeds/formal matrix;
- per-sample aggregation, plots, qualitative panels, paper section, and slides from real records.

Any table builder must fail closed while required records are absent. Historical pilot values must never be copied into the formal tables.
