# Execution status

The dataset and protocol are frozen, but formal feature extraction/training/evaluation has not yet produced a complete result matrix. No main-table number exists in this directory yet. This file is intentionally explicit so submitted/running work cannot be mistaken for completed evidence.

Latest scheduler observation (2026-09-16 16:37:30 UTC): GPU canary job `pt-l1563r8x` (`dseries_feature_canary_20260916_1642_r4`) is `STARTING`, with zero running worker replicas and no start time. It is submitted but not running or complete. Its submission directory already contains the nonempty base commit `95ebf6ce8efdb13ae3691d74c7f9c0deebbeb43d`, scoped patch, hashes, and a 1,487,197-byte source snapshot; the worker path no longer invokes Git or tar. The prior r3 job failed before any extractor log because worker-side Git provenance was unavailable. Machine-readable events are in `job_ledger.jsonl`.

Completed gates:

- canonical manifest derivation and hash verification;
- historical scene-use audit (no defensible held-out test scenes);
- actual frame identity, SE(3), motion magnitude, and directed-overlap materialisation;
- spatial/pooled readout, weak-head, external/context-only predictor, and tokenwise adapter implementations;
- V-JEPA2 distinct-tubelet mapping and prefix-before-attention implementation;
- depth/pointmap metrics and scene-cluster paired bootstrap;
- CPU contract tests.

Remaining evidence gates:

- formal matched caches for all three backbones, especially Wan layer20/index749;
- GPU future-invariance canary through the complete V-JEPA2 data/extraction path;
- train-set overfit canary, then three-seed D1/D2/D3 runs;
- per-sample aggregation, plots, qualitative panels, paper section, and slides from real records.

Any table builder must fail closed while required records are absent. Historical pilot values must never be copied into the formal tables.
