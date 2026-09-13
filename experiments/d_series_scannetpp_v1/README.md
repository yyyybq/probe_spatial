# D-series paper-ready experiment

This directory is the only formal namespace for `d_series_scannetpp_v1`. It is currently a frozen **development-only** protocol with feature/training evidence pending; see `STATUS.md` before citing results.

## Reproduction

Set paths explicitly; never point these commands at a legacy feature root.

```bash
PY=/mnt/umm/users/yinbaiqiao/.conda/envs/probe-spatial/bin/python
SCENES=/mnt/umm/users/yinbaiqiao/InsScene-15K/data/processed_scannetpp_v2/processed_scannetpp_v2
FEATURES=/path/to/d_series_scannetpp_v1_features
RUNS=/path/to/d_series_scannetpp_v1_runs

$PY scripts/d_series/build_manifest.py
$PY scripts/d_series/extract_vjepa2.py --scene-root "$SCENES" --output-root "$FEATURES"
$PY scripts/d_series/extract_dinov2.py --scene-root "$SCENES" --output-root "$FEATURES"
WAN_T2V_DTYPE=bf16 $PY scripts/d_series/extract_wan.py --scene-root "$SCENES" --output-root "$FEATURES"
```

For every model, branch, and seed 0/1/2, train D1. V-JEPA2 needs separate isolated and native-target readouts. The weak linear spatial head is a capacity control and is not substituted for the main spatial head.

```bash
$PY scripts/d_series/run_d1.py --model vjepa2_vitl256 --reference isolated_target --branch spatial --seed 0 --scene-root "$SCENES" --feature-root "$FEATURES" --output-dir "$RUNS/d1_vjepa2_isolated_spatial_s0"
$PY scripts/d_series/run_d1.py --model vjepa2_vitl256 --reference native_target --branch spatial --seed 0 --scene-root "$SCENES" --feature-root "$FEATURES" --output-dir "$RUNS/d1_vjepa2_native_spatial_s0"
$PY scripts/d_series/run_d2.py --model vjepa2_vitl256 --seed 0 --scene-root "$SCENES" --feature-root "$FEATURES" --d1-checkpoint "$RUNS/d1_vjepa2_isolated_pooled_s0/checkpoint.pt" --output-dir "$RUNS/d2_vjepa2_s0"
$PY scripts/d_series/run_d3.py --branch spatial --seed 0 --scene-root "$SCENES" --feature-root "$FEATURES" --native-d1-checkpoint "$RUNS/d1_vjepa2_native_spatial_s0/checkpoint.pt" --output-dir "$RUNS/d3_vjepa2_spatial_s0"
```

Repeat the analogous D1/D2 commands for Wan and DINOv2 and all seeds. No D3 command exists for those models. Every runner audits all cache receipts and hashes before training, trains both horizons jointly, selects on dev, writes its checkpoint, receipt, and per-sample CSV, and refuses an existing output directory.

Paper outputs are fail-closed:

```bash
$PY scripts/d_series/audit_protocol.py --feature-root "$FEATURES" --run-root "$RUNS"
MPLCONFIGDIR=/tmp/dseries_mpl $PY scripts/d_series/build_paper_outputs.py --run-root "$RUNS" --output-dir experiments/d_series_scannetpp_v1/paper_outputs
$PY scripts/d_series/render_qualitative.py --visual-index experiments/d_series_scannetpp_v1/paper_outputs/visual_index.csv --run-root "$RUNS" --feature-root "$FEATURES" --scene-root "$SCENES" --output-dir experiments/d_series_scannetpp_v1/paper_outputs/qualitative
```

The builder requires all three models/branches/conditions and seeds `{0,1,2}` on the exact manifest hash. Missing records, incomplete receipts, or mismatched hashes are hard errors; historical pilot values cannot enter the tables.

## Sharding and recovery

Feature extractors accept `--window-start` and `--max-windows`; each feature has an immutable `.sft` + `.json` pair with model revision, raw IDs, tensor contract, payload size, and SHA-256. A matching pair is a resume hit; a partial or mismatched pair aborts. Use disjoint window shards that share one fresh formal root.

`scripts/d_series/submit_feature_canaries_sco.sh` submits one bounded four-GPU allocation running three isolated extractor workers. A submitted job is not complete until its run directory contains `status.json` with all three return codes zero and each extractor audit reports the requested window count. Only then may `submit_full_features_sco.sh` launch the three model-specific eight-way feature jobs. After all feature audits pass, `submit_training_stage_sco.sh` launches the recoverable `d1` stage; its independently resumable `d2d3` stage is hard-gated on all required D1 receipts and checkpoints. The training orchestrator schedules at most one process per listed GPU and refuses partial output directories.
