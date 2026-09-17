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
# DINOv2 is historical-only under protocol revision d_series_scope_r2_20260916.
WAN_T2V_DTYPE=bf16 $PY scripts/d_series/extract_wan.py --scene-root "$SCENES" --output-root "$FEATURES"
```

For every model, branch, and seed 0/1/2, train D1. V-JEPA2 needs separate isolated and native-target readouts. The weak linear spatial head is a capacity control and is not substituted for the main spatial head.

```bash
$PY scripts/d_series/run_d1.py --model vjepa2_vitl256 --reference isolated_target --branch spatial --seed 0 --scene-root "$SCENES" --feature-root "$FEATURES" --output-dir "$RUNS/d1_vjepa2_isolated_spatial_s0"
$PY scripts/d_series/run_d1.py --model vjepa2_vitl256 --reference native_target --branch spatial --seed 0 --scene-root "$SCENES" --feature-root "$FEATURES" --output-dir "$RUNS/d1_vjepa2_native_spatial_s0"
$PY scripts/d_series/run_d2.py --model vjepa2_vitl256 --seed 0 --scene-root "$SCENES" --feature-root "$FEATURES" --d1-checkpoint "$RUNS/d1_vjepa2_isolated_pooled_s0/checkpoint.pt" --output-dir "$RUNS/d2_vjepa2_s0"
$PY scripts/d_series/run_d3.py --branch spatial --seed 0 --scene-root "$SCENES" --feature-root "$FEATURES" --native-d1-checkpoint "$RUNS/d1_vjepa2_native_spatial_s0/checkpoint.pt" --output-dir "$RUNS/d3_vjepa2_spatial_s0"
```

Repeat the analogous D1/D2 commands for Wan and all seeds. DINOv2 files/logs are preserved but it is not active and cannot block Wan or V-JEPA2. No D3 command exists for Wan. Every runner audits cache receipts, trains both horizons jointly, selects on dev, writes its checkpoint, receipt, and per-sample CSV, and refuses an existing output directory.

Paper outputs are fail-closed:

```bash
$PY scripts/d_series/audit_protocol.py --feature-root "$FEATURES" --run-root "$RUNS"
MPLCONFIGDIR=/tmp/dseries_mpl $PY scripts/d_series/build_paper_outputs.py --run-root "$RUNS" --output-dir experiments/d_series_scannetpp_v1/paper_outputs
$PY scripts/d_series/render_qualitative.py --visual-index experiments/d_series_scannetpp_v1/paper_outputs/visual_index.csv --run-root "$RUNS" --feature-root "$FEATURES" --scene-root "$SCENES" --output-dir experiments/d_series_scannetpp_v1/paper_outputs/qualitative
```

The formal builder requires the active Wan+V-JEPA2 models/branches/conditions and seeds `{0,1,2}` on the exact manifest hash. Seed0 diagnostic tables are separately labelled and do not relax the formal gate. Missing records, incomplete receipts, or mismatched hashes are hard errors; historical pilot values cannot enter the tables.

The completed V-JEPA2 seed0 diagnostic can be rebuilt separately without relaxing the formal gate:

```bash
$PY scripts/d_series/build_seed0_diagnostic.py \
  --run-root experiments/d_series_scannetpp_v1/diagnostic_runs/dseries_vjepa2_seed0_20260916_h800_r1 \
  --output-dir /path/to/fresh_seed0_diagnostic_output
$PY scripts/d_series/build_seed0_visual_index.py \
  --run-root experiments/d_series_scannetpp_v1/diagnostic_runs/dseries_vjepa2_seed0_20260916_h800_r1 \
  --output /path/to/fresh_seed0_diagnostic_output/visual_index.csv
```

The preserved result is labelled `development-only / seed0 / diagnostic`; it contains 92 h2/h4 table rows and 12 deterministic best/median/worst visual selections. It is not a substitute for seeds 1/2 or Wan.

## Sharding and recovery

Feature extractors accept `--window-start` and `--max-windows`; each feature has an immutable `.sft` + `.json` pair with model revision, raw IDs, tensor contract, payload size, and SHA-256. A matching pair is a resume hit; a partial or mismatched pair aborts. Use disjoint window shards that share one fresh formal root.

`scripts/d_series/submit_feature_canaries_sco.sh` submits a bounded single-GPU V-JEPA2 canary. It passes independently when the real V2 read/overfit/identity receipt and native predictor/future-invariance receipt both pass. `submit_vjepa2_missing_features_sco.sh` then fills only the D-series roles absent from V2 and produces a one-time full coverage audit. Wan has its own independent path; neither model waits for DINOv2 or for the other model's canary. Formal training remains recoverable and fail-closed.
