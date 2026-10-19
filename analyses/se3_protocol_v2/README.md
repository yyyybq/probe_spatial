# SE(3) protocol-v2

This package is a geometry-first successor to `se3_latent_mvp`.  It freezes the
repository's world-to-camera convention:

```text
E = inverse(raw ScanNet++ c2w trajectory)
T_AB = E_B @ inverse(E_A)
E_12 = T2 @ T1 @ E_A
E_21 = T1 @ T2 @ E_A
```

`build_protocol` writes a model-independent **optional dual-image** manifest
before any feature is selected.  It requires train-only one-step primitives and
scene-disjoint held-out branches, but its nearest-observed synthetic endpoints
are residual diagnostics only.  They never block or enter the primary observed
video protocol, retrieval, rollout, or cross-model runs.

The M3/M4 command is intentionally narrower: it uses existing C2 isolated DINO
features only for observed one-step pairs, trains its target normalization on
train scenes only, and never treats its R² as evidence for grounded dynamics.

## Reproduce the executed observed-video DINO protocol

```bash
ENV_PY=/mnt/umm/users/yinbaiqiao/.conda/envs/probe-spatial/bin/python
RUN_DIR=results/experimental/se3_protocol_v2_observed_dino_l0_seed42_repro
export MPLCONFIGDIR=/tmp/probe_spatial_mpl

$ENV_PY -m analyses.se3_protocol_v2.observed_order \
  --output-dir "$RUN_DIR"
```

This uses only real observed C2 tuples `(A,B,C)` and real `z_C` endpoints.
It trains one-step predictors and the one-step M4 pose adapter, then evaluates
the held-out order controls.  It refuses to overwrite an existing output.

## SeeSE3 replication sanity (separate Gate)

```bash
REPLICATION_DIR=results/experimental/seese3_replication_dino_preflight_repro

$ENV_PY -m analyses.se3_protocol_v2.seese3_replication \
  --output-dir "$REPLICATION_DIR" \
  --seeds 42,43,44
```

The command uses an observed-pair temporal split, but refuses to call a result
a replication if the input lacks an exported representative middle layer or
the required pair count.  The existing DINO C-series artifact exports only
layer 0, so its preflight is expected to stop rather than relabel the old
single-seed scene-disjoint smoke.

## Optional dual-image extension

If a future project independently supplies a frozen reconstruction and renderer,
it may use the stored `E_*` and `intrinsics_source` to create an additional
dual-image analysis.  It is deliberately out of scope here and is not a
precondition for `metrics.py` or the observed-video protocol.
