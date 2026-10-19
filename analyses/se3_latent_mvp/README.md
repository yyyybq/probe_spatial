# SE(3) latent-structure MVP

This is an experimental, non-canonical analysis. It consumes the frozen,
coverage-complete DINO layer-0 / prefix-8 validation dumps already registered
by the C-series experiments. It never changes the experiment registry,
canonical tables, feature caches, or existing run directories.

The three isolated target states in each C2 record are used as `(A,B,C)`.
Only the one-step pairs `A->B` and `B->C` are training examples. Composition,
cycle, action-order, and rollout objectives are never added to training.

The canonical validation scenes are deterministically re-split by scene into
experimental train/validation/test sets. Consequently, these metrics are an
MVP diagnostic and must not be presented as canonical held-out benchmark
numbers.

Run:

```bash
PY=/mnt/umm/users/yinbaiqiao/.conda/envs/probe-spatial/bin/python
MPLCONFIGDIR=/tmp/probe_spatial_mpl "$PY" -m analyses.se3_latent_mvp.run_mvp \
  --config analyses/se3_latent_mvp/config.json \
  --output-dir results/experimental/se3_latent_mvp_reproduction_seed42

MPLCONFIGDIR=/tmp/probe_spatial_mpl "$PY" -m pytest -q \
  tests/test_se3_latent_mvp.py
```

The runner refuses to overwrite an existing output directory. Use a fresh
`--output-dir` for every reproduction or extension run.
