# Reproduce

```bash
export MPLCONFIGDIR=/tmp/probe_spatial_mpl
/mnt/umm/users/yinbaiqiao/.conda/envs/probe-spatial/bin/python -m pytest -q tests/test_se3_protocol_v2.py tests/test_se3_latent_mvp.py
/mnt/umm/users/yinbaiqiao/.conda/envs/probe-spatial/bin/python -m analyses.se3_protocol_v2.paper_v1 --output-dir results/experimental/se3_observed_paper_v1_REPRO --seeds 42,43,44,45,46 --bootstrap-resamples 2000
```

The runner refuses to overwrite an existing top-level output directory. It reads only the paths and hashes listed in each child `provenance.json`.
