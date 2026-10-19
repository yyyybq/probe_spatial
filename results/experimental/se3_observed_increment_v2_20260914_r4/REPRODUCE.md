# Reproduce

```bash
export MPLCONFIGDIR=/tmp/probe_spatial_mpl
/mnt/umm/users/yinbaiqiao/.conda/envs/probe-spatial/bin/python -m pytest -q tests/test_se3_protocol_v2.py tests/test_se3_latent_mvp.py
/mnt/umm/users/yinbaiqiao/.conda/envs/probe-spatial/bin/python -m analyses.se3_protocol_v2.incremental_v2 --output-dir <NEW_OUTPUT_DIR> --bootstrap-resamples 2000
```
