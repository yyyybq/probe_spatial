# Verification receipt

Status: **PASS**.

Commands:

```bash
MPLCONFIGDIR=/tmp/probe_spatial_mpl /mnt/umm/users/yinbaiqiao/.conda/envs/probe-spatial/bin/python -m py_compile analyses/se3_protocol_v2/incremental_v2.py
MPLCONFIGDIR=/tmp/probe_spatial_mpl /mnt/umm/users/yinbaiqiao/.conda/envs/probe-spatial/bin/python -m pytest -q tests/test_se3_protocol_v2.py tests/test_se3_latent_mvp.py
```

Result: 17 tests passed.

Read-only output audit:

- runner SHA-256: `3c16ad40dfb5b0094c206188e23b57fed4b2665a623846477eaf41b875836e4c`;
- confirmation manifest SHA-256: `b2a687b569161ec73a88e8befd1d397ff29de6f67c3deb61b2089fa5018c39a2`;
- 577 manifest rows; 317/79/100 train/validation/test scenes; zero legacy-scene or split overlap;
- four cells each contain five raw-prediction archives, 70 per-seed metric rows, and the expected per-sample rows;
- 1,729 unique V-JEPA2 real-history contexts, each eight ordered frames ending at the declared state with no future endpoint;
- 5,187 exact missing confirmation feature rows (1,729 each for F3R, V-JEPA2, and CogVideoX);
- all additive paths were independently evaluated and passed the recorded float64 roundoff bound;
- four main-table rows and 32 paired scene-bootstrap comparisons are present.
