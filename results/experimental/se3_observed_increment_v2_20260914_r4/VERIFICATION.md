# Verification receipt

Status: **PASS**.

- `py_compile` passed for `analyses/se3_protocol_v2/incremental_v2.py`.
- 17 protocol/MVP tests passed.
- Runner SHA-256: `063b4602bdc8eeda7a1c788d408d0a55ffd8d1e984f5282955b1ec6defce54fe`.
- Confirmation-manifest SHA-256: `b2a687b569161ec73a88e8befd1d397ff29de6f67c3deb61b2089fa5018c39a2`.
- 577 frozen tuples; 317/79/100 train/validation/test scenes; zero legacy-scene or split overlap.
- Four cells each contain five raw-prediction archives and the expected seed/sample metric rows.
- All additive paths were independently evaluated and passed their recorded float64 roundoff bounds.
- The minimal lists contain 1,729 unique V-JEPA2 real-history contexts and 5,187 missing four-model feature rows.
- The paper table contains four rows with copy, without-condition, and all registered order-control effects; 32 paired scene-bootstrap comparisons are present.
