# Final Recovery Code Identity

Identity type: `RECOVERY_FINAL_SOURCE_SNAPSHOT`  
Not a formal experiment identity: environment gates remain blocked.

## Source identity

- Base Git commit: `2075f3eb270cb0e8907c3927b6b079c64965b75a`
- Branch: `main`
- Worktree: dirty; HEAD alone is not the identity.
- Source manifest: `reports/gate_recovery_20260827T051224Z_2075f3eb/FINAL_SOURCE_MANIFEST.csv`
- Source manifest SHA256: `ee640a94b09849ceaab20947e69e2db09977469462f4c8fc9aa28560a6568ce1`
- Source snapshot: `artifacts/gate_recovery_20260827T051224Z_2075f3eb/final_source_snapshot.tar`
- Snapshot SHA256: `1335b4b4af35b28749c937c4676ea6917bee7591314796fff9036dd0193b0476`
- File count: 739
- Payload bytes: 5,219,428

The source-only selection contains bounded `.py`, `.sh`, `.yaml`, `.yml`, `.json`, `.jsonl`, `.md`, `.txt`, and `.toml` code/configuration files plus this tag's four executable gate helpers. It excludes reports, result tables, logs, checkpoints, and large feature objects. All 34 restored tiny files are present.

## Recovery provenance

- Pre-restore backup tar SHA256: `6f85616d7577593eb698379997731aac4eee9797b3ea38baac3b7f75667de293`
- Recovery-source tar SHA256: `97a4e405c2431aa3abf3ba0d34fed05f03b0de89dcd32a1003ad43b7c7c6ed9e`
- 33 files: dangling tree `e7fc7f857a2dd27eed3626c64d18f083bd1c0d87`, corroborated by Cursor hashes.
- 1 file: Cursor Local History `6081fdd2/vIbd.sh`.
- Reconstructed files: 0.

## Critical hashes

- `features/run_inscene15k.py`: `ee95d1d4d2ba554a919d2668d7be555222dfd1a0ba440a5a83d7a67d1eba9486`
- `scripts/build_b2_strict_manifest.py`: `ddc33beb1c3c40ba378064ba1564684f75e8962307131469fab7c20895129d60`
- `vidfm3d/utils/spatial_diag.py`: `5f4845c2591fa378b386dae9a011d41887f494ad14c43a7ef2b7c30ad9419461`
- Split manifest: `e01a2194143405ed8f08613ec0bc672c7c237fa9bb21d2b8da441e7ad56bf086`
- Canonical C manifest: `772194f077c45cc85a942f17f46f50103125ae8f5b1c983c08881c1759607e0c`
- `gpu_aoss_gate.py`: `7da954703e58e08ca234fea5fc7bc71c16f2f62c79941199663c8fd7f06b1077`
- `make_smoke_split.py`: `51d815754c31090a60a2fabdca2d3b4743823baada94e16d1561c6f8340fc62a`
- `validate_strict_smoke.py`: `133c6f6b94e8fb35ac11048a5b027aa83a4ea481b169b300e79c3e5d7dae3e04`
- `run_gpu_aoss_gate_sco.sh`: `d1dec2b3a3ac16c0a3209e4034b1df51a5776d80a4cb209686367b95236966eb`

## Verification state

- 34/34 restored-file SHA checks: PASS.
- 25 shell syntax checks: PASS.
- 9 YAML parse checks: PASS.
- Core `py_compile`: PASS.
- Required unit tests: 27 passed.
- `git diff --check`: PASS.
- Repository substantive tiny-code scan: 0 remaining.
- GPU/AOSS/DataLoader/strict-B2 real gate: NOT RUN due scheduler HTTP 403.

Use the manifest SHA256 as the immutable recovery code hash. Do not mint `paper_validity_exec_*` from this checkpoint until the real gates pass. After the scheduler ACL is fixed and gates pass without source changes, this snapshot may be promoted into a new formal provenance record; if any source changes, freeze a new snapshot.
