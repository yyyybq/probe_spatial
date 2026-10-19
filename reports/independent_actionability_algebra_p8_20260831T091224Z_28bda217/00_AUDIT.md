# Independent p8 Actionability + Algebra audit

+Study: `independent_actionability_algebra_p8_20260831T091224Z_28bda217`

+Status: `COMPLETE_VALID` for scope quarantine and raw-pose setup.

+This study is independent of the historical C-series.  It does not load old C1/C2 checkpoints, old target lineage, canonical result CSVs, or registry entries.  The historical raw-ID mismatch is recorded as the reason for this isolation, not as a reason to manufacture `n=0`.

+## Frozen pre-study identity

+- Git HEAD: `2075f3eb270cb0e8907c3927b6b079c64965b75a`
+- `git diff --binary` SHA256: `28bda217ce66b6dcdf2063cd0d4e7e6cdf7f95e08a3f8245ed91fcdd70ce9330`
+- Critical source SHA256: `artifacts/independent_actionability_algebra_p8_20260831T091224Z_28bda217/provenance/start_state.json`
+- `submit.sh`: size `1` bytes; SHA256 `01ba4719c80b6fe911b091a7c05124b64eeece964e09c058ef8f9805daca546b`

+## Scope

+- Models: DINO L0; F3R L24; V-JEPA2 L23; Wan L20/t749.
+- Prefix: 8 only.
+- Explicitly excluded: B2, old C checkpoint reuse, canonical registry/results edits, p12/p16/p24, sweeps, text goals, online RL, and backbone fine-tuning.
+