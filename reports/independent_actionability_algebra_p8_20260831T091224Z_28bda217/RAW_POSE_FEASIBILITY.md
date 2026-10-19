# Raw-pose feasibility

+Status: `COMPLETE_VALID`

+All universes were built directly from observed ScanNet++ image/mask-backed raw frames and camera trajectories.  Distances use the project motion-normalized camera metric: translation plus `0.5 ×` rotation geodesic angle in radians.

+## A0 bidirectional progress

+- Raw candidates: 17126
+- Eligible after train-only thresholds: 8289
+- Frozen train / val pilot: 844 / 227
+- Train / val scenes: 289 / 77
+- A1 observed-local-branch: `NOT_RUN` — no branch graph is inferred from a single observed trajectory; no synthetic branch is substituted.

+## Clean dynamics / algebra

+- Raw primitive paths: 150353
+- Eligible primitive paths: 142993
+- Frozen train / val pilot: 1024 / 256
+- Real observed loop records: 824

+Thresholds, rejection reasons, split SHA, and uniqueness checks are in `manifests/manifest_summary.json`.
+