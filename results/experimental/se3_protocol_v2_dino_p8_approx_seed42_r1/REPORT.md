# SE(3) protocol-v2 endpoint audit

Status: **approx-grounded diagnostic only; not canonical; cross-model expansion blocked**.

## Frozen coordinate contract

- Raw `trajectories` are camera-to-world matrices.  This protocol uses world-to-camera
  `E = inv(c2w)` and `T_AB = E_B @ inv(E_A)`.
- Ordered endpoints are `E_12 = T2 @ T1 @ E_A` and
  `E_21 = T1 @ T2 @ E_A`.  All branch matrices use that convention.

## Exact endpoint audit

- Frozen static reconstruction assets: `0`.
- Compatible repository renderer entrypoints: `0`.
- Exact-render Gate: **FAIL**.
- Static-scene Gate: **FAIL**.

No frozen reconstruction+renderer is available locally.  Observed ScanNet++ RGB frames
are not renders at synthetic `T2@E_A` or `T1@T2@E_A` poses; the manifest therefore
records nearest observed images only as `approx_nearest_observed_rgb`, including each
endpoint's translation/rotation residual.  It makes no exact endpoint, static-scene,
or canonical grounded-noncommutativity claim.

## Geometry-only construction

- Train primitive examples: `1000` (required >= 1000).
- Held-out branch rows: `200` (target 200), from `69` scene-disjoint test scenes.
- Requested/selected transform families are in `branch_coverage.json`; selection never
  inspected model features.
- `branches.jsonl` SHA-256: `cffea19e3b8ddd640b75b99e629ab0d4af6600ba419f14e452bcec253d1acf85`.

## Gate decision

Feature extraction, DINO grounded order metrics, and representative cross-model cells
are intentionally **not run** because exact rendering, static certification, and
100%-coverage features on the common branch manifest are prerequisites.  The separate
SeeSE3 primitive-frame smoke may be run, but cannot cure this endpoint failure.
