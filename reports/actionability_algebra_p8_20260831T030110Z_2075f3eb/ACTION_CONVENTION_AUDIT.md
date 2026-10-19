# Action convention audit

## Verdict

`COMPLETE_VALID`: all 7 numerical tests pass. This validates the geometry utility contract only; it does **not** validate any learned representation.

## Exact convention

- Repository extrinsics are world-to-camera.
- `encode_relative_pose(extr_from, extr_to)` maps camera-from coordinates to camera-to coordinates: `R_rel = R_to @ R_from.T`; `t_rel = t_to - R_rel @ t_from`.
- Rotation is the first two **rows** of `R_rel`, flattened. Translation is expressed in the destination-camera coordinates and divided by `scene_scale`.
- Decode orthonormalizes row 0/row 1 and obtains row 2 by their cross product.
- Composition is performed in 4×4 SE(3), never by adding 9D vectors: `T_0to2 = T_1to2 @ T_0to1`.
- The valid identity action is `[1,0,0, 0,1,0, 0,0,0]`, not nine zeros.
- C1 and C2 both receive actions constructed by the same `encode_relative_pose` call in the dataset. C1 receives anchor→target; C2 receives consecutive waypoint actions.

| Test | Max abs error | Tolerance | Status |
|---|---:|---:|---|
| identity_encoding_is_not_all_zero | 0 | 1e-07 | PASS |
| builder_decode_roundtrip_01 | 7.18245446452e-09 | 1e-06 | PASS |
| builder_decode_roundtrip_12 | 3.40753836259e-08 | 1e-06 | PASS |
| composition_order_T12_at_T01_equals_T02 | 2.58127316466e-08 | 2e-06 | PASS |
| compose_encode_decode_roundtrip | 2.22602330596e-08 | 2e-06 | PASS |
| inverse_encode_decode_roundtrip | 1.4216207378e-08 | 2e-06 | PASS |
| inverse_cycle_is_identity | 1.34304717903e-08 | 2e-06 | PASS |

Test source: `artifacts/actionability_algebra_p8_20260831T030110Z_2075f3eb/action_convention_tests.py`. Machine-readable results: `action_convention_test_results.csv/json`. Existing source: `vidfm3d/utils/spatial_diag.py` and `vidfm3d/data/components/inscene15k_dataset.py`.
