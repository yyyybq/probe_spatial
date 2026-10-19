# Live availability audit

Final object-level audit: 2026-09-17 04:49--04:50 UTC, compute job `pt-1c16fxdq`. The audit made fresh AOSS existence checks only for the frozen p12 common set and seven registered Wan layers. It did not use a local dump as evidence and did not change the common set.

## V-JEPA2

L0/L5/L11/L17/L23 are complete on the immutable 436-row common set. Each cell passed payload/sidecar identity checks, materialization, seeds 42--46, and immediate composition/order evaluation. Exact-gap streaming supplies 160 rows per layer; 276 rows use freshly gated immutable reference payloads whose raw IDs were cross-checked against the exact route catalog. All 1,308 isolated A/B/C roles per layer are available.

## Wan final gate

The registered p12 cells are L0/L5/L10/L15/L20/L25/L29 at scheduler argument 749, literal `.` text condition, deterministic sample-local Gaussian noise, 480×832 resize, and repeat-tail padding of the 12 real frames to 81 frames.

| Cell | Complete streaming contexts | Complete target roles | Runnable |
|---|---:|---:|---:|
| wan_l0 | 436 / 436 | 1,308 / 1,308 | yes |
| wan_l5 | 436 / 436 | 1,308 / 1,308 | yes |
| wan_l10 | 436 / 436 | 1,308 / 1,308 | yes |
| wan_l15 | 436 / 436 | 1,308 / 1,308 | yes |
| wan_l20 | 436 / 436 | 1,308 / 1,308 | yes |
| wan_l25 | 436 / 436 | 1,308 / 1,308 | yes |
| wan_l29 | 436 / 436 | 1,308 / 1,308 | yes |

Across the seven cells, 10,913 unique payload, sidecar, and index paths were checked; all 10,913 were present. `inventory/wan_live_availability.json` is the machine-readable audit.

The two formerly missing streaming contexts were repaired by the immutable 2-entry/14-layer-item allow-list under `artifacts/tstep_probe_opt_v1/se3_wan_stream_repair_2x7_20260917/`. Job `pt-8x85qnmy` completed all 14 units and wrote completion receipt content SHA `3972a49f18c566bc19c90326eabd9c324b962272bea209bb677b6cbbb9a22834`. It authorized only:

- `90d8e25cee_w003_p12` at layers 0/5/10/15/20/25/29;
- `927aacd5d1_w000_p12` at layers 0/5/10/15/20/25/29.

Wan target routes use completed worklist SHA `ef69725f9ed8d7ca57d6eba5d0415812219136a5b3905af6559c115403705b42`, with `feature_t749_layer<L>.delta.sft` and `target_indices_layer<L>.npy` under `.canonical_v2_target_delta/ef69725f9ed8d7ca/`. Target payload smoke job `pt-cg3hsxr5` read the L15 safetensors and index, then verified unique finite A/B/C rows. The frozen 436-row common set and its SHA were unchanged.
