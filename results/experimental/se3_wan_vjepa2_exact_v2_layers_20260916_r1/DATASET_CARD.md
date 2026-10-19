# Exact canonical-v2 data card

This experiment consumes only the frozen `inscene_streaming_canonical_v2_obs28_h124` p12 manifest. The common manifest was frozen before any feature value or evaluation score was read.

- Training: 128 source-train scenes, one hash-selected tuple per scene.
- Validation: 32 disjoint source-train scenes, one hash-selected tuple per scene.
- Evaluation: all 276 canonical validation tuples from 69 scenes.
- State anchor A is p12 position 11; B/C are positions 12/13 in the 28-frame canonical window.
- Every row stores the 12 actual source frame IDs and A/B/C frame IDs in `data/common_manifest.jsonl`. The largest source position is strictly earlier than B, so the source never contains a future endpoint.
- Selection is shared across Wan/V-JEPA2 and all layers. A missing payload isolates that cell; the common set is never silently filtered.

## Feature roles and preprocessing

`context` is the causal streaming-prefix feature ending at A. `z_a`, `z_b`, and `z_c` are isolated-target decoder-space features. Isolated features are supervision/evaluation only and are never passed to the streaming context initializer.

- Wan: `Wan-AI/Wan2.1-T2V-1.3B-Diffusers`, transformer blocks 0/5/10/15/20/25/29, 1536 channels. Canonical condition is scheduler argument 749, literal `.` null-prompt surrogate, 480×832 resize, sample-local deterministic Gaussian noise, and repeat-tail padding of the 12 real ordered p12 frames to the 81-frame model input. Export tensors are spatial-temporal mean pooled for the small head.
- V-JEPA2: `facebook/vjepa2-vitl-fpc64-256`, encoder blocks 0/5/11/17/23, 1024 channels, `skip_predictor=True`. `AutoVideoProcessor` performs resize/crop/normalization to 256×256; the 12 real ordered p12 frames use the canonical repeat-tail path to the 64-frame input. Export tensors are spatial-temporal mean pooled.

The live representative sidecars in `inventory/representative_sidecars.json` confirm the tensor shapes and conditions: Wan L15 is `[21,30,52,1536]`; V-JEPA2 L11 is `[32,16,16,1024]`. Both bind the exact 12 raw frame positions through the canonical-list SHA-256. This is an identity/range provenance check, not a strict distribution-inclusion certificate.

## AOSS routing

- Streaming reference: `aoss://baiqiao/InsScene-15K/FEAT_STREAMING`
- Streaming exact-gap: `aoss://baiqiao/InsScene-15K/FEAT_STREAMING_CANONICAL_V2_GAP_DELTA_V1`
- Target reference: `aoss://baiqiao/InsScene-15K/FEAT_TARGET`
- Target exact-gap: `aoss://baiqiao/InsScene-15K/FEAT_TARGET_CANONICAL_V2_GAP_DELTA_V1`

All 436 Wan streaming rows and 1,308 Wan target roles route to exact-gap payloads. Per V-JEPA2 layer, 160 streaming rows route to exact-gap payloads, 276 to fresh-gated exact legacy references, and all 1,308 target roles to exact legacy rows. `data/route_plan.jsonl` is the cell/sample-level source of truth; materialization receipts record every URI, tensor shape, target row, and exact-position certification method.
