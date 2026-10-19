# Feature acceptance and feature–label join report

Status: **COMPLETE_VALID**  
Audit date: 2026-09-16 UTC

## Exact feature coverage

All four registered settings have `requested = verified = 12,358`, with zero
missing/invalid payloads, duplicate destinations, fallback, raw-ID/contract
mismatch, non-finite vectors, or zero/constant vectors. The accepted receipts
are:

- DINO L0: `execution/full_dino_v1/coverage_receipt.json`
- F3R L24: `coverage/f3r_coverage_receipt.json`
- V-JEPA2 L23: `coverage/vjepa2_coverage_receipt.json`
- Wan L20/t749: `execution/full_wan_v1/coverage_receipt.json`

No accepted feature was re-extracted for this audit.

## Complete consumer join

The immutable request contains 9,700 isolated units and 2,658 ordered-context
units per model. Every downstream reference was resolved by exact actual raw
ID or the complete ordered eight-ID tuple. `requested_roles`, directory order,
tail index, filename similarity, and implicit manifest row order were not used
as lookup keys. For each model, all 25,871 references made by the frozen
Actionability, dynamics, and loop manifests resolved uniquely.

The audit independently recomputed every Actionability 9D action, pose-metric
utility, utility margin, goal distance, and best-candidate label from the
camera-to-world poses with rotation weight 0.5 and implicit `scene_scale=1`.
It also recomputed all four primitive actions and motion magnitudes for every
dynamics record, and verified that h1, h2, and h4 identify rollout steps 1, 2,
and 4 of that same ordered primitive sequence.

The immutable feature request registers isolated targets at h1, h2, and h4,
not h3. The probe therefore consumes all four actions but applies feature
supervision at steps 1, 2, and 4. It never uses a contextual target slice as an
isolated h3 substitute.

## Split and actual-input checks

- Actionability: 650 fit / 194 dev / 227 validation.
- Dynamics/algebra: 803 fit / 221 dev / 256 validation.
- Loop control: 824 records (412 paired loop/non-loop cases).
- Union scene boundary: 501 training scenes / 126 validation scenes / zero
  overlap.
- Validation remains validation: it participated in pilot development and is
  not described as a new blind test.

For each model, 33 deterministic checks spanning component, semantic role,
and fit/dev/validation actually opened the resolved source image, converted it
to RGB, applied the registered source-size preprocessing where applicable,
and recorded source-file and preprocessed-pixel SHA256 values. The resulting
132 checks, all manifest hashes, and per-model counts are stored in
`coverage/FEATURE_LABEL_JOIN_AUDIT.json`.

## Learned-input boundary

Learned Actionability scorers receive only registered frozen feature vectors
and the two 9D candidate actions. The recurrent dynamics probe receives only a
frozen current feature and the ordered 9D primitive actions, with isolated
features as supervision. GT poses, utilities, labels, raw IDs, scene IDs,
temporal direction, and difficulty metadata are not concatenated into learned
scorer inputs; they are used only for labels, lookup, stratification, or
evaluation.
