# Canonical-v2 legacy streaming reference overlay (r11)

This overlay is a zero-copy routing catalogue. It does **not** promote legacy
features into `SEEDED_V1`, and it does not claim a byte-level payload audit.
It permits a legacy feature reference only when canonical raw frame IDs and
the physical extractor identity agree and a fresh local or remote metadata
gate passes. Missing or ambiguous evidence fails closed to pending or
re-extraction.

## Frozen lineage

- Catalogue: `artifacts/windowfix_v2/reference_overlay_02e17c40b59a_r11`
- Route namespace: `FEAT_STREAMING_CANONICAL_V2_MIXED_REFERENCE_V1`
- Canonical parent source SHA-256:
  `ebc496eb0587cda372ccf29a7c5244ef454096151f73c8eed9511967ded1902d`
- Canonical universe SHA-256:
  `4771e2859c43889bd5fb75f5876bcf493758f4187336c9bd7eeca5432acbac9b`
- Catalogue manifest SHA-256:
  `31673374d65eb133352e3abcb7d6804e3df8dbedfc69a24063960cea2eca99ea`
- Remote request file SHA-256:
  `c45571dd64604f01fe5c12c44dacc1dd8da946bb455f685821e75befac57a99a`

The builder verifies the namespace lock, the physical parent p24 manifest,
and every p8/p12/p16/p24 derived manifest and `.meta.json`. Every derived meta
must name the same parent and universe, and its declared derived-manifest SHA
must equal the file bytes.

## Current mutually exclusive routes

| Status | Units | Meaning |
|---|---:|---|
| `REFERENCE_SOURCE_READ_ONLY` | 5,298 | Fresh local sidecar/header gate passed |
| `PENDING_REMOTE_GATE` | 212,098 | Exact semantic candidate; current AOSS evidence still required |
| `REEXTRACT_IDENTITY_MISMATCH` | 39,832 | Raw IDs match but physical identity does not |
| `REEXTRACT_RAW_ID_MISMATCH` | 178,132 | No selected legacy source has exact canonical raw IDs |

The universe contains 435,360 units. There are 217,396 semantic reuse
candidates and 217,964 baseline re-extraction units. Exact per-model,
per-prefix counts are in
`candidate_counts_by_model_prefix_status.csv` and
`universe_counts_by_model_prefix_status.csv` inside the catalogue.

## Remote metadata gate

`analyses/windowfix_v2/gate_aoss_streaming_references.py` reads the small
remote `prefix_index.npy` and sidecar, then reads only the 8-byte safetensors
header length and JSON header range. It checks the exact raw IDs, source file,
size, model, mode, prompt, resize, frame count, timestep, shape and dtype. It
does not read the tensor body. Each request has a 30-second fail-closed timeout;
backend errors and timeouts remain pending.

The offline, review-only 128-shard plan is in
`artifacts/windowfix_v2/aoss_reference_gate_plan_r11_hardened_r12_128shards_20260831`.
The separately SHA-bound SCO wrapper catalogue is in
`artifacts/windowfix_v2/aoss_reference_gate_sco_review_r11_hardened_r13_20260831`.
It contains 128 rendered commands and records `submissions=0` and
`remote_reads_performed=0`.  Each stdlib-only worker revalidates every local
input and its own binding before importing the AOSS-capable gate.

After reviewed shards have produced receipts, join them without overwriting
any frozen input:

```bash
/mnt/umm/users/yinbaiqiao/.conda/envs/probe-spatial/bin/python \
  analyses/windowfix_v2/finalize_aoss_streaming_gate.py \
  --catalog-dir artifacts/windowfix_v2/reference_overlay_02e17c40b59a_r11 \
  --receipt-root <reviewed-receipt-root> \
  --output-dir <new-immutable-finalization-directory>
```

The finalizer rejects duplicate or extra receipt IDs, changed request SHAs,
changed catalogue files, and pass receipts without all-true checks. Missing or
retriable receipts remain pending; definitive failures become complete
`REEXTRACT_UNVERIFIED` routes for the gap-worklist builder.

## Baseline extraction gaps

The current gap-only plan is
`artifacts/windowfix_v2/streaming_gap_worklists_ebc496eb0587_catalog_r11_20260831`.
It contains 10 worklists, 59,366 record entries and 217,964 layer-items. It
excludes all pending remote candidates. Failed remote gates can be added later
with `build_streaming_gap_worklists.py --failed-gate-routes ...`.
