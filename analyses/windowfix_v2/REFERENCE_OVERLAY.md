# Fast canonical-v2 legacy reference overlay

`LEGACY_EXACT_REFERENCE_V1` is the fast path for reusing old features in place.
It avoids both full payload hashing and feature copying. It certifies that the
feature route has the correct temporal frames and physical extraction identity;
it does not certify every stored tensor byte.

An old streaming payload is referenced only when all of these are known and
equal to the canonical requirement:

- ordered raw frame IDs;
- VFM, model ID, layer, and timestep;
- prompt, frame-index schema, and resize/preprocessing identity;
- sidecar file name, file size, tensor shape, and dtype;
- safetensors header shape, dtype, offsets, and current source existence.

The old RNG seed, extractor commit/provenance, and payload SHA may be missing.
If a syntactically valid old SHA exists it is copied into the route as an
unverified declaration; it is not recomputed and must never be presented as a
verified checksum.

## Outputs and isolation

Exact items become `REFERENCE_SOURCE_READ_ONLY` rows in
`reference_routes.jsonl`. Mismatched, missing, and unverified items become
re-extraction rows. Logical route namespaces are:

- `FEAT_STREAMING_CANONICAL_V2_MIXED_REFERENCE_V1`
- `FEAT_TARGET_CANONICAL_V2_MIXED_REFERENCE_V1`

These names are metadata routing namespaces, not copied feature directories.
The source legacy paths remain read-only, and `_SEEDED_V1` is never written.

The current output is a routing catalog and dry-run audit. Before probing, the
dataset loader must explicitly consume the audited route catalog (or an
equivalent immutable lookup layer); merely pointing its existing single-root
configuration at the metadata directory will not resolve source payloads.

## Commands

```bash
python3 analyses/windowfix_v2/classify_feature_reuse.py \
  --specs /path/to/enriched_specs.json \
  --legacy-roots /path/to/legacy_roots.json \
  --manifest-dir /path/to/canonical_manifests \
  --payload-hash-policy reference \
  --output-dir artifacts/windowfix_v2/reference_classification

python3 analyses/windowfix_v2/build_reference_overlay_plan.py \
  --classification-dir artifacts/windowfix_v2/reference_classification \
  --protocol protocols/inscene_streaming_canonical_v2.json \
  --streaming-route-root /route/FEAT_STREAMING_CANONICAL_V2_MIXED_REFERENCE_V1 \
  --target-route-root /route/FEAT_TARGET_CANONICAL_V2_MIXED_REFERENCE_V1 \
  --output-dir artifacts/windowfix_v2/reference_overlay_plan

python3 scripts/windowfix_v2/dry_run_reference_overlay.py \
  --plan-dir artifacts/windowfix_v2/reference_overlay_plan
```

All three commands are read-only with respect to feature payloads. The planner
and audit scripts do not submit SCO or start probing.
