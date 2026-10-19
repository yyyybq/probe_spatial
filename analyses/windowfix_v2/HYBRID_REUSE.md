# Canonical-v2 hybrid reuse

The hybrid workflow is an overlay on the frozen canonical-v2 temporal
protocol. It does not change the protocol file or relabel old tensors as
`_SEEDED_V1`.

1. Enrich each legacy feature spec with exact `model_id`, layer, timestep,
   tensor shape/dtype, expected prompt, and preprocessing identity.
2. Run `classify_feature_reuse.py --payload-hash-policy legacy-import` against
   canonical-v2 manifests. Only exact raw-ID and physical-identity matches enter
   `reuse_exact.jsonl`; the classifier rereads each accepted payload to compute
   its current SHA256.
3. Run `build_hybrid_reuse_plan.py`. Exact streaming files become
   `COPY_WHOLE_FILE`; mismatches, missing files, and unverified candidates become
   re-extraction records. Aggregate target caches use row-level import records
   and therefore require a target-specific auditor plus a merge-capable target
   writer rather than a whole-file copy. The current streaming classifier does
   not manufacture target-row records by itself.
4. Run `dry_run_hybrid_materialization.py`. The default pass checks metadata,
   safetensors headers, target-row bounds, worklist hashes, and destination
   collisions. Add `--verify-source-sha` for the final full-read gate.

No script in this workflow currently copies large payloads or submits SCO. That
is intentional: the exact reuse/re-extract ledger can be reviewed before any
external mutation.

For an even faster, zero-copy path, use the separately named
`LEGACY_EXACT_REFERENCE_V1` workflow documented in `REFERENCE_OVERLAY.md`.
That mode does not hash payload bytes and must not be described as byte-level
certification.

Example planning commands:

```bash
python3 analyses/windowfix_v2/classify_feature_reuse.py \
  --specs /path/to/enriched_specs.json \
  --legacy-roots /path/to/legacy_roots.json \
  --manifest-dir /path/to/canonical_manifests \
  --payload-hash-policy legacy-import \
  --output-dir artifacts/windowfix_v2/legacy_exact_classification

python3 analyses/windowfix_v2/build_hybrid_reuse_plan.py \
  --classification-dir artifacts/windowfix_v2/legacy_exact_classification \
  --protocol protocols/inscene_streaming_canonical_v2.json \
  --destination-streaming-root /dataset/FEAT_STREAMING_CANONICAL_V2_MIXED_V1 \
  --destination-target-root /dataset/FEAT_TARGET_CANONICAL_V2_MIXED_V1 \
  --output-dir artifacts/windowfix_v2/hybrid_reuse_plan

python3 scripts/windowfix_v2/dry_run_hybrid_materialization.py \
  --plan-dir artifacts/windowfix_v2/hybrid_reuse_plan
```
