# Canonical-v2 legacy feature classifier

`classify_feature_reuse.py` is a read-only, fail-closed classifier. It emits
`REUSE_EXACT` only after comparing the canonical and legacy raw-frame IDs and
fully validating the local feature payload plus its sidecar (including a full
payload SHA256).

## Inputs

The feature-spec input may be JSON, JSONL, CSV, or TSV. One row describes one
deduplicated `(model, layer, timestep, prefix)` feature requirement:

```json
{
  "spec_id": "cv2_wan_t749_l20_p8",
  "legacy_root_ids": ["default8_wan_p8"],
  "prefix": 8,
  "vfm": "wan",
  "model_id": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
  "feature_layer": 20,
  "timestep": 749,
  "expected_shape": [21, 30, 52, 1536],
  "expected_dtype": "torch.float16",
  "feature_filename": "feature_t749_layer20.sft",
  "expected_mode": "streaming_prefix"
}
```

`model_id`, `feature_layer` (or `layer`), `expected_shape`, and
`expected_dtype` are mandatory evidence for an exact result. If they are not
known, matching records remain `UNVERIFIED`.

The root inventory is a table with one row per candidate cache. Both roots
must already resolve to the directory whose direct children are scene IDs:

```json
{
  "legacy_root_id": "default8_wan_p8",
  "index_scene_root": "/local/index-snapshot/wan_p8",
  "feature_scene_root": "/local/feature-cache/wan/scannetpp",
  "index_path_template": "{scene}/prefix_index.npy",
  "feature_path_template": "{scene}/window_{window_id:04d}/prefix_{tail:06d}/{feature_name}",
  "feature_name_template": "feature_t{timestep}_layer{layer}.sft"
}
```

Templates must render paths below their declared roots. Remote roots are not
probed and are classified `UNVERIFIED`, never `REEXTRACT_MISSING`.

## Run

```bash
python analyses/windowfix_v2/classify_feature_reuse.py \
  --specs /path/to/enriched-feature-specs.json \
  --legacy-roots /path/to/legacy-roots.jsonl \
  --manifest-dir artifacts/tstep_canonical_v2/manifests \
  --output-dir artifacts/windowfix_v2/reuse-audit
```

The output directory contains a summary JSON/CSV plus one JSONL worklist per
status. `--payload-hash-policy sidecar` is a faster discovery pass, but it can
never emit `REUSE_EXACT`; exact certification uses the default `full` policy.
`--require-all-exact` optionally makes any non-exact record return exit code 2
after writing all reports.

The classifier never copies files. A later materializer should consume only
the `reuse_exact.jsonl` worklist and should copy, not symlink, into the new
immutable canonical-v2 namespace.

## Relaxed legacy import (separate namespace)

When the goal is temporal/physical exact reuse rather than byte-identical
`_SEEDED_V1` regeneration, run with:

```bash
--payload-hash-policy legacy-import
```

This mode implements `LEGACY_EXACT_IMPORT_V1`. It allows a missing historical
payload SHA, RNG seed, or extractor provenance. It still reads the complete
payload and records a newly computed SHA. A valid historical SHA that disagrees
with the current bytes is rejected.

The spec must pin `prompt` and preprocessing in `expected_sidecar`, for example:

```json
{
  "prompt": ".",
  "resize": [480, 832],
  "frame_index_schema": "image_mask_intersection_v1"
}
```

Do not write these imports to a `_SEEDED_V1` root. Build a mixed plan with
`build_hybrid_reuse_plan.py`; its destinations must be
`FEAT_STREAMING_CANONICAL_V2_MIXED_V1` and
`FEAT_TARGET_CANONICAL_V2_MIXED_V1`.

## Fast zero-copy reference mode

For the fastest temporally correct reuse without payload hashing or copying:

```bash
--payload-hash-policy reference
```

This emits `REUSE_EXACT` under the separate
`LEGACY_EXACT_REFERENCE_V1` contract only when raw IDs and all physical
identity fields pass. It checks file existence/size, sidecar identity, tensor
shape/dtype, and the safetensors header. It does not read tensor data. A valid
SHA written in the old sidecar is retained as `declared_source_payload_sha256`
with `payload_sha256_verified=false`; missing SHA is allowed.

Reference results are explicitly not byte-level certification. Feed them to
`build_reference_overlay_plan.py`, never to the import/copy planner or a
`_SEEDED_V1` namespace.
