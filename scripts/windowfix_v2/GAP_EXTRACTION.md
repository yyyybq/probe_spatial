# Canonical-v2 gap-only extraction

`prepare_sco_gap_jobs.py` renders SCO review artifacts only for gaps emitted by
a SHA-bound streaming-route or target-row audit. It cannot consume the 1096
inventory directly and has no "extract everything missing" fallback.

The bound worklist contract is `CANONICAL_V2_GAP_WORKLIST_V1`:

- streaming entries allow exactly one `scene/window_id/prefix/layers` set;
- target entries allow exactly one `scene/frame_index/layers` set;
- the file SHA, canonical JSON content SHA, protocol file, canonical source,
  model, timestep, prompt contract, resize, and model frame count are bound;
- an absent record or layer is never work.

Streaming gaps are written below
`FEAT_STREAMING_CANONICAL_V2_GAP_DELTA_V1`. Each feature sidecar binds the
worklist hashes, canonical-source SHA, record key, requested layers, and raw-ID
sequence SHA. Target gaps are written as per-layer `*.delta.sft` plus
`target_indices_layer<L>.npy` under
`FEAT_TARGET_CANONICAL_V2_GAP_DELTA_V1/.../.canonical_v2_target_delta/<worklist-sha>/`.
They never overwrite a full target payload.

After extraction, rerun the route/row auditor with the deltas declared as
sources. Full target materialization is allowed only when the auditor reports
complete unique raw-ID ownership. A legacy/delta duplicate is a hard failure;
there is no overwrite precedence.

Example render command (it does not submit):

```bash
python scripts/windowfix_v2/prepare_sco_gap_jobs.py \
  --source-audit /absolute/path/stream_route_audit.json \
  --source-audit /absolute/path/target_row_route_audit.json \
  --protocol protocols/inscene_streaming_canonical_v2.json \
  --canonical-source manifests/c_retrain_20260722/inscene_c_h124_prefix24.jsonl \
  --output-dir artifacts/windowfix_v2/sco_plans/canonical_v2_gap_only_<stamp>
```

The output contains immutable shards, one serial batch per VFM, review-only
SCO argv/commands, blocked backend-canary worklists, a summary, and
`PLAN.lock.json`. No command is submitted by the renderer.

For the four canary-gated backends, use
`prepare_gap_backend_canary_jobs.py` with the immutable gap plan and a separate
new output directory.  Its four review-only specs bind the blocked worklist,
source audit, source plan, protocol, current extractor, exact prompt/resize/
frame count, and canary runner SHAs.  Missing receipts remain blocked; a
post-canary gap plan accepts only receipts whose two-mode comparisons and all
of those bindings validate exactly.
