# Canonical-v2 SCO feature preparation

This directory prepares feature-extraction work only. It does not start probing
training, and the plan renderer never calls `sco`.

Contract:

- protocol `inscene_streaming_canonical_v2_obs28_h124`;
- one shared 28-observation window layout;
- streaming prefixes `8,12,16,24` are always emitted together;
- isolated targets contain every prefix anchor and horizons `1,2,4`;
- the protocol-declared `_SEEDED_V1` roots are immutable, while every shard and
  receipt binds the protocol-file SHA and exact model/timestep identity;
- legacy roots and symlink aliases are rejected;
- AOSS publication is unavailable until four full alignment audits pass.

Render from a cache-audited inventory:

```bash
SPECS=artifacts/windowfix_v2/reuse/<audited-specs>.csv \
CANONICAL_SOURCE=artifacts/windowfix_v2/<frozen-obs28-source>.jsonl \
RUN_TAG=canonical_v2_feature_fill_YYYYMMDD \
bash scripts/windowfix_v2/render_sco_plan.sh
```

The temporal/label gate stores only one frozen obs28 source. Prefix rows are
derived as `window[:p]` during audit; four redundant ~55 MiB manifest copies
are neither required nor trusted. Every plan binds the source SHA, namespace
lock SHA, and full label-lineage receipt SHA.

`PENDING_CACHE_AUDIT` is rejected by default. For a conservative worst-case
capacity plan, set `ALLOW_PENDING=1`; this schedules pending rows as extraction
but still does not submit anything.

Important outputs are `summary.json`, `PLAN.lock.json`, `shards/*.json`,
`batches/*.json`, `jobspecs/*.json`, and `sco_create_commands.txt`. The last file is review text,
not an executable submitter. Each prepared worker command invokes
`run_canonical_feature_shard.py --batch ... --execute`; invoking the runner
without `--execute` is a read-only command preview. All timesteps/modes for one
VFM are serialized in one batch because they share scene-level structural
index files; this prevents concurrent `prefix_index.npy` writer races.

After extraction, execute every command in `audit_plan.json`. Only after all
four prefixes have full PASS reports may publication be previewed:

```bash
python scripts/windowfix_v2/publish_after_audit.py \
  --plan-dir artifacts/windowfix_v2/sco_plans/<run-tag>
```

Actual publication additionally requires `--execute --confirmation <plan_id>`.
It syncs to the versioned AOSS URI and never deletes local files.

The current certified launch whitelist is Wan, CogVideoX, MiniMax-H3, DINO,
V-JEPA2, and VideoMAE. The last three use an explicit
`deterministic_forward_no_noise_v1` contract, so a newly written cache can be
strictly validated during resume. Aether, F3R, and LingBot now receive
sample-local generators and write matching provenance, but remain blocked
until a same-sample cross-process/order GPU equality canary passes. Qwen now
has a canonical obs28 streaming/target extractor with strict no-noise
provenance, but is likewise blocked until its cross-process/order GPU canary
passes. Any backend outside the whitelist is fail-closed until its extractor,
resume validator, and provenance have been certified.

Workers also remove any inherited `EXTRACT_DONE_INDEX`: a remote object key
alone cannot establish tensor, sidecar, or provenance integrity in seeded-v1.

Aggregate isolated-target reuse is a separate mixed-lineage workflow.  See
`analyses/windowfix_v2/TARGET_ROW_MERGE.md`.  It publishes only to
`FEAT_TARGET_CANONICAL_V2_MIXED_V1`, never to `_SEEDED_V1`, and requires a
complete raw-ID route audit before the no-overwrite materializer can execute.

After the four backend canary jobs finish, rerender into a new immutable plan
directory with `BACKEND_CANARY_DIR=<old-plan>/backend_canaries`. The renderer
checks each receipt's protocol, extractor, canary-script, model/timestep/layer,
and both streaming/target comparisons before it can unblock that backend. A
missing, stale, edited, or failed receipt remains blocked; the original plan is
never modified in place.
