# Canonical-v2 feature preparation handoff

This document covers feature preparation only.  No probing run is allowed to
consume these artifacts until both the streaming route catalogue and every
target row merge report are complete.  None of the review plans below has
been submitted to SCO.

## Frozen temporal contract

- protocol: `inscene_streaming_canonical_v2_obs28_h124`
- protocol file: `protocols/inscene_streaming_canonical_v2.json`
- protocol file SHA-256:
  `cb4feaecc1deaeb9be0ee7e9e66c0e205d96528e23513def934aeaf93f01465c`
- canonical parent:
  `manifests/c_retrain_20260722/inscene_c_h124_prefix24.jsonl`
- canonical parent SHA-256:
  `ebc496eb0587cda372ccf29a7c5244ef454096151f73c8eed9511967ded1902d`
- one fixed 28-observation parent window; p8/p12/p16/p24 and h1/h2/h4 are
  derived from that same raw-ID record.

The compact raw-label lineage audit is
`artifacts/windowfix_v2/canonical_v2_compact_source.02e17c40b59a.audit.json`
(file SHA-256
`31b9a6423dc8aa69e9ec6fc2fc9b94772794b5cff8023d203105722833c9e6aa`).
It passed with zero problems after checking 681 scenes, 2,721 canonical
windows, 74,440 raw frame identities and their image/depth/mask/pose records,
and all 10,884 derived prefix rows.  This is the label-side contract that
future feature routes must match.

## Streaming input features

The frozen initial catalogue is
`artifacts/windowfix_v2/reference_overlay_02e17c40b59a_r11`.
It partitions all 435,360 canonical feature units exactly once:

| status | units | action |
|---|---:|---|
| local read-only reference | 5,298 | reusable now |
| remote semantic candidate | 212,098 | run the AOSS metadata/header gate |
| extractor-identity mismatch | 39,832 | re-extract |
| raw-ID mismatch | 178,132 | re-extract |

Thus 217,396 units are semantic reuse candidates, but only 5,298 are already
routable.  The other 212,098 must not be called reusable until their fresh
remote receipts pass.  A failed or unreadable gate becomes an explicit
re-extraction route; a timeout remains pending.

The initial positive gap worklists are at
`artifacts/windowfix_v2/streaming_gap_worklists_ebc496eb0587_catalog_r11_20260831`.
They contain 217,964 layer-items and exclude all 212,098 pending candidates.
The review-only extraction plan is
`artifacts/windowfix_v2/sco_plans/canonical_v2_gap_only_ebc496eb0587_streaming_r13_20260831`:

- 178,194 layer-items across CogVideoX, DINO, MiniMax-H3, VideoMAE, V-JEPA2,
  and Wan have runner dry-run PASS;
- 39,770 layer-items across Aether, Fast3R, LingBot, and Qwen remain blocked
  until their deterministic backend canaries pass.

The four review-only canary specs are in
`artifacts/windowfix_v2/sco_plans/canonical_v2_gap_backend_canaries_ebc496eb0587_streaming_r13_20260831`.
Qwen's receipt additionally supplies exact p8/p12/p16/p24 tensor shapes.  Once
that receipt exists, rebuild a new immutable catalogue with:

```bash
/mnt/umm/users/yinbaiqiao/.conda/envs/probe-spatial/bin/python \
  analyses/windowfix_v2/build_streaming_reference_catalog.py \
  --qwen-shape-canary-receipt <reviewed-qwen-pass-receipt.json> \
  --output-dir <new-immutable-catalogue-directory>
```

Never overwrite r11.  Rebuild the remote-gate and gap plans from the new
catalogue; counts may stay equal, but every downstream SHA must change.

### Streaming remote gate

The frozen offline 128-shard source plan is
`artifacts/windowfix_v2/aoss_reference_gate_plan_r11_hardened_r12_128shards_20260831`.
Its SCO-wrapped review catalogue is
`artifacts/windowfix_v2/aoss_reference_gate_sco_review_r11_hardened_r13_20260831`.
It contains 128 commands for 212,098 requests and records zero executions and
zero submissions.  Each worker revalidates the plan, catalogue inputs,
outputs, shard, gate implementation, schema, Python executable, AOSS config,
and its own binding receipt before any AOSS read.

After all reviewed gate receipts are available, finalize them into a new
directory:

```bash
/mnt/umm/users/yinbaiqiao/.conda/envs/probe-spatial/bin/python \
  analyses/windowfix_v2/finalize_aoss_streaming_gate.py \
  --catalog-dir <the-catalogue-used-by-the-gate> \
  --receipt-root <complete-receipt-root> \
  --output-dir <new-immutable-finalized-route-directory>
```

Only `PASS` receipts become read-only routes.  Feed the finalizer's explicit
failed routes into a second immutable worklist build:

```bash
/mnt/umm/users/yinbaiqiao/.conda/envs/probe-spatial/bin/python \
  analyses/windowfix_v2/build_streaming_gap_worklists.py \
  --catalog-dir <the-catalogue-used-by-the-gate> \
  --failed-gate-routes <finalized>/failed_gate_reextract.routes.jsonl.gz \
  --protocol protocols/inscene_streaming_canonical_v2.json \
  --canonical-source manifests/c_retrain_20260722/inscene_c_h124_prefix24.jsonl \
  --output-dir <new-immutable-gap-worklist-directory>
```

The extractor's resume identity is per canonical unit rather than global
worklist SHA, so this second stage does not recompute a valid first-stage
feature.

## Isolated target features

The immutable target merge plan is
`artifacts/windowfix_v2/target_row_merge_initial/target_merge_plan.json`.
It covers 32 physical layers x 681 scenes = 21,792 units and 1,205,664
row-layer items.

The current offline audit is
`artifacts/windowfix_v2/target_row_merge_initial/route_audit_offline_pending_r5`:

- 16,344 local Wan rows are full-SHA verified;
- 828,894 rows remain pending remote source gates;
- 360,426 rows are definite gaps;
- zero duplicates and zero hard errors were observed.

The current definite-gap review plan is
`artifacts/windowfix_v2/sco_plans/canonical_v2_target_definite_gaps_r6_20260831`.
CogVideoX and Wan account for 285,072 ready row-layer items.  Aether and Qwen
account for 75,354 blocked items.  Their target-specific canaries are in
`artifacts/windowfix_v2/sco_plans/canonical_v2_target_gap_backend_canaries_r6_20260831`
and assert row tail shape/dtype as well as deterministic payload equality.

The hardened ten-shard remote audit review is
`artifacts/windowfix_v2/target_row_merge_initial/sco_audit_review_e5b4582cd499_r5_hardened`.
Each worker verifies the full plan, shard, auditor, source reader, binding
schema, and wrapper SHA before remote access.  After all ten outputs exist,
they must be combined into one full-plan-bound route audit before gap planning
or delta augmentation; a single shard audit is never sufficient.

```bash
/mnt/umm/users/yinbaiqiao/.conda/envs/probe-spatial/bin/python \
  analyses/windowfix_v2/merge_target_route_audit_shards.py \
  --plan artifacts/windowfix_v2/target_row_merge_initial/target_merge_plan.json \
  --review-manifest artifacts/windowfix_v2/target_row_merge_initial/sco_audit_review_e5b4582cd499_r5_hardened/manifest.json \
  --output-dir <new-immutable-unified-route-audit-directory>
```

The merger verifies exact disjoint coverage of all 21,792 parent units,
re-checks every shard/binding/output SHA, keeps pending rows pending, rebuilds
unified gap worklists, and emits the standard full-parent-bound audit schema.

Target extraction writes only SHA-isolated deltas below
`FEAT_TARGET_CANONICAL_V2_GAP_DELTA_V1`.  It never overwrites a full target
cache.  After a complete extraction stage, add verified deltas to a new plan:

```bash
/mnt/umm/users/yinbaiqiao/.conda/envs/probe-spatial/bin/python \
  analyses/windowfix_v2/augment_target_merge_plan.py \
  --initial-plan <immutable-input-plan.json> \
  --route-audit <full-plan-bound-route-audit.json> \
  --delta-root /mnt/umm/users/yinbaiqiao/InsScene-15K/FEAT_TARGET_CANONICAL_V2_GAP_DELTA_V1 \
  --completion-binding <shard.json> <COMPLETE-receipt.json> \
  --output <new-immutable-augmented-plan.json>
```

Materialization remains forbidden until a fresh audit has `PASS`, with zero
missing rows, zero pending rows, and zero unresolved duplicate owners.  The
writer is `scripts/windowfix_v2/materialize_target_row_merge.py`; invoke it
without `--execute` first.

## Fastest safe execution order

These independent jobs may run in parallel after human review:

1. six ready streaming gap batches (178,194 layer-items);
2. two ready target gap batches for CogVideoX/Wan (285,072 row-layer items);
3. four streaming backend canaries;
4. two target backend canaries;
5. 128 read-only streaming AOSS gate shards;
6. ten read-only target AOSS audit shards.

Then, in order:

1. validate all receipts and rebuild immutable route catalogues;
2. create second-stage worklists only for definitive gate failures/new gaps;
3. run the remaining gap-only extraction batches;
4. re-audit every streaming raw-ID route and every target row;
5. materialize target mixed caches only after full PASS;
6. freeze feature/index/sidecar hashes and only then render probing jobs.

No old probing checkpoint, optimizer state, metric file, or result directory
may be reused.  Future probing outputs must use a new canonical-v2 namespace
and run tag so that the historical misaligned results remain quarantined.
