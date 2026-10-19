# Canonical-v2 target row reuse and merge

This path repairs aggregate isolated-target caches without pretending that a
legacy import was newly generated under `_SEEDED_V1`.  The only output
namespace accepted by the materializer is
`FEAT_TARGET_CANONICAL_V2_MIXED_V1`.

## Frozen plan

Build the plan (metadata/index reads only; no AOSS access and no feature
writes):

```bash
python analyses/windowfix_v2/build_target_merge_plan.py
```

The committed initial plan is
`artifacts/windowfix_v2/target_row_merge_initial/target_merge_plan.json`.
It expands 32 physical model/layer/timestep identities over 681 scenes, or
21,792 scene-layer units.  The canonical target union contains 37,677 rows per
full layer and 1,205,664 row-layer items in total.

Candidate does not mean reusable.  Every declared source still has to pass:

- exact raw target ID and unique ownership;
- exact VFM/model/layer/timestep/mode/prompt/preprocessing identity;
- readable safetensors key, tail shape and dtype;
- sidecar-to-file size/index/hash binding.

The six historical flat Wan layers with `prompt=''` are intentionally not
declared: canonical Wan uses `prompt='.'`, so those 194,100 row-layer items are
raw-ID-only candidates and semantically incompatible.  The six-layer local
manifest-exact Wan cache does use `'.'`; r5 fully SHA-verified all 4,086 local
payloads and confirmed its 16,344 matching row-layer items.  Twenty-one flat AOSS layers
have an upper bound of 679,350 candidates based on the locally preserved
32,350-row union index; the AOSS gate must prove that bound independently for
every actual payload.

DINO layer 0 declares the four historical prefix roots (24, 16, 12, 8) with a
fixed priority in that order.  Their union has an unverified upper bound of
32,350 rows.  The only permitted duplicate policy is DINO-only: every
overlapping raw-ID row must be bitwise equal across all owners before the
higher-priority owner may be selected.  A non-DINO source ID, changed priority,
gap-delta owner, or unequal row fails closed.

Even if every declared candidate passes, at least 477,620 row-layer items need
new extraction.  A failed remote identity or payload gate increases that
number.

## Audit, extract gaps, and merge

First run an offline audit without contacting AOSS.  `--verify-source-sha`
fully verifies local candidates while remote candidates remain explicitly
`PENDING_AOSS_GATE` and are not emitted as gaps:

```bash
python analyses/windowfix_v2/audit_target_row_routes.py \
  --plan artifacts/windowfix_v2/target_row_merge_initial/target_merge_plan.json \
  --output-dir <new-immutable-offline-audit-dir> \
  --remote-policy pending --verify-source-sha
```

The exact remote gate command is recorded in
`target_merge_plan.summary.json`.  Run it only on a worker with AOSS access;
it uses `--remote-policy gate --source-timeout-seconds 30`.  A timeout or
transport failure remains pending instead of becoming a re-extraction task.

The audit emits `target_row_routes.jsonl`, `missing_target_units.jsonl`, and
hash-bound `gap_worklists/*.json`.  Only explicit missing rows enter a
`CANONICAL_V2_GAP_WORKLIST_V1` target-isolated extractor job.  Delta outputs
must stay under `.canonical_v2_target_delta/<worklist-sha-prefix>/`; they must
never overwrite a full target cache.

The current offline r5 audit is
`artifacts/windowfix_v2/target_row_merge_initial/route_audit_offline_pending_r5/target_row_route_audit.json`.
Its 360,426 definite row-layer gaps have a review-only extraction plan at
`artifacts/windowfix_v2/sco_plans/canonical_v2_target_definite_gaps_r6_20260831`.
CogVideoX and Wan account for 285,072 ready row-layer items; Aether and Qwen
account for 75,354 blocked items and require bound backend-canary receipts
before rerender.  Their target-specific review-only canaries are in
`artifacts/windowfix_v2/sco_plans/canonical_v2_target_gap_backend_canaries_r6_20260831`.
No SCO command in either plan has been executed.

The hardened remote review plan is
`artifacts/windowfix_v2/target_row_merge_initial/sco_audit_review_e5b4582cd499_r5_hardened`.
It splits the immutable plan into ten VFM shards.  Every worker validates the
parent plan, its shard, auditor, source reader, wrapper, schema, and binding
receipt before any remote read.  The ten resulting shard audits must be
combined into one full-parent-bound audit with
`merge_target_route_audit_shards.py`; passing a shard audit directly to the
gap planner or augmenter is forbidden.

After all ten worker-bound output directories exist, merge them without any
AOSS access or feature writes:

```bash
python analyses/windowfix_v2/merge_target_route_audit_shards.py \
  --plan artifacts/windowfix_v2/target_row_merge_initial/target_merge_plan.json \
  --review-manifest artifacts/windowfix_v2/target_row_merge_initial/sco_audit_review_e5b4582cd499_r5_hardened/manifest.json \
  --output-dir <new-immutable-unified-route-audit-dir>
```

By default the ten report paths are taken from the reviewed worker receipts.
For an additional operator-side check, repeat
`--shard-audit VFM=/absolute/path/target_row_route_audit.json` for **every**
reviewed VFM; any omitted, extra, or path-divergent report fails closed.  The
unified report keeps audit type `canonical_v2_target_row_route_audit`, binds
`plan_sha256` to the full 21,792-unit parent plan, and regenerates its own
`gap_worklists`.  It is therefore the report to pass to
`prepare_sco_gap_jobs.py` and `augment_target_merge_plan.py`.

After every scheduled shard has a `COMPLETE` receipt, build a **new** plan with
the verified deltas (one `--completion-binding` per shard/receipt pair):

```bash
python analyses/windowfix_v2/augment_target_merge_plan.py \
  --initial-plan <immutable-input-plan.json> \
  --route-audit <immutable-route-audit.json> \
  --delta-root /mnt/umm/users/yinbaiqiao/InsScene-15K/FEAT_TARGET_CANONICAL_V2_GAP_DELTA_V1 \
  --completion-binding <shard.json> <receipt.json> \
  --output <new-immutable-augmented-plan.json>
```

The augmenter requires exact missing raw IDs, full feature/index SHA checks,
both strict delta sidecars, exact worklist and receipt bindings, and complete
scene-range receipts.  It never changes the input plan and refuses an existing
output.  A later remote gate can emit a second immutable audit/worklist; its
different worklist file SHA selects a different
`.canonical_v2_target_delta/<sha-prefix>/` directory, and the next augmentation
must again write a new plan.  Thus second-stage deltas cannot overwrite the
first stage.

For DINO, a
post-gap plan must not retain the conditional multi-prefix policy alongside a
delta; first resolve/validate the overlapping legacy prefix rows and freeze an
unambiguous source route.  Materialization is allowed only when the audit
status is `PASS`, with zero missing, pending, or forbidden duplicate owners.

Preview (read-only):

```bash
python scripts/windowfix_v2/materialize_target_row_merge.py \
  --plan <complete-immutable-plan.json>
```

Execute only after reviewing the complete audit:

```bash
python scripts/windowfix_v2/materialize_target_row_merge.py \
  --plan <complete-immutable-plan.json> --execute
```

The writer range-reads only selected AOSS tensor rows, reorders them into the
canonical sorted raw-ID order, checks every reused row bitwise after reopening
the staged output, and publishes data/index/sidecars/receipt with no-overwrite
semantics.  A duplicate owner, missing row, source mutation, shape/dtype
change, partial transaction, or existing foreign destination fails closed.
