# Result and experiment-status audit — 2026-10-11

**Follow-up reconciliation:** the [three-part inventory](RELEASE_READINESS_20261011.md)
supersedes the coarse campaign counts below. It verifies 696/736 canonical-V2
training completions, including CogVideoX C48 and all MiniMax L30 A/B16;
576 timestep completions with 216 matched independent evaluations; and the
three completed V-JEPA2 D2 development runs. The earlier dated snapshots below
are retained as history, not current missing-training claims.

This audit reads existing local ledgers, reports, receipts, and result tables. It
does not launch training, evaluation, extraction, or scheduler queries. A stored
`RUNNING` state is a historical observation, not a statement that the job is
running today. Counts below belong to separate experiment universes and must
not be added together.

The repository contains several completed result packages, but neither the old
`paper_ready_results.csv` nor a successful training marker establishes a final
test result. Public documentation should distinguish execution completion,
metric validity, aggregate completion, and the split on which a claim rests.

## Historical 1,260-experiment ledger

Source: [task_status_1096.csv](../experiments/task_status_1096.csv). The filename
is historical; the file has **1,260** rows. These are recorded ledger states,
not a refreshed inventory of physical checkpoints.

| Recorded evaluation state | Experiments | Interpretation |
|---|---:|---|
| `COMPLETE_CANONICAL` | 625 | Candidate canonical metrics; final-claim policy is not frozen |
| `COMPLETE_RAW` | 43 | Raw metric exists; canonicalization remains unresolved |
| `METRIC_UNRESOLVED` | 160 | Provisional historical Wan/Cog metrics; not formal best results |
| `NOT_STARTED` | 341 | Training complete in the ledger; matched evaluation missing |
| `NOT_APPLICABLE_TRAIN_INCOMPLETE` | 91 | Training failed, partial, or not started |

The reported **828 `TRAIN_AND_EVAL_COMPLETE`** equals 625 + 43 + 160.
It therefore includes 160 unresolved metrics. It is not a count of accepted
final results. Training status records 1,169 complete, 42 failed, 21 retrying,
16 partial, 8 not started, and 4 declared jobs. The lifecycle field folds the
42 failed and 21 retrying rows into 63 `TRAIN_FAILED` rows.

The 341 evaluation gaps are LingBot 101, CogVideoX 113, Wan 80, MiniMax-H3 28,
V-JEPA2 17, F3R 1, and VideoMAE 1. By probe they are mask 91, action dynamics
66, path integration 64, counterfactual 51, depth 32, view consistency 25, and
ego belief v2 12. Do not rerun these automatically: later independent or
canonical-v2 experiments have different identities and do not silently update
this historical ledger.

The 160 unresolved metrics are CogVideoX 94 and Wan 66. Their own
`status_reason` explicitly says they are provisional. The 43 raw-only rows
are 23 ego-belief-v2 and 20 mask rows. The ledger has no `final_claim_ready=YES`:
631 rows say `POLICY_NOT_FROZEN`, and 629 say `NO`.

[paper_ready_results.csv](../results/paper_ready_results.csv) has **2,184 metric
rows for 576 experiment IDs**, all `candidate_v1_val_protocol`, all
`final_claim_ready=POLICY_NOT_FROZEN`. **1,739 rows specify `val`; 445 leave the
split blank.** None specifies `test`. All 576 IDs map to eligible ledger rows,
but **49 eligible ledger IDs have no row in this export** (28 V-JEPA2 and 21
VideoMAE). Thus the export is neither 2,184 distinct experiments nor a complete
export of all 625 eligible ledger IDs.

At audit start, the historical status documents disagreed: `PROJECT_STATUS.md` and
`docs/task_status_dashboard.md` show 1,260/828, whereas
`experiments/task_status_summary.csv` still says 1,096/835. The prose training
summary in that pre-audit `PROJECT_STATUS.md` also collapsed the retrying state. The root status page now points to this audit; the original is archived. Regenerate
these views from one explicitly dated source before advertising a completion
percentage.

## Implemented work without a complete final result

| Work | Verified evidence | Remaining result gap |
|---|---|---|
| Main canonical-v2 campaign | Follow-up inventory verifies main512/552 + Wan168/168 + MiniMax L30 A/B16/16 training completions | LingBot C40 completion remains unconfirmed; 28 Wan A2 gate conflicts and formal metrics require acceptance |
| Timestep seeded-v1 | 576/576 recorded training completions; 216 separate A1-depth/A2/B2 validation metrics exist | Full valid official metric coverage, especially C1/C2/C3; MiniMax degeneracy remains visible |
| V1 C3-from-C1 action sensitivity | 40-cell selection, 32 physical checkpoints, five failed canary attempts | Exact V1 feature inputs for 32 cells; physical checkpoints for the other 8; no scored summaries |
| Active Spatial representation MVP | State-bank builder, extractors, trainer, a real v50 single-state forward | Valid 48-state matched canaries for all requested checkpoints and the approximately 2,000-state probe result |
| SeeSE3 replication sanity | Separate preflight and three-seed request are implemented | Representative middle-layer features and sufficient pair counts; preflight is blocked |
| Probe-validity B2/P16 four-model extension | B2 perturbation code and CogVideoX-only evidence exist | Exact V-JEPA2, VideoMAE, and Wan checkpoints are missing; no accepted four-model P16 result |
| D-series unified formal export | Completed Wan D1/D2 and V-JEPA2 D1/D3 three-seed aggregates; all three V-JEPA2 D2 runs and dev evaluations verified complete | D2 aggregation and full formal export; retain dev scope, which does not require inventing a test split |

### Main canonical-v2: historical snapshots, subsequently reconciled

The directory name `main_v2_probe_v2_816_20260917` is not the current completed
experiment count. Its original campaign selected 816 of an older 1,096
universe; a later 552 queue and separate Wan/MiniMax scopes coexist inside it.

Evidence:

- [2026-10-01 snapshot](../artifacts/main_v2_probe_v2_816_20260917/probe_progress_20261001/summary.json): 430/552 successes, with dated per-model counts.
- [2026-10-02 update](../artifacts/main_v2_probe_v2_816_20260917/feature_progress_20261001T0834Z/OTHER_TASKS_STATUS_20261002T0413Z.md): Wan's remaining 48 C jobs succeeded, so this update plus the earlier 120 accounts for 168 Wan training successes; MiniMax L30 was 15/16. CogVideoX C had 48/48 validation PASS but 24/24 failed scheduler wrappers and no formal release job IDs.
- [LingBot C validation bundle](../artifacts/main_v2_probe_v2_816_20260917/probe_validation_lingbot40_c_20261004_r1/bundle.json): all 40 `cases/*/result.json` files locally say `PASS`.
- [LingBot formal submission ledger](../artifacts/main_v2_probe_v2_816_20260917/single_gpu_lingbot40_c_h800_20261004_r1/submissions.json): 19 entries, all `SUBMIT_INTENT`, and zero recorded `job_id` fields. Validation success and submit intent do not establish formal training submission or completion.
- [Watcher summary](../artifacts/main_v2_probe_v2_816_20260917/probe552_progress/summary.json): `updated_utc` is 2026-10-05, but `scheduler_observed_utc` is 2026-09-28 and `feature_observed_utc` is 2026-10-01. Its 376 successes conflict with the later-observed 430-success snapshot. The write timestamp cannot determine evidence freshness.

These snapshots alone did not establish a current global success count.
The [follow-up per-run audit](CANONICAL_V2_INVENTORY_20261011.md) now establishes
696 completed trainings, using exact identities, normal-stop logs and
nonempty checkpoints. Cog C48 ended on October 4–5, and the last MiniMax
L30 A/B run ended after the October 2 snapshot quoted above. Formal metric
acceptance remains separate; scheduler success alone cannot establish it.

### Timestep: training and evaluation are distinct

[FINDINGS.md](../results/analysis/tstep_canonical_v2_seeded_v1/FINDINGS.md)
records 576 training completions, 513 finite training-time primary metrics,
63 missing/nonfinite primary metrics, and 32 exact-zero A1 losses excluded
from plots. It explicitly warns about constant B2 values and saturated C
cosines.

That report predates a useful partial completion:
[official_eval_results.csv](../results/analysis/tstep_canonical_v2_seeded_v1/official_eval_results.csv)
now has **216 finite evaluator rows**, 72 each for depth, view consistency,
and ego-belief-v2. The [evaluation plan](../artifacts/tstep_eval_a1a2b2_20260923_r1/PLAN.json)
explicitly uses `eval_split=val`. **16 depth rows are exactly zero**; finite
does not mean valid. The CSV has no C1/C2/C3 rows and does not replace their
required retrieval/intervention metrics. The other 360 cells of the
576-cell training universe are outside this 216-row export, not necessarily
failed evaluations.

The earlier mixed-window timestep R1 is a separate, invalid protocol:
[INVALID_R1_20260830.md](../reports/tstep_canonical_v2/INVALID_R1_20260830.md)
invalidates all 84 recorded completion markers because feature and label
window identities disagreed. These markers must not fill canonical-v2 gaps.

### Active Spatial: partial execution supersedes the old NOT_RUN prose

The [MVP report](../artifacts/active_repr_probe/MVP_REPORT.md) and top-level
canary status are older blocked-host observations. Later local artifacts
show actual execution:

- [v50 forward manifest](../artifacts/active_repr_probe/runs/asrp_v50_smoke_r9_20260906T144500Z/v50_smoke_manifest.json) says `FORWARD_PASSED` for one real state and all three taps.
- [r8 state-bank result](../artifacts/active_repr_probe/runs/asrp48_state_r8_20260906T144700Z/state_bank_result.json) says `FAILED`: 25 rows were built instead of 48. Its nested state validator accepting those 25 rows is not acceptance of the 48-row canary contract.
- `artifacts/active_repr_probe/results.csv` still contains only a `NOT_RUN` result row; no completed MVP result is established by these smoke artifacts.

### C3 and replication are real gaps, not successful null findings

[C3 input availability](../results/analysis/c3_action_sensitivity_v1_20260923T110445Z/input_availability.json)
records 32 `UNAVAILABLE` and 8 `NOT_APPLICABLE` cells. The
[selection audit](../results/analysis/c3_action_sensitivity_v1_20260923T110445Z/selection_audit.json)
confirms 32 checkpoint hashes and 8 missing checkpoints. All five recorded
canary shard exits are 1, and the checked cells have no evaluation summary.
Canonical-v2 features are explicitly ineligible substitutes for these V1 runs.

[SeeSE3 preflight](../results/experimental/seese3_replication_dino_preflight_seed42_43_44_r1/REPORT.md)
says `blocked_missing_replication_prerequisites`: the only DINO representation
is labeled layer 0, and there are too few pairs for the declared 1,000 training
and 200 evaluation pair minima. This does not invalidate the separate completed
observed-order experiments.

## Existing result packages that should not be reported as missing

| Package | Locally checked completion evidence | Claim boundary |
|---|---|---|
| Independent P8 actionability/algebra | `paper/FINAL_INTEGRITY_AUDIT.md` says `COMPLETE_VALID`; tables, per-sample CSVs, and figures exist | Frozen validation, three training seeds; pilot development used validation, so it is not blind test evidence |
| Relational continuous-action V2 | `RUN_MATRIX.csv` has 100/100 `COMPLETE_VALID` rows; integrity receipt records 32,768 evaluation rows and 140 paired-effect rows | 256 trajectories/108 validation scenes; five seeds; supervised first-action readout, not an online policy |
| SE(3) Wan/V-JEPA2 final r3 | Freeze audit `PASS`; 12 layers and five seeds; all 21 generated-file SHA256 entries match local bytes | 276 evaluation tuples in the same 69 scenes used by the pilot; no independent-scene confirmation |
| C1/B2 probe-validity section | `paper/paper_results.csv` contains 68 rows across four models | Development validation and one probe-training seed; B2 uses P12, not a complete P16 extension |
| D-series development section | Wan has 138 per-seed/46 aggregate/18 paired rows; V-JEPA2 D1/D3 has 276/92/16 | Development-only; 612 train/69 dev scenes; no historically unused held-out test set |

Evidence locations:

- [Independent P8 integrity audit](../results/independent_actionability_algebra_p8_final_v1_20260913/paper/FINAL_INTEGRITY_AUDIT.md).
- [Continuous-action integrity audit](../results/independent_goal_conditioned_continuous_action_p8_relational_v2_20260923/FINAL_INTEGRITY_AUDIT.md).
- [SE(3) freeze manifest](../results/experimental/se3_wan_vjepa2_paper_final_20260923_r3/FREEZE_MANIFEST.json) and [statistical audit](../results/experimental/se3_wan_vjepa2_paper_final_20260923_r3/STATISTICAL_AUDIT.json).
- [Probe-validity scope](../analyses/probe_validity/paper/README.md).
- [D-series September 23 report](../experiments/d_series_scannetpp_v1/paper_outputs/dseries_development_20260923_r1/analysis_report.md), [Wan aggregate](../experiments/d_series_scannetpp_v1/diagnostics/dseries_wan_d1_d2_three_seed_20260923_zoetrope_r2/summary.json), and [V-JEPA2 aggregate](../experiments/d_series_scannetpp_v1/diagnostics/dseries_vjepa2_d1_d3_three_seed_20260917_zoetrope_r1/summary.json).

D-series `STATUS.md` and `README.md` still describe Wan as incomplete, but the
September 23 Wan result package establishes completion of that development
matrix. V-JEPA2 D2 has all three local `checkpoint.pt`, `run_receipt.json`, and
`per_sample_metrics.csv` sets under
`diagnostic_runs/dseries_vjepa2_d2_seed{0,1,2}_20260917_103733_zoetrope_r1/`.
Its rows are absent from the checked D1/D3 aggregate, and the named
`paper_outputs/dseries_development_20260923_r1/` directory contains only prose,
TeX, and slides. The [follow-up analysis inventory](ANALYSIS_INVENTORY_20261011.md)
verifies all three D2 training and dev-evaluation receipts. The required gap
for a full dev release is D2 aggregation and the unified export. Independent
test confirmation is an optional extension, not a reason to rerun completed
Wan or V-JEPA2 development training.

## Release-facing corrections

1. Keep historical ledger results, canonical-v2 reruns, and independent paper
   packages separately identified. Publish one dated result index linking each
   accepted package to its split, protocol, metric table, seed count, and scope.
2. Replace completion claims based on directory names, submit intent, scheduler
   exit state, or marker files with checks of the corresponding durable result
   and protocol identity. Preserve the old receipts as provenance.
3. Label the old `paper_ready_results.csv` as a candidate validation export.
   Reconcile its 49 missing eligible IDs and 445 unspecified split rows before
   using it as the public result index.
4. Mark failed, superseded, and incomplete experiments explicitly in public
   documentation. Keep invalid R1 and obsolete pilot values out of accepted
   tables without deleting the evidence needed to explain their exclusion.
5. Do not invent a blind test result to satisfy a release checklist. The
   completed validation/development packages can be released with their actual
   claim boundaries; additional independent confirmation is separate work.

Earlier coarse scope summary (superseded for current per-run status):
[release_result_gaps_20261011.csv](../metadata/release_result_gaps_20261011.csv).
