# GitHub release audit — 2026-10-11

This audit inspected the current working tree, repaired reproducible defects,
and separated portable source/tests from local campaign evidence. It did not
rerun research training, change published metric values, or submit cluster
jobs. A clean CPU test run is a source-code check, not certification of the
existing scientific results.

The checkout already contained 36 modified tracked files, extensive untracked work, and 151
tracked-file deletions. Those changes were preserved. The audit did not reset,
stage, commit, or publish the repository, and did not delete unique experiment
artifacts. New source and audit files must be included in the eventual release
commit; `git add -u` alone would omit untracked implementations and tests.

## Correctness fixes

| Problem | Fix and verification |
| --- | --- |
| `seed=0` silently skipped seeding | Train entrypoint tests explicitly exercise zero, nonzero, and absent seeds |
| Import-time checkpoint compatibility patch changed global `torch.load` behavior | Remove the patch; pass `weights_only=False` explicitly for project checkpoints through Lightning's public API; regression verifies default and explicit restricted loading |
| Resumed training could test its original input checkpoint | Training followed by testing now selects the best newly trained checkpoint, or current weights when none was saved; test-only mode still uses the supplied checkpoint |
| Resume preflight unnecessarily loaded tensors onto saved devices; `logger=null` could fail | Read checkpoint metadata on CPU and handle absent logger configuration |
| Training crashed with `scheduler=null` although that option is supported | LR logging reads the optimizer in both training modules |
| Invalid batches received validation weight 1 | Use actual valid counts, sum count metrics, and prevent undefined retrieval ranks from contaminating valid batches; tested with Lightning CPU epoch aggregation |
| Invalid and valid samples emitted different metric names on different ranks | Keep the same ordered metric schema for each probe; contract tests cover six diagnostic probes |
| A3/B1 lost manifest IDs; A3 invalid rows could look 100% valid | Preserve lineage and calculate validity from actual record flags; regression checks the emitted evaluation JSON |
| Typoed dataset/evaluation splits silently selected validation or test | Reject unknown split values at the entrypoint |
| SAE angular loss treated the two sides of ±π as far apart | Wrap azimuth differences consistently in loss and metrics |
| One-frame SAE overlap produced empty-pair NaNs | Skip an absent overlap objective and retain reconstruction loss; count available overlap pairs |
| `compile=true` silently did nothing | Explicitly reject unsupported compiled diagnostic checkpoints instead of implying a working strict-evaluator round trip |
| A1 overall validation loss weighted unequal batches equally; test outputs used validation labels | Weight by sample count, reset metrics for each split, and label test metrics/visualizations as test |
| CPU training-module import required optional 3D rendering packages | Import 3D alignment/rendering dependencies only in branches that use them |
| Offline model lookup used a personal cache path | Honor standard Hub cache environment precedence and the current user's cache; preserve generator-supplied required-file provenance |
| AOSS network/authentication errors became cached missing inputs | Propagate failed lookups; a successful empty child-prefix listing cannot certify that the object is absent; positive directory fallback still works |
| AOSS config default pointed at a personal account | Use `AOSS_CONF` or `~/.petreloss.conf` |
| Reproduction script silently substituted duplicated model defaults when the registry failed | Resolve through the shared registry, report failure, support/deduplicate `default`, `last`, `all`, and integer layers |
| Rejected feature-sync candidates lost their diagnostic output when every candidate failed | Print nonempty invalid-candidate reports even when the accepted snapshot is empty |
| Historical consistency check accepted an incomplete paper export | Detect eligible ledger IDs missing from the export and add explicit `--require-final`; report the current 49-ID and 2,184-nonfinal-row gaps |

## Redundancy and tests

- Removed the unused `ProbeExtensionLitModule` validation `MeanMetric` copy and
  unused scheduler import; merged duplicated validation/test logging.
- Removed duplicated shell metadata fallbacks that could hide a broken
  installation or silently use stale layer/channel values.
- Removed one ineffective test that created random features without passing
  them to the function being tested. Causal-input and target-identity tests
  remain in place.
- Kept exact-target, cache identity, feature shape, frozen-protocol, and
  temporal-validity checks. These prevent known scientific errors and are not
  redundant merely because the failure is uncommon.
- Kept historical campaign tests as explicit `artifact` tests. Seven modules
  require frozen runtime copies or produced result artifacts; a separate
  frozen-inventory class also needs local data. They no longer break ordinary
  collection on a fresh checkout. See [the test guide](../tests/README.md).
- Made temporary test paths and schema fixtures independent of the author's
  working directory. Local synthetic shell/subprocess tests remain enabled.
- Moved the standalone root `tmp_ddp_smoke.py` into
  `scripts/diagnostics/ddp_smoke.py` with an explicit manual GPU usage guide;
  its communication behavior is unchanged and it is not a CI test.

Large dated submission/recovery script families were retained. Similar names
do not prove identical behavior: they encode different model, protocol,
window, retry, and storage contracts, and some are referenced by receipts.
They are operational history, not public quick-start commands. A later removal
should first produce a reference inventory and replacement mapping.

## Public source organization

The root README now provides the supported entrypoints, environment choices,
probe contracts, test commands, and a dated result index. The previous README
and status page are preserved under `archive/reports/`.

`PROJECT_STATUS.md` now distinguishes historical 1,260-task counts from newer
campaigns. The older dashboard is explicitly marked historical. No result
table was regenerated from stale logs or silently promoted to final status.

Python 3.11 is declared in package metadata. CPU CI installs a dedicated test
dependency set, disables model downloads, limits collection to this project's
tests, and validates public shell entrypoint syntax. `.gitignore` excludes
local runtime copies, model caches, the separate robotics workspace, temporary
test directories, credentials, and test/tool caches. The broad `temp**` rule
was replaced with narrow generated-file patterns so source files with names
such as `temporal_*` are not accidentally hidden.

## Results: implementation does not mean final completion

Read [the reconciled experiment inventory](RELEASE_READINESS_20261011.md)
and [per-experiment CSVs](../metadata/release_inventory/README.md) for current
evidence. The [earlier audit](RESULTS_STATUS_AUDIT_20261011.md) preserves the
historical ledger and superseded campaign snapshots.

- The historical ledger has 341 trained tasks without matched evaluation,
  43 raw-only metrics, 160 unresolved metrics, and 91 incomplete trainings.
- The old paper export has 2,184 metric rows for 576 experiments, all
  `POLICY_NOT_FROZEN`; 445 rows omit the split and 49 eligible experiments are
  missing from the export.
- Follow-up reconciliation verifies 696/736 canonical-v2 trainings; LingBot
  C40 remains unconfirmed, 28 completed Wan A2 runs have conflicting gate
  criteria, and formal metrics are not yet accepted. Timestep has 576
  completions and 216 matched independent evaluations, with missing
  C-series diagnostics and degenerate scores unresolved.
- C3-from-C1, Active Spatial MVP, SeeSE3 replication, the B2/P16 extension, and
  D-series unified exports have identified result gaps.
- Independent P8, continuous-action V2, SE(3) final r3, C1/B2, and D-series
  development packages already have results. Their validation/development
  boundaries must accompany release; a blind test is not a prerequisite for
  publishing honestly scoped development results.

Historical results are not retroactively repaired by code changes. In
particular, seed-zero training, selected checkpoint identity, invalid-sample
weighting, and validity metadata must be checked against each run's source
revision before interpreting its metrics.

## Remaining release blockers and verification limits

1. **Source attribution:** the README's unsupported MIT description was
   corrected to match the root CC BY-NC 4.0 declaration. Several vendored
   source trees reference licenses absent from their local roots. Recover the
   actual imported revisions' notices before redistribution; see
   [THIRD_PARTY.md](../THIRD_PARTY.md). No upstream license was guessed.
2. **Result index:** reconcile the missing eligible rows, split fields,
   checkpoint-selection policy, and newer campaign receipts. Keep unresolved
   experiments visibly incomplete. The final-result check currently fails by
   design because those scientific requirements are unmet.
3. **Runtime compatibility:** the global `torch.load` patch was removed using
   Lightning's public checkpoint-loading arguments. The private-method
   `ModelCheckpoint` save-last compatibility patch remains and needs a fresh
   check when changing Lightning. The tested CPU environment is pinned; the
   general research requirements are not a universal backend lock. Diagnostic
   `compile=true` is explicitly unsupported until checkpoint interoperability
   is implemented.
4. **Execution coverage:** this audit does not verify real multi-GPU reduction,
   CUDA extractors, all model backends, or private AOSS/cluster credentials.
   CPU schema and synthetic storage tests cannot substitute for those runs.
5. **Release contents:** the local evidence audit links some receipts under
   `artifacts/` and result packages that are not yet part of a reviewed release
   commit. Publish an intentional source/evidence selection, keep large inputs
   in external storage, and check links in that selected tree. The current
   working directory is not itself a release artifact.
6. **Runtime-only implementation coverage:** for example,
   `test_aether_streaming_temporal_mapping.py` imports
   `runtime/main_v2_probe_v2_aetherfix_r1/source/vidfm3d/utils/aether_temporal.py`;
   no matching helper exists at `vidfm3d/utils/aether_temporal.py` in the public
   source tree. Passing that artifact test certifies its frozen runtime copy,
   not the main extractor. Reconcile the intended canonical implementation
   before claiming the generic source reproduces that campaign. Frozen
   protocol changes were not silently copied into main during cleanup.

## Verification record

| Check | Outcome |
| --- | --- |
| Default CPU suite in a temporary source/config checkout, excluding root `artifacts/`, `runtime/`, `results/`, `logs/`, and `model_cache/` | **446 passed, 1 artifact case deselected**, 317.33 seconds; the seven artifact modules are excluded before collection |
| Final affected regressions after the A1 test-output naming fix, also removing copied `analyses/probe_validity/evidence/` and `paper/` | **66 passed**, 60.26 seconds: probe contracts, paper validity, metric accounting, and training regressions |
| Core regression imports/execution with PyTorch3D, Roma, Trimesh, and Open3D explicitly blocked | **30 passed**, including actual Lightning CPU validation/test |
| Sync integration regressions after restoring rejection diagnostics | **26 passed**, 313.50 seconds; no test assertions were weakened |
| Schema tests invoked outside the checkout working directory | **11 passed** |
| Offline-cache tests with unrelated higher-priority cache variables in the host environment | **3 passed** |
| Python AST parsing and Bash syntax checks | **714 Python files / 168 shell files passed**, including the relocated manual diagnostic |
| Public reproduction dry-runs | Layer aliases deduplicate correctly; invalid layers and a failed resolver exit before execution in serial and parallel modes |
| Maintained documentation links | **47 local links resolved**, no broken targets |
| `git diff --check` | Passed |
| Historical table consistency gate | Expected failure: **49 eligible experiment IDs are missing from the export** |
| `--require-final` gate | Expected failure: the same 49 IDs plus **2,184/2,184 exported rows not final-claim-ready** |

Tests used the existing Python 3.11.15 / PyTorch 2.6.0 / Lightning 2.6.1
environment with CPU execution, offline Hub mode, and disabled W&B. The
source-copy check verifies independence from the named local artifact roots;
it is not a newly installed dependency environment. Dependency pins were
checked against installed versions, but installation in an empty environment
and the hosted GitHub Actions run have not been executed here. Existing
deprecation and small-dataloader warnings did not fail the tests. GPU model
extraction and actual distributed training remain outside this verification.
