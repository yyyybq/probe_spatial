# Tests

Run `python -m pytest -q` from the checkout, using Python 3.11 and
`requirements-dev.txt`. The default collection is limited to `tests/`; it does
not traverse downloaded models, vendored examples, runtime copies, or other
projects stored next to this repository.

| Selection | Command | Inputs |
| --- | --- | --- |
| Default regression suite | `python -m pytest -q` | CPU tensors and temporary synthetic files |
| Local workflow tests | `python -m pytest -m integration` | Synthetic subprocess/shell workflows; no scheduler submission |
| Historical artifact checks | `python -m pytest --run-artifact-tests -m artifact` | Original frozen runtime copies, manifests, inventories, and result packages |

`tests/conftest.py` lists modules that cannot even be imported without local
campaign artifacts. They are excluded before import unless explicitly enabled.
They remain available for provenance validation. A failing portable test must
be repaired or reported; it must not be reclassified merely to make CI pass.

The Timestep sync integration tests use the system temporary directory by
default. `PROBE_TEST_TMPDIR` can select a particular mounted filesystem when
reproducing its behavior is the purpose of the test.

The 2026-10-11 audit removed one ineffective assertion that created unrelated
random VFM features, never supplied those features to the label computation,
and recomputed identical label inputs. Behavioral tests for future-pixel
independence, masks, exact target identities, and causal feature contracts were
retained. Source-text assertions in historical workflow tests remain useful as
configuration checks, but cannot certify successful GPU or cluster execution.

The 2026-10-15 audit added `test_nonfinite_handling.py`: NaN/inf feature
batches must abort training (`NonFiniteInputError`), both evaluators must flag
non-finite predictions and exclude them from means instead of scoring them as
zero loss or bin 0, A2 summaries must use the module's overlap thresholds, and
the cache-name registry must produce `feature_t250_layer{L}.sft` for
MiniMax-H3/LingBot. Run it alone with
`python -m pytest -q tests/test_nonfinite_handling.py`.

## Release tests and internal tests (2026-10-18, updated 2026-10-19)

Release tests cover the training, data, model and evaluation code (`vidfm3d/`),
the feature-extraction contracts (`features/`), the timestep manifest protocol,
the D-series code and the MiniMax bf16 stage-5 tooling. They run on a clean
checkout with `python -m pytest -q`, without campaign artifacts or extra
`PYTHONPATH` entries.

Internal tests cover cluster submission, upload and recovery, extraction
campaigns (windowfix, timestep, Wan target), the 816 overlay audit and the
unreleased analyses (SE3, active representation, probe validity). They were
untracked on 2026-10-18 and tracked again by commit `8a8948d` (2026-10-19) as
provenance. Several import frozen `runtime/` copies, read untracked artifacts
or call the cluster Python, so `tests/conftest.py` (`INTERNAL_MODULES`, the same
list as the "Internal campaign tests" block of `.gitignore`) excludes them
unless `--run-artifact-tests` is given. Run them on the cluster checkout where
those files exist.

Clean-checkout results, Python 3.13 with `requirements-dev.txt` (2026-10-19,
CPU; 2026-10-18 reported 198 passed for the 25 release files alone):

| Selection | Result |
| --- | --- |
| `python -m pytest -q` (29 tracked release modules) | 221 passed, 1 skipped (`pytorch3d` not installed) |
| `python -m pytest -q --run-artifact-tests` without cluster artifacts | 2 collection errors (missing `runtime/` copies); further failures come from untracked `runtime/`, `artifacts/`, `results/analysis/` files, two untracked `scripts/windowfix_v2/*.schema.json` files, and the cluster Python path |
