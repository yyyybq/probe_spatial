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

## Release tests and internal tests (2026-10-18)

The repository tracks only release tests: tests of the training, data, model and
evaluation code (`vidfm3d/`), the feature-extraction contracts (`features/`), the
timestep manifest protocol and the D-series code. They run on a clean checkout with
`python -m pytest -q`, without campaign artifacts or extra `PYTHONPATH` entries.

Tests of cluster submission, upload and recovery, extraction campaigns (windowfix,
timestep, Wan target), the 816 overlay audit and the unreleased analyses (SE3,
active representation, probe validity) are internal. They are not tracked; they stay
next to the internal tools they test in the working checkout, and are listed in
`.gitignore`. Earlier commits of the 2026-10 history still contain them.
