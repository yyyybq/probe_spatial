# Final integrity audit

- Required final report/table files: present.
- Python compilation: passed for every changed Python file in this run.
- Tests: 27 passed (`tests/test_probe_contracts.py` + `tests/test_paper_validity.py`); 8 non-fatal Transformer nested-tensor warnings.
- `git diff --check`: passed.
- Status vocabulary: valid; 160 `COMPLETE_WITH_CAVEAT`, 832 `BLOCKED_MISSING_ARTIFACT`, and zero unsupported `COMPLETE_VALID` claims.
- Selected inputs: 64/64 C checkpoints and 64/64 prediction dumps exist and have SHA256 recorded.
- Historical provenance: 0/64 selected C run manifests contain both a known git commit and split-manifest hash; this is why all numeric rows remain caveated.
- Common sample audit: 128/128 C model×prefix×horizon rows have native=common=276 and dropped=0.
- Protected files: `experiments/registry.csv` and all seven canonical result CSVs retain the pre-audit SHA256 values recorded in the existing validity report.
- New checkpoints: none. New strict JSONL manifests: none (the zero-valid/data gate blocked before real generation).
- Local processes matching strict builder, validity runner, training, or eval: none after completion.
- Scheduler: concurrent jobs owned by `guchenyang` were observed, including unrelated Wan exact reruns and separately named `bq_probe_validity_*` jobs. None carries this run tag; none was launched, consumed, or reported complete by this run.
- Background jobs launched by this run: none.
