# Wan/Cog Lineage Resolution

## Summary

- historical rows reviewed: 213
- provisional historical rows reviewed: 162
- supersession confirmed: 0
- formal provenance confirmed: 51
- duplicates: 0
- provisional-only: 162
- conflicting: 0
- ambiguous: 0
- missing artifacts: 0

## Confirmed Chains

- None. No old -> new replacement relation met the evidence threshold in this pass.

## Still Unresolved

- PROVISIONAL_ONLY_NO_REPLACEMENT: 162
- `PROVISIONAL_ONLY_NO_REPLACEMENT`: parseable historical metrics exist, but the source marks them provisional and `experiments/supersession_candidate.csv` records no newer completed rerun artifact.

## Recommended Future Actions

- keep isolated: all 162 provisional Wan/Cog rows until a frozen protocol or explicit rerun evidence exists.
- archive duplicate: no duplicate same-result rows were found in this pass.
- human review: required only if later evidence claims a specific old -> new replacement.
- later frozen re-evaluation: needed before turning noncanonical Wan/Cog rows into paper claims.
- candidate for future canonicalization: the 3 formal raw-only rows have provenance confirmed but were not canonicalized in this scoped pass.

Machine-readable ledger: `experiments/wan_cog_lineage_resolution.csv`.
