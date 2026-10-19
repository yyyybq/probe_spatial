# Q00 loose all-results table (frozen 1096)

This is a display-first table. It retains canonical, raw historical, provisional, and legacy evidence even when lineage metadata warns that a value is not scientifically comparable. It does not modify the registry or canonical tables.

## Counts

- TOTAL_ROWS: 1096
- FINITE_SELECTED_RESULTS: 897
- CANONICAL_RESULTS: 642
- RAW_HISTORICAL_RESULTS: 31
- PROVISIONAL_RESULTS: 160
- LEGACY_RESULTS: 0
- LOOSE_REEVAL_RESULTS: 64
- NO_NUMERIC: 199
- KNOWN_MISMATCH_RESULTS: 668
- KNOWN_MANIFEST_MISMATCH_RESULTS: 332
- HISTORICAL_VAL_SUMMARY_PATHS_READABLE: 592
- HISTORICAL_VAL_SUMMARY_PRIMARY_CANDIDATES: 495
- LEGACY_CANDIDATES_AVAILABLE: 30

## Selection and safeguards

Selection is per frozen parent and uses only exact identity bridges: canonical atomic IDs from reconciliation, exact default-B2 summary/run paths, Wan/Cog lineage `old_experiment_id`, and exact legacy `universe_id` plus full identity. The source order is canonical finite → raw historical summary → historical provisional → legacy packaged → loose reevaluation → NA.

T0 is an evidence-source tier rather than an endorsement of the result's scientific validity. `known_label_feature_mismatch`, `known_manifest_mismatch`, `lineage_status`, `scientifically_comparable`, and `warning` make the existing limitations explicit. Collision candidates remain separate JSON records and are never pooled.

The table has 30 rows with a legacy B2 candidate, but T1 raw summaries outrank all of them; therefore LEGACY_RESULTS can be zero while legacy evidence remains visible.
It also contains 64 finite T4 legacy-as-is evaluations. These are display-only values and retain their mismatch warnings; they do not alter canonical validity or scientific comparability.

## Outputs

- `results/analysis/q00_1096_loose/all_results_1096_loose.csv`
- `results/analysis/q00_1096_loose/all_results_1096_loose.parquet`
- `results/analysis/q00_1096_loose/missing_numeric.csv`

## Frozen input hashes

- `canonical_all`: `2a04a5e1f0165f1d7a06e7edcff38d956b845910ad95af3f7bbede3704133b17`
- `default_b2_raw`: `1f5a9b45afe554c2afb0fe82db4ffe004b04309af969dac71736ce9c6dca77c3`
- `legacy_b2`: `1bb77ba69ece843725ef5a5db1b0d8feb1f83c0ca12eb2930baa2a094345159d`
- `legacy_b2_namespace`: `83c0ed31669bbdf249836377f9fbc2ce01a1a4f0f49d14daabf909b67d9ebe99`
- `phase1_live`: `55eb8a2c9dbd9e0a72bdfe2742e41a7a2cbf3543b3d9e8610c4ffb955cad695f`
- `reconciliation`: `3cda065533eb060e63cf01d3a7ab8252a081fcd3830d6bdbb1f51b6f1d9a4842`
- `registry`: `efc533bbe6286090d6c111c9d272ffb28cf7477b937f799f312eedf2a944431e`
- `universe`: `9c7d6807b701de4b052819971a5c011e29c717e1c3e958ec9ae23116cdcfe779`
- `wan_cog_lineage`: `e470af537741da5c7128256321b72ed069d0e371b97b58ed70c677c3cc9e5d32`
- `window_audit`: `e19401ca83009e2d0233fad970bac0f35fa4158768b7a18cb039686a703f21d7`
