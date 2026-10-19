# SE3-Observed-v1 dataset card

Status: **legacy development / not independent confirmation**.

The only fully routed, cross-model, artifact-embedded target-isolated inventory contains 276 unique physical triples from 69 scenes. The frozen split has 192/40/44 triples, 48/10/11 scenes, and 576/120/132 unique scene-frame keys for train/validation/development-test. All 69 scenes were already present in the r2 pilot, so none is called confirmation. Exact scene IDs are in `split.json`.

The manifest is `se3_observed_v1_manifest.jsonl` (SHA-256 `5116871268a80fddc205ff0901f495fc4d05ab5f08ba33bf1329356648ac6b0e`). It separates physical keys from model/context feature keys; records raw indices and names, all c2w/w2c/intrinsics and relative transforms, ordered repeated-frame inputs, temporal gaps, feature/source hashes and contracts, motion/non-commutativity, and observed overlap. The source metadata has no timestamp field, so timestamps are explicitly null rather than inferred from filenames.

Selection is the immutable four-model intersection of the historical C1/C2 p8 validation dumps. There are no duplicate physical triples. Synthetic-nearest endpoints are absent. Splits are scene-disjoint, and the nested development test is held out from this runner's probe training, but it is not scientifically untouched because its scores appeared in the earlier pilot.

The canonical-v2 feature preparation tree was inspected but not consumed: its handoff forbids probing until routing/merge completion. The nominal master test is also not independent of historical C training. Therefore no legal old-69-scenes-excluded confirmation set can be formed from the ready inventory. Requested 4,000/1,000/1,000 counts and 50 new test scenes are unsupported rather than fabricated.
