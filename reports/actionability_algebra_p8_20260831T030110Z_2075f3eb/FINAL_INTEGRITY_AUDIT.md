# Final integrity audit — `actionability_algebra_p8_20260831T030110Z_2075f3eb`

## Verdict

Artifact integrity: **PASS**. Scientific completion: **BLOCKED**, not complete. No model metric was computed or promoted.

- Required/nonzero files: 42/42.
- Missing/zero files: [].
- Status counts: `{"BLOCKED_ENVIRONMENT": 1, "BLOCKED_MISSING_ARTIFACT": 9, "COMPLETE_VALID": 1}`.
- Manifest records are explicit `BLOCKER` metadata (`is_scientific_sample=false`), not fabricated samples.
- Result estimates/CIs are blank at n=0; plots say NOT RUN.
- No backbone extraction, probe training, checkpoint loading, or prediction was performed.
- No background process was launched by this run.

## Protected canonical files

| Path | Expected SHA256 | Current SHA256 | Unchanged |
|---|---|---|---|
| `experiments/registry.csv` | `efc533bbe6286090d6c111c9d272ffb28cf7477b937f799f312eedf2a944431e` | `efc533bbe6286090d6c111c9d272ffb28cf7477b937f799f312eedf2a944431e` | True |
| `results/canonical/a1_results.csv` | `f7777676eafbc77dc8a8856e86a0901c98d6b5b3bdaddb568a975ef1c9f94d86` | `f7777676eafbc77dc8a8856e86a0901c98d6b5b3bdaddb568a975ef1c9f94d86` | True |
| `results/canonical/all_results.csv` | `2a04a5e1f0165f1d7a06e7edcff38d956b845910ad95af3f7bbede3704133b17` | `2a04a5e1f0165f1d7a06e7edcff38d956b845910ad95af3f7bbede3704133b17` | True |
| `results/canonical/b2_results.csv` | `0d4941709af83ba72820e4c463b96775d240096b44fca14bd4c5243e2dfa5f20` | `0d4941709af83ba72820e4c463b96775d240096b44fca14bd4c5243e2dfa5f20` | True |
| `results/canonical/c1_results.csv` | `b88a37e239d150d297a4e97838d93888d5b6d3d0780df510e65b7f916b2ade1a` | `b88a37e239d150d297a4e97838d93888d5b6d3d0780df510e65b7f916b2ade1a` | True |
| `results/canonical/c2_results.csv` | `8c96a38500420af11427ed506bc92d4e1fc9d6b636bc34f0c4baf2d279fd9524` | `8c96a38500420af11427ed506bc92d4e1fc9d6b636bc34f0c4baf2d279fd9524` | True |
| `results/canonical/c3_results.csv` | `6cb2bec11abecb7cdf7549f14db84d58345e2bb5e9a0172f9435c80e5e2b7c5d` | `6cb2bec11abecb7cdf7549f14db84d58345e2bb5e9a0172f9435c80e5e2b7c5d` | True |
| `results/canonical/a2_results.csv` | `2613650bc7dcaa7a5bb5292c1953b9742657fbcb7967da3016b8c84c9f6ffa5f` | `2613650bc7dcaa7a5bb5292c1953b9742657fbcb7967da3016b8c84c9f6ffa5f` | True |

## Artifact inventory

| Path | Bytes | SHA256 |
|---|---:|---|
| `artifacts/actionability_algebra_p8_20260831T030110Z_2075f3eb/action_algebra_manifest_common.jsonl` | 331 | `b8f8cf9ff1075c31cda5bca15c2a88f12117198ac17e7d69c669e70577f7c10d` |
| `artifacts/actionability_algebra_p8_20260831T030110Z_2075f3eb/action_convention_test_results.csv` | 450 | `5f0a3e8ee31fe22a7d76984ad9b3fcb94509570fa18010442e2163db7f9f8f57` |
| `artifacts/actionability_algebra_p8_20260831T030110Z_2075f3eb/action_convention_test_results.json` | 1535 | `9e55371b1459ab296aa696c3a459b064e57182bbf3aa30786110ab42db35c08b` |
| `artifacts/actionability_algebra_p8_20260831T030110Z_2075f3eb/action_convention_tests.py` | 5945 | `dc678b63733df5ca91500d008c7c5f4eb868aab9f72fde8018d33c7bd499096c` |
| `artifacts/actionability_algebra_p8_20260831T030110Z_2075f3eb/actionability_manifest_all.jsonl` | 327 | `b8d37875b0c9d0fb05146bcaa69e3655fbbd06cc8eb407d51944efbded918fa5` |
| `artifacts/actionability_algebra_p8_20260831T030110Z_2075f3eb/actionability_manifest_common.jsonl` | 330 | `6c95c6bf24aa42bdfeba93d097d5b9de41b1e7e17afa5ffc8896399a357b0c03` |
| `artifacts/actionability_algebra_p8_20260831T030110Z_2075f3eb/build_deliverables.py` | 35643 | `d38d9853797e8445a7cb9cda99bec2e41c9445bc8c45a61543cf3d641225e862` |
| `figures/actionability_algebra_p8_20260831T030110Z_2075f3eb/actionability_model_comparison.csv` | 159 | `60ce4ad207f968db5f562a93fa5dd6714c589a7f57a544458c5987015ed47582` |
| `figures/actionability_algebra_p8_20260831T030110Z_2075f3eb/actionability_model_comparison.pdf` | 16176 | `0d6fe990e5e2ab09cc66de184f3cb8bd0b80e7aecdfeb4a340c8481dc72f7722` |
| `figures/actionability_algebra_p8_20260831T030110Z_2075f3eb/actionability_model_comparison.png` | 52254 | `a7fae73baef317b1c510eed3df535cfc0212f4e39af5cd8faca19adc9e4a7f22` |
| `figures/actionability_algebra_p8_20260831T030110Z_2075f3eb/actionability_regret_by_difficulty.csv` | 159 | `60ce4ad207f968db5f562a93fa5dd6714c589a7f57a544458c5987015ed47582` |
| `figures/actionability_algebra_p8_20260831T030110Z_2075f3eb/actionability_regret_by_difficulty.pdf` | 16195 | `94227dced12098db4be0ab6e499c73e63adfcb1b040f61ff4f7bdf33def7bf05` |
| `figures/actionability_algebra_p8_20260831T030110Z_2075f3eb/actionability_regret_by_difficulty.png` | 53164 | `f93269b3936b0a320f98b5835f1dee6f74e1fec69e076e11dac9346342d64b15` |
| `figures/actionability_algebra_p8_20260831T030110Z_2075f3eb/composition_gap.csv` | 159 | `60ce4ad207f968db5f562a93fa5dd6714c589a7f57a544458c5987015ed47582` |
| `figures/actionability_algebra_p8_20260831T030110Z_2075f3eb/composition_gap.pdf` | 16536 | `5710dc57dfad54f225bfa9dfd429a38a6345adad461ed0dfb80792c5224bcb50` |
| `figures/actionability_algebra_p8_20260831T030110Z_2075f3eb/composition_gap.png` | 56475 | `bebdb4be46c099667932b6547c0d72209700772d0e98801d40e058447d37d7d1` |
| `figures/actionability_algebra_p8_20260831T030110Z_2075f3eb/cycle_drift.csv` | 159 | `60ce4ad207f968db5f562a93fa5dd6714c589a7f57a544458c5987015ed47582` |
| `figures/actionability_algebra_p8_20260831T030110Z_2075f3eb/cycle_drift.pdf` | 16138 | `3972ddd716b287523b199b75f01667dab9a40df81d8a8f24291343c1c160741e` |
| `figures/actionability_algebra_p8_20260831T030110Z_2075f3eb/cycle_drift.png` | 51949 | `2f0b86674481e34baecec80529422e1e190bceaf887c72cba78aba97dd9b79a9` |
| `figures/actionability_algebra_p8_20260831T030110Z_2075f3eb/loop_drift_vs_path_length.csv` | 159 | `60ce4ad207f968db5f562a93fa5dd6714c589a7f57a544458c5987015ed47582` |
| `figures/actionability_algebra_p8_20260831T030110Z_2075f3eb/loop_drift_vs_path_length.pdf` | 16087 | `09868c72128202e65a0886d544322909db1d3686550ad578386fff7e0c89fd58` |
| `figures/actionability_algebra_p8_20260831T030110Z_2075f3eb/loop_drift_vs_path_length.png` | 53821 | `e0fab7118b96df8d0286b4e67f8f93bd74671f02290adf9d3c93c82fa7908fa4` |
| `figures/actionability_algebra_p8_20260831T030110Z_2075f3eb/predictor_vs_direct_rank.csv` | 159 | `60ce4ad207f968db5f562a93fa5dd6714c589a7f57a544458c5987015ed47582` |
| `figures/actionability_algebra_p8_20260831T030110Z_2075f3eb/predictor_vs_direct_rank.pdf` | 15730 | `2011e8b8923e9224e0e753fd5ad8145ede9705d25a842a72e8e398305789f0ff` |
| `figures/actionability_algebra_p8_20260831T030110Z_2075f3eb/predictor_vs_direct_rank.png` | 53090 | `658301210c48261ac6a89cd0f72c2a6d7dc43f3fdfd738aecd9951c9ef411c33` |
| `logs/actionability_algebra_p8_20260831T030110Z_2075f3eb/action_convention_tests.log` | 1231 | `5fa27f5951d6853e55ddb60d57e743d939c69658a46bcf562d135641d5b341b1` |
| `logs/actionability_algebra_p8_20260831T030110Z_2075f3eb/dino_smoke_preflight.json` | 736 | `83f1b17ef2e731df30d679b066ecb2fb503237ce5d0ae054e99b0803a39c4dbf` |
| `logs/actionability_algebra_p8_20260831T030110Z_2075f3eb/environment_preflight.json` | 302 | `42c6d394c91c54fac3f25329c2719d09c5aeafad4f94c47c0f571a4a5bdc6528` |
| `reports/actionability_algebra_p8_20260831T030110Z_2075f3eb/00_AUDIT.md` | 3334 | `fe010f85a172b6febb69afd1a3cfe22ce5c6c04e08636f32220bfc61e65f94e7` |
| `reports/actionability_algebra_p8_20260831T030110Z_2075f3eb/ACTIONABILITY_REPORT.md` | 1874 | `2466289f6858bf982712d2fd23f35b03b1f0ab6ba958b4617c6d89f3460e4950` |
| `reports/actionability_algebra_p8_20260831T030110Z_2075f3eb/ACTION_ALGEBRA_REPORT.md` | 1469 | `32ea6dceee2e0aafcb411afbcb1414e32d21d68ef085c9a21980a4b49f8dc483` |
| `reports/actionability_algebra_p8_20260831T030110Z_2075f3eb/ACTION_CONVENTION_AUDIT.md` | 1809 | `9ba3237b230aa711ed028a8a2a44cda903a8a14f2cf9887711d753afcb32ce22` |
| `reports/actionability_algebra_p8_20260831T030110Z_2075f3eb/CHECKPOINT_INVENTORY.csv` | 16508 | `49a1340c6841c9fbf3eefa86d1b6310dca53ddce537811fd5fa530b93b247efc` |
| `reports/actionability_algebra_p8_20260831T030110Z_2075f3eb/FEATURE_INVENTORY.csv` | 6393 | `8bbf41d46a3538b29790e227681fd81a30eb71968bab070e38ff8a55aa1dd1c1` |
| `reports/actionability_algebra_p8_20260831T030110Z_2075f3eb/INTEGRATED_ACTION_STRUCTURE_REPORT.md` | 1350 | `514153244d742a5ec1641bb4cb9b1e245b84a90c73fdeae6d6075f5ee5501045` |
| `reports/actionability_algebra_p8_20260831T030110Z_2075f3eb/STATUS_MATRIX.csv` | 3560 | `bcc8dda3374f826a51b46017da2da38ca772d46606fec77234b3745c5355dda2` |
| `results/actionability_algebra_p8_20260831T030110Z_2075f3eb/action_algebra_bootstrap.csv` | 603 | `53d6767c929a21ee0359f3cc9674e6451860574510b412b5f1f7929e239a1f87` |
| `results/actionability_algebra_p8_20260831T030110Z_2075f3eb/action_algebra_per_sample.csv` | 603 | `53d6767c929a21ee0359f3cc9674e6451860574510b412b5f1f7929e239a1f87` |
| `results/actionability_algebra_p8_20260831T030110Z_2075f3eb/action_algebra_summary.csv` | 603 | `53d6767c929a21ee0359f3cc9674e6451860574510b412b5f1f7929e239a1f87` |
| `results/actionability_algebra_p8_20260831T030110Z_2075f3eb/actionability_bootstrap.csv` | 599 | `0daa4dc0b723dfd92dd22c870bdf12d5cd5670a5fd95f97e3c4836ccf5a21059` |
| `results/actionability_algebra_p8_20260831T030110Z_2075f3eb/actionability_per_sample.csv` | 599 | `0daa4dc0b723dfd92dd22c870bdf12d5cd5670a5fd95f97e3c4836ccf5a21059` |
| `results/actionability_algebra_p8_20260831T030110Z_2075f3eb/actionability_summary.csv` | 599 | `0daa4dc0b723dfd92dd22c870bdf12d5cd5670a5fd95f97e3c4836ccf5a21059` |

## Completion-standard audit

The requested four-model pilots do not meet the scientific completion standard because exact fixed-setting inputs/checkpoints and common sample IDs are absent. Convention tests, output isolation, blocker evidence, status vocabulary, and non-overwrite checks do meet their audit requirements. This distinction is intentional.
