# Q01 global feature semantic lineage audit

生成日期：2026-08-29（UTC）  
模式：严格只读证据审计；未训练、未恢复训练、未提取 feature、未 eval、未提交 SCO/SSH 作业、未修改 registry/canonical/feature root。

## 结论

Phase 0 同时通过：负控复现 streaming mismatch `1029/2724`、target mismatch `948/2724`；正控复现两种 role 均 `2724/2724` exact、mismatch `0`，冻结 Dataset/DataLoader gate 为六层各 `2724/2724`、fallback `0`。

冻结 1096 parent 全部映射到 522 条去重 lineage。只有 manifest-exact Wan p8 versioned root 可给 exact；共享 AOSS root 因代理不可达而没有逐样本 actual-ID 读取证据，显式 manifest 的条目最多给 provenance-only。历史 manifest 缺失时没有 fallback 到 canonical。

642 canonical：exact-safe 6，provenance-only 71，eval-risk 0，checkpoint-risk 0，both-risk 0，unauditable 565。现有 canonical 仅生成 ledger，未改动。

99 eval-only：allowed 0，blocked mismatch 0，blocked manifest 0，blocked unauditable 99。本轮没有提交任何 eval。

## Phase 0：审计器正/负控

| control | streaming exact | streaming mismatch | target exact | target mismatch | Dataset load | fallback |
|---|---:|---:|---:|---:|---:|---:|
| known negative `wan_p8_action_dynamics_canonical_20260820_025949_2075f3e` | 1695/2724 | 1029 | 1776/2724 | 948 | n/a | n/a |
| known positive `wan_p8_action_dynamics_canonical_manifest_exact_20260821_200604_2075f3e` | 2724/2724 | 0 | 2724/2724 | 0 | 2724/2724 × 6 layers | 0 |

负控逐样本例证保存在 `known_control_validation.json`；第一例 `00777c41d4_w001_p8` 的 manifest streaming raw IDs 为 `[295,300,306,311,314,317,323,327]`，实际 index 为 `[332,335,337,340,343,345,348,351]`。负控源证据是 `streaming_alignment_audit.json` 与 `target_raw_mismatch_summary.json`。

正控直接比较 manifest 显式 raw IDs 与 versioned root 的 `prefix_index.npy.actual_input_raw_ids` / `target_indices.npy`。冻结 `full_dataset_dryrun.json` 的 SHA256 与 gate contract 完全一致；其 Dataset code run-time hash为 `a5c6094a...`，当前工作树 Dataset hash不同，因此没有用当前脏代码冒充历史 canary。

## Lineage 恢复与语义规则

Lineage key 包含 model、model_variant、role、root realpath/URI、manifest SHA（或 missing identity）、extractor mode/config hash、snapshot、feature layer/cache group、prefix、timestep、source。campaign/model_variant/full identity 始终保留；Q00 的 21 个 collision groups / 42 registry IDs 没有合并，并在 atomic ledger 中按单一 registry ID 显式回映所属 group。

证据优先级为：checkpoint/raw-summary 实际 run 的 `run_manifest.json` 与 Hydra config → train/eval log 的 manifest load 记录 → eval job TSV → frozen Q00/task ledger。eval-only 的 manifest 只在 Wave B 计划中复用其实际 training manifest，并明确标为 `PLANNED_NOT_RUN`；历史 eval 不做此推断。

Expected IDs 仅来自 JSONL 的 `streaming_indices`、显式 target raw 字段或 `streaming_window_indices[streaming_tail+1]`；没有调用 generic window generator。Actual IDs 仅接受 index/sidecar/sample map。路径、`window_XXXX` 名称或 feature tensor 存在不构成 raw identity 证明。

共享 AOSS index 只读探测在沙箱外也因 `ProxyConnectionError` 失败。因此这些 cache 即使有历史消费记录也不记 mismatch=0；有显式 manifest 和一致 run/root 的记 `NO_MISMATCH_FOUND_PROVENANCE_ONLY`，没有历史 manifest 的记 `MANIFEST_MISSING`。

## 642 canonical 影响

| impact | parents | atomic canonical IDs |
|---|---:|---:|
| CANONICAL_PROVEN_SAFE | 6 | 6 |
| CANONICAL_PROVENANCE_ONLY | 71 | 87 |
| CANONICAL_EVAL_AT_RISK | 0 | 0 |
| CANONICAL_CHECKPOINT_AT_RISK | 0 | 0 |
| CANONICAL_BOTH_AT_RISK | 0 | 0 |
| CANONICAL_UNAUDITABLE | 565 | 641 |

`CANONICAL_PROVENANCE_ONLY` 不等于 exact pass；它只表示未发现互相矛盾的 run/root/manifest provenance。`CANONICAL_UNAUDITABLE` 也不等于 proven invalid，主要边界是历史 expanded manifest 缺失或未绑定。

## 99 eval-only gate

| gate | count |
|---|---:|
| EVAL_ALLOWED | 0 |
| EVAL_BLOCKED_FEATURE_MISMATCH | 0 |
| EVAL_BLOCKED_MANIFEST | 0 |
| EVAL_BLOCKED_UNAUDITABLE | 99 |

Gate 是 fail-closed：provenance-only cache 没有逐样本 Dataset canary，故归入 `EVAL_BLOCKED_UNAUDITABLE`；不能据此提交 Wave B。

## 完整 1096 与 remediation

| remediation | parent count |
|---|---:|
| NO_ACTION | 77 |
| REEVAL_ONLY | 99 |
| CLEAN_RETRAIN_AND_EVAL | 58 |
| REEXTRACT_THEN_RETRAIN_AND_EVAL | 35 |
| RECONSTRUCT_MANIFEST_THEN_DECIDE | 582 |
| PERMANENTLY_UNAUDITABLE | 245 |

已知 VideoMAE p24 hole 去重为 5 个 cache/layer lineage，影响 35 个 parents、0 个 frozen atomic canonical IDs。按冻结 Q00 wave 计划是 2 个 feature jobs（canary + shared full）；当前 gate 下确定性的 train/eval 计划分别为 93/192，但本报告不授权也不执行任何 job。

全局有 1090 个 parents / 728 个 frozen atomic canonical IDs 未达到 exact-safe；其中 1019 个 parents / 641 个 atomic IDs 有非 `NO_ACTION` 计划。manifest 或 index 证据缺口导致无法完成语义审计的配置为 777。

无 mismatch 证据且 checkpoint 可保留的 parents 为 227（其中 provenance-only 不升级为 exact）。需要 clean retrain 的确定性 parents 为 93；只需 eval 的配置为 99，但 99 个 eval-only 仍须先解除 raw-ID canary gate；永久无法闭合 raw-ID 证据的配置为 245。

缺失的 historical filtered manifest 只进入 `RECONSTRUCT_MANIFEST_THEN_DECIDE`；报告没有创建 reconstructed manifest，也没有使用 canonical 替代。无路径/无唯一约束的 legacy manifest 保持 permanently unauditable。

## Evidence 与产物

- `feature_lineage_inventory.csv`：逐唯一 role/cache/manifest lineage，actual/mismatch 未知时留空而不是写 0。
- `canonical_642_impact.csv`：642 parent 风险 ledger。
- `eval_only_99_gate.csv`：99 个待 eval gate。
- `universe_1096_feature_semantics.csv`：1096 full identity 与 train/eval 状态。
- `atomic_canonical_impact.csv`：734 个映射回 frozen universe 的 atomic canonical IDs，含 21 个 collision groups / 42 IDs 的单独 group 字段；27 个 post-freeze IDs 不在本轮 universe。
- `remediation_queue.csv`：按 lineage + remediation 去重，`execution_authorized=false`。
- `known_control_validation.json` 与 `summary.json`：control、hash、计数、限制。

关键 Dataset contract 在 `vidfm3d/data/components/inscene15k_dataset.py`：streaming 使用 `prefix_index.npy`，isolated target 使用 `target_indices.npy`；Wan exact verifier 还要求 sidecar raw IDs 与 Dataset 选取 tensor 一致。每条 CSV 的 `evidence_paths` 指向实际 manifest/root/index pattern/config/log/checkpoint/summary。

## 受保护文件完整性

| file | before SHA256 | after SHA256 | unchanged |
|---|---|---|---|
| `experiments/registry.csv` | `efc533bbe6286090d6c111c9d272ffb28cf7477b937f799f312eedf2a944431e` | `efc533bbe6286090d6c111c9d272ffb28cf7477b937f799f312eedf2a944431e` | true |
| `results/canonical/a1_results.csv` | `f7777676eafbc77dc8a8856e86a0901c98d6b5b3bdaddb568a975ef1c9f94d86` | `f7777676eafbc77dc8a8856e86a0901c98d6b5b3bdaddb568a975ef1c9f94d86` | true |
| `results/canonical/a2_results.csv` | `2613650bc7dcaa7a5bb5292c1953b9742657fbcb7967da3016b8c84c9f6ffa5f` | `2613650bc7dcaa7a5bb5292c1953b9742657fbcb7967da3016b8c84c9f6ffa5f` | true |
| `results/canonical/all_results.csv` | `2a04a5e1f0165f1d7a06e7edcff38d956b845910ad95af3f7bbede3704133b17` | `2a04a5e1f0165f1d7a06e7edcff38d956b845910ad95af3f7bbede3704133b17` | true |
| `results/canonical/b2_results.csv` | `0d4941709af83ba72820e4c463b96775d240096b44fca14bd4c5243e2dfa5f20` | `0d4941709af83ba72820e4c463b96775d240096b44fca14bd4c5243e2dfa5f20` | true |
| `results/canonical/c1_results.csv` | `b88a37e239d150d297a4e97838d93888d5b6d3d0780df510e65b7f916b2ade1a` | `b88a37e239d150d297a4e97838d93888d5b6d3d0780df510e65b7f916b2ade1a` | true |
| `results/canonical/c2_results.csv` | `8c96a38500420af11427ed506bc92d4e1fc9d6b636bc34f0c4baf2d279fd9524` | `8c96a38500420af11427ed506bc92d4e1fc9d6b636bc34f0c4baf2d279fd9524` | true |
| `results/canonical/c3_results.csv` | `6cb2bec11abecb7cdf7549f14db84d58345e2bb5e9a0172f9435c80e5e2b7c5d` | `6cb2bec11abecb7cdf7549f14db84d58345e2bb5e9a0172f9435c80e5e2b7c5d` | true |
| `experiments/task_status_1096_before_expand_20260822.csv` | `9c7d6807b701de4b052819971a5c011e29c717e1c3e958ec9ae23116cdcfe779` | `9c7d6807b701de4b052819971a5c011e29c717e1c3e958ec9ae23116cdcfe779` | true |
| `results/analysis/q00_1096/universe_1096_reconciled.csv` | `3cda065533eb060e63cf01d3a7ab8252a081fcd3830d6bdbb1f51b6f1d9a4842` | `3cda065533eb060e63cf01d3a7ab8252a081fcd3830d6bdbb1f51b6f1d9a4842` | true |
| `results/analysis/q00_1096/execution_waves.csv` | `0369a6295fb7fc233e89e11e72b802b1977e6af7b230d9da80cce4265bea947f` | `0369a6295fb7fc233e89e11e72b802b1977e6af7b230d9da80cce4265bea947f` | true |
| `reports/q00_universe_reconciliation_1096.md` | `529594a6bda0b96fc27744abdbabcec14a23e0976270bc85ef7bb1dd318dd76e` | `529594a6bda0b96fc27744abdbabcec14a23e0976270bc85ef7bb1dd318dd76e` | true |

## 硬性验收

```text
KNOWN_NEGATIVE_REPRODUCED: true
KNOWN_POSITIVE_REPRODUCED: true
UNIVERSE_ROWS: 1096
CANONICAL_PARENT_ROWS: 642
EVAL_ONLY_ROWS: 99
PARENTS_MAPPED: 1096
UNIQUE_FEATURE_LINEAGES: 522
LINEAGES_PROVEN_SAFE_EXACT: 2
LINEAGES_PROVENANCE_ONLY: 246
LINEAGES_PROVEN_MISMATCH: 0
LINEAGES_UNAUDITABLE: 1
LINEAGES_MANIFEST_MISSING: 268
LINEAGES_FEATURE_MISSING: 5
CANONICAL_PROVEN_SAFE: 6
CANONICAL_PROVENANCE_ONLY: 71
CANONICAL_EVAL_AT_RISK: 0
CANONICAL_CHECKPOINT_AT_RISK: 0
CANONICAL_BOTH_AT_RISK: 0
CANONICAL_UNAUDITABLE: 565
EVAL_ONLY_ALLOWED: 0
EVAL_ONLY_BLOCKED_MISMATCH: 0
EVAL_ONLY_BLOCKED_MANIFEST: 0
EVAL_ONLY_BLOCKED_UNAUDITABLE: 99
REEVAL_ONLY_REQUIRED: 99
CLEAN_RETRAIN_REQUIRED: 93
REEXTRACTION_REQUIRED_LINEAGES: 5
PERMANENTLY_UNAUDITABLE: 245
COUNT_SUM_EQUALS_642: true
COUNT_SUM_EQUALS_99: true
COUNT_SUM_EQUALS_1096: true
PROTECTED_FILES_UNCHANGED: true
```

审计完成后停止；未启动 Wave B、feature extraction、重训练或 canonical promotion。
