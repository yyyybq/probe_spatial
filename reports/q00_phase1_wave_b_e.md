# Q00 Phase 1：Wave B direct-node 执行与 Wave E lineage resolution

生成时间：`2026-08-30T04:56:02.583509+00:00`。本报告冻结 1096 universe，不含 diffusion-step sweep。

## 结论

Wave B 没有产生新结果。三个 direct-node canary 都在 evaluator 启动前发现 canonical h124 manifest 与 generic feature cache 指向不同 raw-frame window，严格 fail closed；剩余 96 个未运行，promotion 未发生。有限数值仍为 **833/1096**，主表 canonical parent 仍为 **642/1096**。

这次失败揭示的是历史训练 lineage 问题，而不是临时执行错误：generic cache 使用共享 24-frame observation window，h124 manifest 分别使用 12/16/20/28-frame window；旧 loader 只以 `(window_id, tail)` 命中，不校验 raw IDs。99 个 Wave-B checkpoint 已在混合错配上训练，不能通过重新 eval 修复。

Wave E 最终为 PACKAGE=30、EVAL_ONLY=0、CLEAN_RETRAIN=17、FEATURE_THEN_RETRAIN=182、PERMANENTLY_BLOCKED=20。30 个 legacy B2 已进入独立 namespace，但只表示 artifact-complete protocol package；由于 BY_PREFIX raw-frame lineage 未 run-bound，它们不进入严格全局分析。

更重要的是，642 个现有 canonical parents 的表级 canonical-valid 状态没有被改写，但科学 lineage 审计只确认 **6** 个 Wan p8 action_dynamics rows 使用 manifest-exact versioned cache；444 个确认受 shared24↔h124 错配影响，另 192 个 BY_PREFIX 缺 raw-ID 证明。严格主分析因此暂时只能使用 6 个 exact-safe parent，其余 636 个 quarantine。Wan p8 的六行还受既有 action-validity construct audit 限制，只能作为 weak-signal case study，不能支撑全局空间结论。

## Direct-node 执行证据

- Active successor bundle：`reports/q00_phase1_runs/q00_phase1_wave_b_e_direct_shapecontract_20260830T040218Z_2075f3eb`，SHA256 `ed5ee041e18efd45ede9c40ff79e488fcf3c1a9ec2bc7a2fa3ff918a6df0cd5c`。
- 旧 SCO writers 在 direct execution 前均确认未启动并取消；direct host 为 8×H800 节点。
- 三个 canary 的 checkpoint、manifest SHA、CUDA/runtime 和 tensor shape gate 通过；full prefix-index identity gate 分别在 LingBot、MiniMax-H3、Wan 上拒绝。
- 2,208 个 val associations。按 prefix 聚合：p8 240/312、p12 276/276、p16 348/204、p24 342/210（exact/mismatch）；对应的每个 model×prefix group 模式分别为 120/156、138/138、174/102、171/105，8/8 groups 均非 exact。
- evaluator started=0，summary=0，prediction=0，新增 numeric=0，canonical overwrite=0。

## Wave B 99

99/99 重分为 `FEATURE_PARTIAL_OR_MISSING / FEATURE_EXTRACTION / FEATURE_THEN_RETRAIN`。Wan p8 path_integration 六行可以复用 exact streaming 与 h1 target，但 h2/h4 target 覆盖为 0/2724；它们仍须补 target 后 clean retrain。其余 93 行没有 qualified full exact root。逐行见 `wave_b_eval_results.csv`。

## Wave E 249

- 30 PACKAGE：已写入 `results/canonical_protocols/legacy_b2_shared_hidden_8_12_16_24_val_v1/`；30 protocol IDs、150 artifact hashes、指标重算和 prefix 内 label identity 均 PASS。`protocol_canonical_valid=true`，但 `scientifically_comparable=false`、`analysis_eligible=false`。
- 0 EVAL_ONLY：旧 22 行全部撤销；V-JEPA/Cog/Wan family canary 均复现 lineage mismatch。
- 17 CLEAN_RETRAIN：均为 Wan p8 streaming-only rows，可用 manifest-exact versioned root；仍需新的 frozen train bundle 与每 campaign/model canary。
- 182 FEATURE_THEN_RETRAIN：包含原 177 clean 中的 162 行、旧 eval-only 中的 20 行；F3R/Qwen BY_PREFIX 也因缺 run-bound raw-ID 证明 fail closed。
- 20 PERMANENTLY_BLOCKED：MiniMax standalone camera contract 不得由 joint depth+camera 静默替代。

## Canonical lineage impact

| 分类 | Parents | 处理 |
|---|---:|---|
| exact/versioned Dataset-canary safe | 6 | 严格分析暂可保留，但需 Wan action-validity construct caveat |
| confirmed generic shared24 vs h124 error | 444 | quarantine；manifest-exact feature + clean retrain + eval |
| BYPREFIX raw-ID unverified | 192 | quarantine；先历史 index/provenance audit，不能直接判 safe |

主 canonical 表仍是 642 parents；独立 legacy namespace 使 artifact-level protocol-canonical union 为 672，但不增加 strict analysis eligibility。

## 下一训练队列

`next_training_queue.csv` 包含 **843** 个确定性 replacement jobs：444 个 confirmed-invalid canonical remediation + 399 个 missing-or-noncanonical completion。后 399 个中，243 个没有任何数值，156 个已有 provisional 或 protocol-unfrozen 数值但不能直接 canonicalize。执行类为 FEATURE_THEN_RETRAIN=826、CLEAN_RETRAIN=17。另外 192 个 canonical BYPREFIX 留在 lineage audit hold，5 个 Wave-F rows 留在人为 protocol decision hold。

所有 843 行当前 `dispatch_eligible=false`。Priority A 只选每个 campaign/model 一个 train+eval canary；canary 必须先有 immutable exact feature/manifest bundle。历史 probe median 粗略训练点估为 **16095.56 GPU-h**，eval 点估 **42.15 GPU-h**；826 行还需 feature remediation，feature GPU-hours/磁盘必须按 unique cache group canary 后报价，旧 Q00 总成本已失效。

## Protected files

Registry 与七张 main canonical CSV 的 before/after SHA256 完全相同。唯一新增的 result namespace 是独立 legacy package；没有改写已有 canonical 行。

## 验收

```text
WAVE_B_CANARY_PASS: false
WAVE_B_SUCCESS: 0/99
NUMERIC_RESULTS_BEFORE: 833
NUMERIC_RESULTS_AFTER: 833
CANONICAL_PARENTS_BEFORE: 642
CANONICAL_PARENTS_AFTER: 642
PROTOCOL_CANONICAL_PARENTS_AFTER: 672
STRICT_MAIN_ANALYSIS_ELIGIBLE: 6
CANONICAL_CONFIRMED_LINEAGE_ERROR: 444
CANONICAL_LINEAGE_UNVERIFIED: 192
B2_PACKAGEABLE: 30
B2_NEEDS_REEVAL: 0
WAVE_E_EVAL_ONLY: 0
WAVE_E_CLEAN_RETRAIN: 17
WAVE_E_FEATURE_THEN_RETRAIN: 182
WAVE_E_PERMANENTLY_BLOCKED: 20
NEXT_TRAINING_JOB_COUNT: 843
PROTECTED_FILES_UNCHANGED_OR_EXPECTED_DIFF_ONLY: true (UNCHANGED)
```

## Stop point

本轮在 fail-closed classification、legacy packaging 和确定性 queue 处停止：没有启动 remaining-96、Wave-E eval、训练或 feature extraction。下一步只可从 queue 的 Priority-A canary 开始，并先完成 exact feature namespace。diffusion-step、planning、prediction-feature 和新增 baseline 继续暂停。
