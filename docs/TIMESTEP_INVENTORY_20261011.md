# Timestep 实验与发布库存 — 2026-10-11

**canonical-v2 seeded-v1 的 576 个训练均已结束，且本地均有终点 checkpoint；这不等于 576 个科学结果已经验收。** 216 个独立 official validation 指标逐行匹配实际摘要、计划与 checkpoint。其中 80 行有明确退化表现；其余 136 行是尚待验收的候选，不自动标为 scientifically verified。本库存的 `YES_WITH_SCOPE` 为 0，`NO` 为 576；这表示现有证据尚未完成结果验收，不表示已证明所有训练错误。

机器清单：[timestep.csv](../metadata/release_inventory/timestep.csv)。它有 **576 个 `entry_level=run`**，外加 **1 个 `entry_level=scope`** 描述旧 R1 的 84 个无效 completion markers。不要将 scope 行当成新增实验，或把 84 加进 canonical-v2 分母。

本次仅定点读取已有 CSV、JSON、manifest、completion marker 与 checkpoint 文件名；没有启动训练、评估、联网或调度器查询，也没有改写原始科学结果。库存观察时间为 `2026-10-11T18:42:33Z`。训练 census 的原始观察时间为 `2026-09-23T05:36:13.567963+00:00`；official 行另保留 launcher receipt 的 `time` 字段。历史 receipt 不是实时任务状态。

## 固定矩阵与完成证据

| 模型 | scheduler timestep | layer | prefix | seed | 每个 probe | 全部 8 probes |
|---|---|---|---|---|---:|---:|
| Wan | 499、999 | 15、20、25 | 8、12、16、24 | 42 | 24 | 192 |
| CogVideoX | 499、999 | 15、20、25 | 8、12、16、24 | 42 | 24 | 192 |
| MiniMax-H3 | 500、1000 | 15、20、25 | 8、12、16、24 | 42 | 24 | 192 |

数据为 ScanNet++，特征 namespace 为 `FEAT_STREAMING_CANONICAL_V2_SEEDED_V1`，窗口契约为 canonical-v2。两个 timestep 仅表示 early/late scheduler index，不推断噪声高低。八个训练 probe 是 `streaming_depth`、`streaming_mask`、`view_consistency`、`ego_belief`、`ego_belief_v2`、`action_dynamics`、`path_integration`、旧 `counterfactual`；每个 probe 72 行。

对 [results.csv](../results/analysis/tstep_canonical_v2_seeded_v1/results.csv)、[census.csv](../results/analysis/tstep_canonical_v2_seeded_v1/census.csv) 的所有 576 个选中 run，本次确认：

- `TRAIN_COMPLETE`、`run_manifest.json` 与物理 checkpoint 均存在；每个选中目录恰有一个 `VideoProbe3D/*/checkpoints/*.ckpt`。
- A1-depth/mask 共 144 个 checkpoint 文件名为 `epoch=99-step=244500.ckpt`；其他 432 个为 `epoch=49-step=61100.ckpt`，均吻合 census 的终点预算。没有反序列化或验算 checkpoint tensor。
- manifest 的 model/timestep/layer/prefix 与库存逐行一致；seed 全为 42，`test=false` 全为 576。四种 scene manifest 的现有 SHA256 与 576 个训练 manifest 中记录的 SHA256 均匹配。
- 47 个 run 记录了 resume checkpoint，其余 529 个为空。所有 run 均 `test=false`，216 个 official evaluator 又显式使用终点 checkpoint，不能据近期修复的“resume 后测试旧 checkpoint”问题，追溯宣布这些 official 结果选错 checkpoint；seed-zero 缺陷也不适用于 seed 42。
- 576 个 manifest 的 `git.commit` 全为 `unknown`，需要从冻结源码/启动记录补齐实际源版本并核查对应版本的有效样本计数、NaN 处理与指标契约。当前源码修复不会追溯修复历史训练或指标。

[collection_summary.json](../results/analysis/tstep_canonical_v2_seeded_v1/collection_summary.json) 记载 668 个历史 attempt dirs，census 已选出 576 个完整 run；41 个遗留 writer claims 不覆盖 completion 事实。这里没有把 attempt 数当实验数。

## Training-time metric 与 official metric 分开统计

| 模型 | 完成训练 | 有限 training primary | 缺失/非有限 training primary | A1 零损失（有限数子集） | official 行 | official 退化行 | 未见下述退化的 official 候选 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Wan | 192 | 192 | 0 | 0 | 72 | 0 | 72 |
| CogVideoX | 192 | 192 | 0 | 0 | 72 | 24 | 48 |
| MiniMax-H3 | 192 | 129 | 63 | 32 | 72 | 56 | 16 |
| 合计 | **576** | **513** | **63** | **32** | **216** | **80** | **136** |

513 个有限 training primary 含 32 个零损失；原图排除这些零值后仅剩 481 行。它们仍不是 481 个已验证科学结果。63 个缺失/非有限值全部位于 MiniMax-H3 L20/L25：C1、C2、旧 C3 各 16，B1 为 15。B1 在 `t1000/L25/P8` 的一行有限（35.0645°），也没有因此通过科学验收。

### 独立 official evaluation 实际已有的内容

[PLAN.json](../artifacts/tstep_eval_a1a2b2_20260923_r1/PLAN.json) 指定 `eval_split=val`，并非独立 test；[official_eval_results.csv](../results/analysis/tstep_canonical_v2_seeded_v1/official_eval_results.csv) 为三种 probe 各 72 行。

| Probe | official 主指标 | 有限行 | 已见退化行 | 未见下述退化的候选 | 验收前具体事项 |
|---|---|---:|---:|---:|---|
| A1 depth | `mean_loss_depth` | 72 | 16 | 56 | 导出现有 AUC，排除零损失并核查有效性/来源 |
| A2 view consistency | `balanced_acc` | 72 | 16 | 56 | 核查 NaN 与有效样本处理，诊断无判别力输出 |
| B2 ego-belief-v2 | `mean_ang_err_deg` | 72 | 48 | 24 | 验证常数预测、输入响应性和过滤样本契约 |
| 合计 | — | **216** | **80** | **136** | 没有新增 `VERIFIED` 或 `YES_WITH_SCOPE` |

216 个 key 均有 `rc=0` 的最终 receipt、实际 `val_summary.json`、物理 checkpoint；summary 的指标值等于 CSV，checkpoint 路径等于计划指定且属于 census 选中 run，model/timestep/layer/prefix 均匹配。首次 MiniMax shard 中 22 次失败后来有 22 次成功 retry，不能把这些已恢复的重试算成当前缺口。

**A1 AUC 已存在于实际摘要，不能再统称“尚缺 AUC 评估”。** 全部 72 个 depth 摘要已有 `mean_Auc_30`/`Auc_30`；原 official CSV 只导出了 loss。本库存另存 `official_mean_Auc_30`，但不将其提升为最终表。未退化的 56 个 depth 行首先需要现有指标导出与有效性验收；没有必要仅为获得 AUC 重跑相同评估。

具体退化证据：

- MiniMax-H3 两个 timestep × L20/L25 × 所有 prefixes：16 个 official depth loss 为 0，AUC_30 也为 0；训练 depth 与 mask 共 32 个零损失已被原 [FINDINGS.md](../results/analysis/tstep_canonical_v2_seeded_v1/FINDINGS.md) 排除。不能作“完美几何”解释。
- 同一 MiniMax L20/L25 区域的 16 个 A2：`balanced_acc=0.5`、`mean_overlap_mae=NaN`、`pos_acc=0`、`neg_acc=1`。这里的退化判断同时使用非有限输出和无判别力表现；单独的 chance-level 分数不证明实现错误。
- CogVideoX 与 MiniMax-H3 的全部 48 个 B2：两个模型、所有 layers 和两个 timesteps 对同一 prefix 精确相同，仅有四个角误差值：P8 90.1461778641°、P12 80.1995698248°、P16 73.3987003014°、P24 70.5168665445°。其中 P8 的 12 行接近 chance，但其他 36 行也不能直接解释为 timestep 效应。需检查预测 tensor 是否有限以及输入/动作是否影响输出；经验证的真实负结果仍可以单独发布。
- 全部 72 个 B2 的 loader 均为 276 行；P8 使用 80 个 valid records、排除 196，P12/P16/P24 使用 171、排除 105。不能把角误差称为 276 个有效样本的结果；过滤可能来自任务有效性规则，不能只凭排除数宣布错误。

`correctness_status=DEGENERATE` 在此指观察到退化指标或输出特征，不等同于已证明训练实现错误或必须重训。576 个 run 中该状态为 **96**：32 个 A1 零损失 + 16 个 A2 + 48 个 B2；另 **480** 为 `NOT_VERIFIED`。只有 80 个退化 run 在 official 216 行里，另外 16 个是没有 official export 的 A1 mask。

## 尚缺哪些结果，以及评估与重训如何区分

216 行之外的 **360 个训练 run 没有进入这份 official export**；它们不是 360 次失败评估。按 probe 恰好五组各 72：

| 已训练的矩阵组 | 尚缺的结果 | 应优先采取的工作 |
|---|---|---|
| A1 mask，72 | matched official mask 指标 | 56 个非零 training primary 先核查后补评估；MiniMax L20/L25 的 16 个零值先诊断输入/权重 |
| B1 ego-belief，72 | official B1 指标、valid-record 契约 | B1 被 216-job plan 明确排除；57 个有限 training primary 先补协议与评估，另 15 个非有限先诊断 |
| C1 action dynamics，72 | `global_R@1/R@5` 与 no-action/shuffle/copy controls | 56 个有限 cosine 不等于 retrieval；匹配已训练 checkpoint 补 official 评估，另 16 个非有限先诊断 |
| C2 path integration，72 | multi-horizon retrieval 与 controls | 同上：56 个先评估，16 个先诊断，并保留各 horizon 契约 |
| 旧独训 counterfactual，72 | 仅有旧 C3 training-time cosine；无当前 C3 结果 | 保留为明确命名的 legacy ablation；当前 C3 应复用 matched C1 checkpoint 做 evaluation-only，不再训练旧 counterfactual head |

[当前 C 协议](../protocols/C_PROBES.md) 已将 C3 定义为 frozen C1 predictor 的 action-sensitivity evaluation。完整 timestep C3 比较因此还需要 **72 个与 C1 一一匹配的 C3 evaluation cases**，使用真实有效的 action branches/equivalence labels、`intervention_validity`/margin 与 controls。这些是新的评估实例，不能算成新增训练，也不能拿 72 个旧 counterfactual 训练 completion 抵账。若缺 intervention 输入，应先构建输入，而非宣称 C3 是完成的 null finding。

有限 C1–C3 cosine 原始范围约 0.9865–0.9991；接近饱和的 cosine 或 1e-4 级差值不能代替 global retrieval / intervention 结论。

工作顺序应是：

1. **已完成可复用的执行产物：** 576 个 terminal checkpoint、216 个可逐行追溯的 evaluator summary。不要因为结果未验收就自动重开整个训练矩阵。
2. **导出/验收：** 先整理 136 个无上述退化表现的 official 候选，补齐源版本、指标/有效样本契约与 scoped validation 声明；A1 AUC 直接从现有 summary 导出。
3. **补评估：** 对有效的已训练 mask/B1/C1/C2 checkpoints 补 official 指标；当前 C3 在 matched C1 上做 evaluation-only，并验证所需 interventions。
4. **先诊断再决定重训：** 63 个缺失/非有限 training primary、32 个 A1 零值、16 个 A2 退化以及 48 个 B2 常数均有逐行标记。几组有不同含义，不能全部直接判为失败训练。只有证明特征/标签错配、非有限训练输入、坏权重或实现问题影响了训练后，才修复特征/实现并重训受影响的 cells；若只是日志/汇总或 evaluator 缺陷，应优先修复后重评/重导出。

所有列出的评估都须保留 `val`、seed 42、固定 matrix 的范围。独立 test 与多 seed 确认可以作为后续科学增强，不能凭不存在的 test 阻止诚实发布已验收的 validation 结果；本次未标 ready 的原因是协议/有效性尚未验收。

## 旧 R1：单独隔离，不补 canonical-v2 缺口

[INVALID_R1_20260830.md](../reports/tstep_canonical_v2/INVALID_R1_20260830.md) 明确作废 84 个完成标记：mixed observation windows 将特征与几何/标签配到了不同时间窗口，10,890 个有效样本中 4,991 个受影响。旧 R1 的 feature/checkpoint/index 不能晋升为 canonical-v2。库存只写一个 `PROTOCOL_INVALID` scope 行，不伪造未重建的 84 个精确 run 身份，也不删除历史证据。

## 字段语义与复核边界

`training_status=COMPLETE` 表示 ledger、completion marker 和终点 checkpoint 文件名一致；不证明 tensor 有限或训练过程科学正确。`result_status=OFFICIAL_EVAL_MATCHED_NOT_ACCEPTED` 表示已有独立 evaluator 的指标、身份、checkpoint 匹配，但尚未验收；`ready_for_release=NO` 是科学结果发布状态，不是全局代码/许可打包状态。

每个 run 提供 original experiment key、model/probe/layer/prefix/timestep/seed、training metric、official metric/样本数、额外 AUC、精确 summary/receipt/checkpoint 路径、原始观测时间、退化标记与具体 remaining work。复核通过的结构计数为 576 个唯一 run、216 个唯一 official key、513 个有限 training primary、136 个 matched official candidates、80 个 official degenerate rows；所有链接目标按本地库存检查，checkpoint bytes/输入 tensor 和冻结代码语义并未在本次重跑验证。
