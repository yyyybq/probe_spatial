# 实验条目与最终发布计划（2026-10-11）

这份清单把 **canonical V2 主线、timestep、其他 analysis** 分开。
依据本地实验回执、run manifest、checkpoint 路径和评估产物，不查询实时调度器。
目录日期、文件修改时间和记录中的实际观测时间不同；采用可定位到具体实验的较新证据，
不把旧任务账本中的成功数补进新协议。

“训练结束”与“训练正确、结果可发布”分别记录：

| 判断 | 需要的证据 |
| --- | --- |
| 训练结束 | 对应实验身份的正常结束记录、训练配置及协议要求保留的产物；主线/timestep 同时核对非空 checkpoint，部分 analysis 只归档 receipt、训练曲线和结果 |
| 正确性已核验 | 在该条目声明的核验范围内有对应证据。主线的 `VERIFIED` 仅覆盖训练身份及输入协议 gate；完整结果验收还需 split、有效样本、权重与指标身份及有效性审计 |
| 可作为发布结果候选 | 上述证据完整，结果可追溯，评估和汇总齐全，并明确声明 validation/development 的结论范围 |
| 代码与产物已可公开发布 | 另需统一源码版本、依赖、复现命令、许可证、可获取的权重/数据索引及最终发布包验证 |

`YES_WITH_SCOPE` 只表示第三行，不能解释为整个仓库已完成第四行。
较差或接近随机的成绩本身不是错误；NaN、异常零损失、输入或身份不匹配需要查证原因。
没有独立 blind test 不阻止发布明确标注的 validation/development 结果。

## 逐实验清单

以下三个清单保留模型、任务、层、prefix、seed、训练状态、正确性状态、结果状态、
剩余工作和来源。`run`、`package`、`scope` 是不同统计单位；不相加。

| 分组 | 逐条记录 | 详细说明 |
| --- | --- | --- |
| Canonical V2 | [canonical_v2.csv](../metadata/release_inventory/canonical_v2.csv) | [主线对账](CANONICAL_V2_INVENTORY_20261011.md) |
| Timestep | [timestep.csv](../metadata/release_inventory/timestep.csv) | [timestep 核验](TIMESTEP_INVENTORY_20261011.md) |
| Other analysis | [analyses.csv](../metadata/release_inventory/analyses.csv) | [独立分析与 D-series](ANALYSIS_INVENTORY_20261011.md) |

合并视图：[all_experiments.csv](../metadata/release_inventory/all_experiments.csv)。
按分组和单位的统计：[summary.json](../metadata/release_inventory/summary.json)。
更新三个输入表后运行 `python scripts/summarize_release_inventory.py` 重建。
这会记录输入表的 SHA256，但不会查询调度器或重跑实验。

## 一、Canonical V2：696 个训练已结束，正式结果尚未验收

当前可逐条定位的范围为 **736 个 run = main552 + Wan168 + MiniMax L30 A/B16**。
旧目录名中的 816、旧账本的 1,096/1,260 都不是这个分母。

| 模型 | 层 | 本次矩阵 | 训练结束 | 输入协议/身份核验与缺口 |
| --- | --- | ---: | ---: | --- |
| DINO | 0 | 24 | 24 | 24 有匹配身份与 canary PASS |
| F3R | 24 | 24 | 24 | 同上 |
| Qwen2.5-VL | -1 | 24 | 24 | 同上 |
| Aether | 1 | 24 | 24 | 同上；发布时还需对齐冻结 runtime 实现 |
| V-JEPA2 | 5/11/17/23 | 96 | 96 | 96 有匹配身份与 canary PASS |
| VideoMAE | 5/11/17/23 | 96 | 96 | 同上 |
| CogVideoX | 0/5/10/30/35/40 | 144 | 144 | 包括已于 10 月 4–5 日结束的 C48，旧快照已过时 |
| LingBot | 10/20/30/40/48 | 120 | 80 | A/B80 完成；C40 输入验证通过，但正式训练完成未确认；逻辑 L48 对应磁盘 L47 |
| Wan | 0/5/10/15/20/25/29 | 168 | 168 | 140 有逐例 PASS；A2 的 28 个逐例 3-step gate FAIL 与代表性 one-batch 验收记录冲突 |
| MiniMax-H3 | 30 | 16 | 16 | 仅四项 A/B probe，不能外推其他层或 C 系列 |
| 合计 | — | **736** | **696** | **668 有身份匹配与输入 gate PASS，28 待澄清 gate，40 未确认完训** |

共同 prefix 为 8/12/16/24，probe seed 为 42。完整层的六项训练是
depth、mask、A2 view-consistency、B2 ego-belief-v2、C1、C2；MiniMax L30 只含前四项。
当前 C3 是复用 C1 的评估，不在这六项独立训练计数中。

这里的“668”是**训练身份及输入协议核验**，不是 668 个已经验收的最终分数。
696 项均有正常训练终止日志及非空 checkpoint；尚未对权重 tensor、全部有效样本、
实际 evaluator 与最终指标逐项验收。主线正式发布表、对应图和可复现聚合入口仍待完成。
Wan A2 的 gate 冲突应先查明，不能仅凭冲突直接要求全部重训。
MiniMax 非 L30 的工作另记为 feature-recovery scope，不伪造尚未冻结的训练矩阵。

## 二、Timestep：576 个训练结束，216 个独立评估可追溯

每模型为 2 个 timestep × 3 层（15/20/25）× 4 prefixes × 8 probes，均 seed42。
Wan/CogVideoX 的 timestep 为 499/999，MiniMax 为 500/1000。

| 模型 | 完训 | 已匹配的独立评估 | 有明确异常的评估 | 未见这些异常、待验收的评估 |
| --- | ---: | ---: | ---: | ---: |
| Wan | 192 | 72 | 0 | 72 |
| CogVideoX | 192 | 72 | 24 | 48 |
| MiniMax-H3 | 192 | 72 | 56 | 16 |
| 合计 | **576** | **216** | **80** | **136** |

216 项为 depth/A2/B2 各 72 项；72 个 depth summary **已经包含 AUC_30**，需要导出而非仅为 AUC 重跑。
80 项异常是 MiniMax depth 损失与 AUC 均为零的 16 项、A2 含 NaN 的 16 项，以及
Cog/MiniMax B2 跨模型/层/timestep 指标相同的 48 项；最后一类还需检查预测 tensor，
不能预先认定训练 bug，也不能据此解释 timestep 效应。

另外 **360 项未进入这个 official export**：mask、B1、C1、C2、旧独训 counterfactual 各 72。
前四项需补匹配评估；当前 C3 需在对应的 72 个 C1 checkpoint 上做新的评估，旧 C3 不能抵账。
全部训练中有 63 个缺失/非有限 training primary、32 个 A1 零损失，需要定点诊断；
这些统计与上述 80 个评估异常的口径不同，不直接相加。
576 个 manifest 的 source commit 都为 `unknown`，还需从冻结 runtime/启动记录恢复来源。
因此当前没有 timestep 行标为结果已验收，但也没有理由自动重训全部 576 项。

## 三、其他 analysis：六个结果包可按现有范围整理发布

| 已有结果包 | 已完成实验单位 | 可声明的范围与尚需说明 |
| --- | --- | --- |
| Independent P8 | 四模型 × 45 最终训练配置 = 180；三 seeds | Validation 结果已审计；只持久化了 12 个 full-dynamics 权重，不能承诺 180 个权重下载 |
| Continuous-action relational V2 | 四模型 × 五 variants × 五 seeds = 100 checkpoints | Validation 结果已审计；Wan 为 L29，与上项 L20 不同 |
| SE3 Wan/V-JEPA2 final r3 | 60 layer/seed bundles，内含 240 heads；另五个 synthetic controls | 冻结包 21 项文件 hash 均匹配；仍是 pilot scenes 上的 development evidence。L11 的 20 heads 源 receipt 当前无读取权限，需补可访问归档，不据此重训 |
| Probe-validity paper | 8 个 C1 + 4 个 B2 evaluation cells，68 metric rows | 四模型 C1 P12/P16 与 B2 P12；本分析未新增训练，不包含 B2 P16 四模型扩展 |
| D-series Wan D1/D2 | 三 seeds，9 stage bundles | Dev 聚合已完成，0 test |
| D-series V-JEPA2 D1/D3 | 三 seeds，24 stages（含六项 evaluation-only） | Dev 聚合已完成，0 test |

**V-JEPA2 D2 的三个 seed 也已经训练并完成 dev evaluation**，每 seed 3,864 条条件/样本记录；
缺的是三种子聚合、paired contrasts 和统一 D-series 导出，不是默认需要重训。
六个结果包的科学范围已通过现有审计；许可证、干净环境与公共下载整理是整个代码/产物包的发布工作，
不要据此把已有科学结果重新标为实验失败。

仍没有可用完整结果的扩展：

- **V1 C3-from-C1：** 40 cells 中 32 个源 checkpoint 存在但缺 exact V1 inputs，另 8 个 checkpoint 缺失；尚无 scored summaries。
- **B2 P16 四模型扩展：** V-JEPA2、VideoMAE、Wan 的 exact checkpoint 缺失，Cog 单模型不能代替完整比较。
- **Active representation MVP：** 只有单 state forward 成功，48-state canary 未完成，约 2,000-state probe 未完成。
- **SeeSE3 replication：** 缺请求的中间层特征，pair 数未达要求；与已完成的 SE3 r3 分开。

## 哪些工作必须完成，哪些不必重做

| 优先级 | 工作 | 完成标准 | 是否需要重新训练 |
| --- | --- | --- | --- |
| P0 | 冻结这次 release 的实验范围 | 明确纳入哪些主线 cell、timestep 子表和 analysis package；未完成项目显式排除 | 否 |
| P0 | 固定“结果—训练产物—数据协议—源码”映射 | 每项有稳定 ID、split、seed、layer/prefix/timestep、feature/manifest hash、源代码与 evaluator 版本；按协议记录选中权重或训练 receipt/accepted results。若发布权重，另提供对应文件及 hash | 通常否；身份确实无法恢复时另行决定 |
| P0 | 将主线正式完成证据与 protocol gate 对上 | 验证任务 PASS、正式训练 job、结束回执、checkpoint、评估分别关联，重试去重 | 只补真正未完成或输入无效的 cell |
| P0 | 清理异常指标的原因 | MiniMax 深度损失为零、A2 overlap MAE 非有限、B2 跨模型恒定指标等有输入有效性、有效样本数和预测检查；恒定指标尚不能证明预测 tensor 恒定 | 输入/训练坏则重提特征或重训；只是评估/汇总错误则重评 |
| P0 | 冻结 checkpoint 选择与评估规则 | 保留已接受结果的冻结选择协议；待验收项明确使用的权重，不能按 final split 选最好结果 | 已接受结果不默认重评；身份、选择规则或 evaluator 改变时补评；若选择依据受 bug 影响则检查保存的 epoch 权重 |
| P0 | 统一 C3 定义 | 新 C3 从冻结的 C1 checkpoint 做 action-sensitivity evaluation；旧独训 `counterfactual` 只以 `legacy_trained_c3` 收录 | 新 C3 不需要单独训练 |
| P0 | 补齐 release 范围内的正式指标 | C1/C2 retrieval、控制基线和对应 C3 intervention 指标；不能用训练期 cosine 代替；已在 summary 中的 AUC 直接导出 | 大多只需评估/汇总 |
| P0 | 固定每个结果包的取数入口 | 接受的 per-sample 数据、summary、主表、图和文字能从同一来源重建；保留有效数与排除原因 | 否 |
| P0 | 合并实际使用的源码到可发布版本 | 对 accepted runs 的 frozen runtime/source snapshot 做差异清单；把必要代码合入正式目录或提供明确兼容入口 | 先做兼容重放，不默认重训 |
| P0 | 补齐源码/模型/数据的归属与分发文件 | 补入实际版本的第三方许可文件；发布需要的下载索引，不把私有缓存路径当下载方法 | 否 |
| P1 | 验证干净环境和 GPU 路径 | 从最终发布目录安装依赖；CPU CI、至少一个代表性提特征→probe→eval 流程通过；声明支持的多 GPU 范围 | 小规模验收，不是全矩阵重训 |
| P1 | 发布包可获取且可核验 | 权重、必要小型样例或特征索引可下载，哈希匹配；文档链接只指向实际发布内容 | 否 |
| 可选扩展 | 增加训练 seeds、独立 confirmation/test、新后端或未完成探索 | 只有把这些新结论纳入本次 release 才成为其前置条件 | 视新增范围决定 |

不能因为上一轮代码修了 bug，就推断所有历史训练错误：timestep 的 576 个 manifest
均记录 seed 42、`test=false`，其中 216 个已核对的独立评估显式指定并匹配终点 checkpoint，因此 `seed=0` 与
训练后 test 错载续训前 checkpoint 这两项不能直接作为整批重训的理由。
但无效样本加权、输入退化、具体 epoch 选择等仍要按每次 run 使用的源码核对。

## 最终代码版本应收敛到哪里

1. 以正式目录中的训练、提特征、评估入口为准，固定 canonical-v2 的窗口、prefix、
   target、deterministic-noise 和 namespace 合约；不要把名称中的 `SEEDED_V1` 误读为旧 V1 数据。
   独立 P8、continuous、SE3、D-series 保留各自冻结协议及 namespace，不强制迁移到 canonical-v2。
2. 审计 accepted run 使用的 runtime 副本。当前 Aether temporal mapping 的某些测试
   只覆盖冻结 runtime helper，不能据此声称正式主目录包含同一实现。
3. 保留本次已经验证的 seed、checkpoint、有效样本计数、split 命名、AOSS 错误处理等修复；
   建立结果兼容说明。新 evaluator 修正的数字应标注新版本，不悄悄覆盖旧冻结包。
4. 将通用入口与集群提交、恢复、一次性历史脚本分开记录。主 README 只给可移植命令，
   历史脚本保留到 archive/operational 范围及迁移映射中，不凭相似文件名批量删除。
5. 固定一个实际可安装的基础环境和各 backend 的附加依赖。目前 CPU 测试已通过，
   但空白环境安装、真实 GPU 提特征和 hosted GitHub CI 仍需要最终发布包验收。
6. 完成 [第三方文件补齐](../THIRD_PARTY.md)、源码/权重分发清单、配置示例与发布目录链接检查。

更完整的已修问题和验证记录见 [代码发布审查](RELEASE_AUDIT_20261011.md)。

## 可以分批发布，不必补完全部历史任务

可以先发布通用代码和已通过完整性审计的独立 analysis 结果，清楚标注其模型、seed、
split 与结论边界；主线或 timestep 未验收的 cell 标成 pending/excluded。
若目标是发布“完整主线 + timestep”论文主表，则需完成所选矩阵的正式评估、异常原因确认和汇总冻结。

旧 1,260-task ledger 及其 2,184 行 candidate export 不作为新主线的默认结果来源。
其缺少 49 个 eligible ID、445 行 split 未标等问题，只有在决定继续发布这张旧表时才必须补齐。
不需要为了新的 release 自动重跑旧账本里的所有待评估任务。
