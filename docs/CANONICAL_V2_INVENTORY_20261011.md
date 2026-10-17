# Canonical V2 逐实验库存核验 — 2026-10-11

本次在 **736 个有明确当前身份的正式 probe 条目中，核实 696 个已完成训练**：主线 552 中 512 个、Wan 独立 168 个、MiniMax-H3 L30 A/B 16 个。696 个均找到正常预算结束的 `Trainer.fit` 日志和非空 checkpoint，并逐项匹配 model、task、逻辑层/磁盘层、prefix、seed、timestep。其余 40 个为 LingBot C：逐例 validation PASS，但未建立正式训练完成证据。

**668 个同时满足逐例 protocol canary PASS；另外 28 个 Wan A2 已训练结束，但验证验收口径存在冲突。当前没有任何条目被本库存认定为“正式科学指标已核验、可直接发布结果”。** 这与“已有大量训练成功”是两个独立判断。不要因为缺正式指标而重跑已经完成的训练。

机器可读清单：[canonical_v2.csv](../metadata/release_inventory/canonical_v2.csv)。共 737 行：736 行 `entry_level=run`，另 1 行 MiniMax non-L30 特征恢复 `entry_level=scope`。统计实验数量时必须只取 `run`。没有混入旧 1,096/1,260 ledger，也没有读取其旧 checkpoint 作为新 V2 成功证据。

## 可核验的已完成矩阵

全部 run 为 **seed=42，prefix={8,12,16,24}**。表中 A/B 四项指 `streaming_depth`（depth+camera）、`streaming_mask`、`view_consistency`、`ego_belief_v2`；C 两项指 `action_dynamics` 和 `path_integration`。这里没有 `counterfactual` 或新增 `streaming_movement` 训练。

| 模型 | 本次明确范围 | 已完成任务 × 层 × 4 prefixes | 完训/范围 | 完训且逐例 gate PASS | 未确认完成 |
|---|---|---|---:|---:|---:|
| Aether | 主线 | A/B+C；L1 | 24/24 | 24 | 0 |
| CogVideoX | 主线 | A/B+C；L0,5,10,30,35,40 | 144/144 | 144 | 0 |
| DINO | 主线 | A/B+C；L0 | 24/24 | 24 | 0 |
| F3R | 主线 | A/B+C；L24 | 24/24 | 24 | 0 |
| LingBot | 主线 | **A/B**；L10,20,30,40,48（最后一层磁盘编号47） | 80/120 | 80 | **40 个 C** |
| Qwen2.5-VL | 主线 | A/B+C；L−1 | 24/24 | 24 | 0 |
| VideoMAE | 主线 | A/B+C；L5,11,17,23 | 96/96 | 96 | 0 |
| V-JEPA2 | 主线 | A/B+C；L5,11,17,23 | 96/96 | 96 | 0 |
| **主线小计** | **552条** | | **512/552** | **512** | **40** |
| Wan | 独立 Wan168 | A/B+C；L0,5,10,15,20,25,29 | 168/168 | **140** | 0 |
| MiniMax-H3 | 独立 L30 A/B16 | A/B；L30 | 16/16 | 16 | 0 |
| **本次 run 合计** | **736条** | | **696/736** | **668** | **40** |

LingBot 与 MiniMax-H3 的 timestep 为 250；其余条目为 749。CSV 同时保存 `layer` 和 `disk_layer`：LingBot 逻辑 L48 对应磁盘 L47，最新 C validation bundle 的 key 使用 `l47`；依据主线 `experiments_552.json` 的显式映射关联，未把二者误判成新增第六层。

例如可以直接按 `model=cogvideox, probe=action_dynamics, layer=30, prefix=16, seed=42` 从 CSV 得到精确 experiment_id、最新正式 job、checkpoint、训练日志、run manifest、validation result 和观测时间，无需从目录名推断成功。

## 本次推翻或补齐的旧状态

1. **CogVideoX C48 已完成训练。** 10/2 文档的“validation 48 PASS、formal job IDs 尚无”是当时观测。现有 [正式 ledger](../artifacts/main_v2_probe_v2_816_20260917/single_gpu_cog48_c_h800_20261001_r1/submissions.json) 48 条均有 job ID，`resumed_utc` 在 10/4 下午；48 个正式输出均有完整正常结束日志及非空 checkpoint。结束时间为 **2026-10-04 21:28:50 至 2026-10-05 21:26:14 UTC**。这48项不能再列作“待提交”。
2. **主线 10/1 的 430/552 可以被逐行新证据更新为 512/552 已完成。** 此数字来自当前明确身份及日志/checkpoint 核验，并非将不同快照的汇总直接相加。10/1之后的 Cog/LingBot A/B replacement 已纳入；缺失的 Cog replacement submission ledger 通过冻结 plan 的精确 item、snapshot 命令及实际输出补链。
3. **Wan168 全部有完训实物。** 112 A/B + L20 的8个C + 其余48个C。10/2报告的 C48成功已由逐实验日志/checkpoint证实；无需重训。
4. **MiniMax-H3 L30 A/B16 全部训练结束。** 10/2 04:13报告尚记录15成功、1运行；最后一项 `streaming_mask, L30, P24, seed42` 的正常结束日志为 **2026-10-02 05:48:39 UTC**。16个run都有checkpoint及匹配manifest。
5. **LingBot C40仍只确认 validation，不能计作完训。** [10/4 bundle](../artifacts/main_v2_probe_v2_816_20260917/probe_validation_lingbot40_c_20261004_r1/bundle.json) 对应40份 `result.json` 全PASS；[正式 ledger](../artifacts/main_v2_probe_v2_816_20260917/single_gpu_lingbot40_c_h800_20261004_r1/submissions.json) 仍是19个 `SUBMIT_INTENT`、0个job ID。其余21项连正式intent也未记入该ledger。未据此声称这些任务今天仍在运行或肯定从未提交。

当前主线条目来自 [552 progress rows](../artifacts/main_v2_probe_v2_816_20260917/probe552_progress/experiments_552.json)，Wan 来自 [168队列](../artifacts/main_v2_probe_v2_816_20260917/wan_probing_168_20260923_r2/queue_wan_168.jsonl)，MiniMax L30 来自 [正式提交清单](../artifacts/main_v2_probe_v2_816_20260917/minimax_l30_ab16_20260930_r2/formal_submission_ledger.json)。目录中的 `experiments_816.csv` 是旧ledger的选择结果，不是此处当前run分母；`frozen_queue_r1/queue_816.jsonl` 自身实际上也只有760个去重条目。

## 28个 Wan A2 的验证冲突

范围精确为：**Wan × view_consistency × L{0,5,10,15,20,25,29} × P{8,12,16,24} × seed42 = 28项**。

最新逐例 `probe_validation_wan_ab112_20260923_r4/cases/*/result.json` 对这28项记录 `FAIL`，原因是未核验到要求的3个optimizer steps和12个train samples；代表例为 [L0/P12结果](../artifacts/main_v2_probe_v2_816_20260917/probe_validation_wan_ab112_20260923_r4/cases/wan__scannetpp__t749__l0__p12__view_consistency__seed42/result.json)。这不是正式训练退出失败：其后28项正式训练均正常结束并留下checkpoint。

与此同时，[production_gate.json](../artifacts/main_v2_probe_v2_816_20260917/single_gpu_wan_ab112_20260923_r1/production_gate.json) 明确写出28/28 feature inventory groups PASS，并采用代表性的 `PASS_REAL_LOADER_FORWARD_BACKWARD_VALIDATION_ONE_BATCH` 放行 view-consistency。它支持“当时采用较窄的代表性验证放行”，不能改写成“28个逐例3-step验证全PASS”。

因此这28行保持 `training_status=COMPLETE, correctness_status=NOT_VERIFIED`，并同时保留两类证据。所缺工作是明确验收口径、补齐必要的逐例验证证据和指标审计；**不是已证明要重做28次训练**。

## 哪些结果可以发布、还缺什么

本表 `correctness_status=VERIFIED` **仅覆盖正式训练身份与本地 protocol canary**。并不表示模型结论、独立 evaluator、checkpoint 数值或正式科学指标已被完整验证。CSV 中的 `metric_status=NOT_AUDITED_NO_MATCHED_OFFICIAL_EXPORT_LOCATED` 和 `ready_for_release=NO` 是独立字段。`NO` 指尚不能据本证据直接形成可接受的结果发布表，不阻止发布带有明确状态说明的代码、训练工件及库存本身。

- **668个已完训且逐例gate PASS条目**：从现有checkpoint回收/整理匹配的正式评价指标，检查非有限值、恒定值、全零等退化，保存评价器版本、split、样本/场景数、checkpoint身份和聚合规则。此处不把训练loss、训练cosine或三步canary的CSV当作正式评估结果。
- **Wan A2 28个完训条目**：先解决上节验证口径，再作同样指标核验与结果导出。
- **LingBot C40**：明确正式提交身份，定位或完成正式训练、checkpoint和最终评价；40个PASS仅证明前置验证结束。
- **MiniMax其他范围**：最新明确文件仍是 non-L30 的特征恢复 scope；streaming L10/20/40/49，target L20/40/49。10/2报告记录streaming 43,536/43,536、target 1,203/2,043，当时8个worker运行。没有据此发明当前完整probe矩阵或完成数；需将更新的特征receipt与新正式probe清单匹配。旧MiniMax148项不可直接填入当前V2库存。

[2026-09-22任务设计审查](../reports/canonical_v2_task_design_review_20260922.md) 还指出 C horizon字段、冻结runtime独立评价字段、C2 camera-center几何、A2投影视野重叠标签及其解释边界问题。此库存未声称这些问题已在各run对应的评价版本修复。C1/C2需要相应动作/视觉基线与准确指标解释；A2应保持实际的投影overlap分类口径。这里训练中存在的 `val` 使用不能改称独立盲测 `test`。

## 证据时间、权限与核验限制

- [10/1 summary](../artifacts/main_v2_probe_v2_816_20260917/probe_progress_20261001/summary.json) 的观测为10/1 10:15；同目录 [scheduler_jobs.json](../artifacts/main_v2_probe_v2_816_20260917/probe_progress_20261001/scheduler_jobs.json) 的**内容观测时间实际为10/4 07:50:55**。两文件不能当作同步快照，后者仍早于 Cog C48 的10/4下午正式恢复。
- [watcher summary](../artifacts/main_v2_probe_v2_816_20260917/probe552_progress/summary.json) 的写入时间10/5不能覆盖其scheduler观测9/28。CSV逐行 `observed_at` 优先记录正常训练结束时间，另外保留 `scheduler_observed_at`、`validation_observed_at` 和正式提交时间。
- 很多 `completed/*.json` / `TRAIN_COMPLETE.json` 是存在但当前用户无读权限的文件。没有更改权限或把“文件存在”冒充读取内容。完训判定使用可读的完整预算正常结束日志与非空checkpoint；CSV标明 `PRESENT_UNREADABLE`。
- checkpoint只核验精确路径与非零文件大小，**没有反序列化、重新前向或逐文件哈希**；不能保证所有checkpoint当前字节均未损坏。正式发布前仍需目标checkpoint可加载与评价复现检查。
- 本次是本地只读证据库存；没有训练、评价、抽取、scheduler请求或联网。只新建CSV与本报告。定点读取runner引用的输出，未递归扫描logs/runtime/cache。没有找到或核验完整正式metric export，不等价于证明整个仓库绝不存在任何未索引的指标文件。
