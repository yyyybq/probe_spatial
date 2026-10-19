# 主表 1096 个 probing 设置的时间窗回顾

日期：2026-08-30 UTC  
冻结全集：`experiments/task_status_1096_before_expand_20260822.csv`，1096 行，SHA256 `9c7d6807...`。  
审计原则：只有逐样本 raw-frame identity 加 Dataset/DataLoader canary 通过，才记为 exact-safe；长度相同、路径名相同、feature 文件存在或 provenance-only 均不升级为 safe。

## 结论

1096 个设置逐行判断后：

| 判断 | 设置数 | 含义 |
|---|---:|---|
| `EXACT_SAFE` | 6 | 仅 Wan p8 C1 的六个 manifest-exact 层 `{0,5,10,15,25,29}` |
| `CONFIRMED_ERROR_SHARED24_VS_H124` | 862 | 实际训练绑定 h124 manifest，但输入来自共享 obs24 cache；实验协议已确认有非零、且通常大量 raw-frame mismatch |
| `HIGH_RISK_DYNAMIC_OBS_VS_PER_PREFIX` | 89 | 早期动态 loader 的 obs_len 与 per-prefix cache 的生成长度存在结构冲突；历史 cache index 未取回，故不冒充“逐样本已证明” |
| `HIGH_RISK_PER_PREFIX_VS_H124` | 30 | h124 manifest 配 per-prefix cache；生成协议指向 `p` 对 `p+4`，但历史 index 不在当前可读存储中 |
| `UNRESOLVED_NO_OBS_CONFLICT` | 105 | 长度层面没有发现冲突，但没有 raw-ID exact 证据，不能视为 safe |
| `UNRESOLVED_FEATURE_ROOT` | 4 | MiniMax p24 未启动设置，feature root 未恢复 |

因此，当前能进入 exact-safe 集合的是 `6/1096`。已确认错误为 `862/1096`；另有 119 个高风险结构冲突和 109 个尚未闭合的设置。这里“已确认错误”是实验设置使用了混合时间窗，不代表该设置中的每个样本都错位。

对冻结 canonical 的影响更直接：

| 层级 | 总数 | exact-safe | 已确认错误 | 高风险 | 未证明 |
|---|---:|---:|---:|---:|---:|
| canonical parent | 642 | 6 | 444 | 96 | 96 |
| atomic canonical ID | 734 | 6 | 504 | 96 | 128 |

也就是说，不能继续把原 canonical 标记当作“科学有效”标记；它只说明旧流水线曾接受该数值。窗口审计后，除六个 exact-safe 结果外，其余 canonical 数值均应隔离，不能直接写入论文主表。

## 为什么 862 个设置可以确认有错误

历史共享 feature extraction 一次传入前缀 `{8,12,16,24}`。旧 extractor 用最大 prefix 建 motion-uniform window，因此 shared cache 的 window 长度为 24。h124 manifest 则按 `prefix + max_horizon(4)` 独立重算：

| prefix | shared feature obs_len | h124 manifest obs_len |
|---:|---:|---:|
| 8 | 24 | 12 |
| 12 | 24 | 16 |
| 16 | 24 | 20 |
| 24 | 24 | 28 |

motion-uniform 不是“先生成同一条长序列，再裁前缀”。改变 observations-per-window 会改变 obs_start、采样密度和 window_id 对应的 raw frames，所以四组都不是同一 window 定义。

当前可见 index 的只读全量比较进一步确认这不是 Wan 特例：Wan 与 MiniMax 的计数完全一致，CogVideoX 可见子集也复现同样比例。

| prefix | Wan exact/checked | MiniMax exact/checked | CogVideoX 可见子集 |
|---:|---:|---:|---:|
| 8 | 1134/2723 (41.65%) | 1134/2723 (41.65%) | 776/1859 (41.74%) |
| 12 | 1362/2723 (50.02%) | 1362/2723 (50.02%) | 930/1859 (50.03%) |
| 16 | 1709/2723 (62.76%) | 1709/2723 (62.76%) | 1164/1859 (62.61%) |
| 24 | 1694/2721 (62.26%) | 1694/2721 (62.26%) | 1159/1857 (62.41%) |

Wan/MiniMax 合计均为 `5899/10890` exact，`4991/10890` mismatch。862 个设置的共享 root 包括直接 `aoss://.../FEAT_STREAMING` 和实际指向该 root 的 prefix alias；六个 versioned manifest-exact root 被单独排除，不在这 862 个中。

## 119 个高风险为什么没有写成“已证明错误”

这 119 个都来自早期 `FEAT_STREAMING_BY_PREFIX` 世代。

- 30 个设置的实际结果后来绑定了 h124 manifest，而 feature root 仍是 per-prefix root。生成脚本和命名均指向 feature window `obs_len=p`，manifest 是 `p+4`。
- 89 个没有 manifest，只能从实际训练日志恢复 loader 的动态 window：B2 有 23 个短前缀使用 obs24 对 per-prefix `p`；C1 的 28 个为 `p+1`；C2 的 16 个和 C3 的 22 个为 `p+4`。
- 已知 Wan p8 负控证明“obs8 cache 对 h124 obs12”确实产生 `1029/2724` streaming mismatch，但不能把这一个 cache 的逐样本结论机械复制到所有历史模型/root。

因此这些设置必须隔离，并优先读取历史 `prefix_index.npy`；如果取不回，安全做法是按 manifest-exact 协议重提 feature 和重训，而不是把它们升级为 exact-safe。

## 105 个“无长度冲突”也不是 safe

这 105 个全部来自早期 default8：64 个 A1、32 个 A2、9 个 B2。日志中 loader obs_len 与 per-prefix root 名义长度一致；9 个 B2 是八个 p24 和一个 Qwen p16。

这只能说明当前没有发现长度级矛盾。历史 extractor 参数、window raw IDs 和当时实际消费的 index 没有形成逐样本闭环，因此判断是 `UNRESOLVED_NO_OBS_CONFLICT`，不是 `SAFE`。这正是“并非所有格子都已证明错误”在逐行账本中的体现。

## probing 实验进行顺序

下面是冻结 ledger 中每个 setting 的首次训练启动顺序。2026-08 以后多个 campaign 有重叠，并不是一个 campaign 完全结束后才开始下一个。

1. `default8`：224 个，2026-07-15 19:12 开始，最晚记录完成到 2026-07-20 12:31。
   `streaming_depth → streaming_mask → view_consistency → ego_belief_v2 → action_dynamics → path_integration → counterfactual`。
2. `wan_cog_layer_sweep`：292 个，2026-07-24 23:45 开始并持续到 8 月。
   首次启动次序为 `counterfactual → view_consistency → streaming_depth → ego_belief → ego_belief_v2 → path_integration → action_dynamics`。其中 path/action 是 8 月补跑。
3. `videomae_vjepa_layer_sweep`：280 个，2026-08-08 17:49 开始。
   `path_integration → streaming_depth → counterfactual → ego_belief_v2 → action_dynamics → view_consistency → streaming_mask`。
4. `minimax_h3_layer_benchmark`：160 个，2026-08-10 01:26 开始。
   `ego_belief_v2/path_integration（同刻）→ streaming_depth → view_consistency → action_dynamics → counterfactual → streaming_mask → streaming_camera`。
5. `lingbot_video_layer_benchmark`：140 个，2026-08-15 03:12 开始。
   `counterfactual/path_integration（同刻）→ action_dynamics/ego_belief_v2（同刻）→ streaming_depth → streaming_mask → view_consistency`。
6. 2026-08-20 至 21 日做 Wan p8 C1 恢复；最终六个 manifest-exact 结果替换对应 frozen parent 的 feature lineage。这是恢复事件，不额外增加 1096 的行数。

逐行的全局次序在 `probing_sequence_1096.csv`。需要注意：该文件的时间是 setting 首次启动时间，而窗口判断绑定的是 Q01 恢复出的实际结果 lineage；发生过 retrain/recovery 的行，两者可能不是同一次运行。

## target 的判断

共享 flat target union 的只读 index 复核显示 Wan 与 MiniMax 一致：

- p8、p16、p24 的三个 future horizons 在 flat union 中均可找到，但 streaming input 仍错位，所以 C 结果仍不安全。
- canonical p12 为 `1800/2724` 三 horizon 全有效；历史 filtered manifest 去掉 `48da625e8c_w003` 后是 `1799/2723`。单 horizon 覆盖分别约为 `1806/2724`、`1804/2724`、`1816/2724`。
- 零张量 fallback 和 valid mask 会改变有效训练样本量与分布；它不能修复 input/label 时间窗错位。
- anchor isolated target 覆盖没有得到全量 exact 证明，仍需单独 canary。

## 每类结果的处理决定

| 判断 | 当前动作 |
|---|---|
| exact-safe 六格 | 保留，可作为 clean baseline |
| 862 个 confirmed error | 已有数值作废/隔离；未 eval 的旧 checkpoint 不应直接 eval；应按 exact manifest 重提 feature、clean retrain、再 eval |
| 119 个 high-risk | 先取历史 index 做 raw-ID audit；取不回则重提+重训，不允许直接入表 |
| 105 个 no-conflict unresolved | 读取历史 index 并跑 Dataset canary；通过前不得标 safe |
| 4 个 root unresolved | 本来就没有可用结果；修复协议后重新启动 |

## 产物

- `window_audit_1096.csv`：1096 个设置逐行结论、证据等级、target 风险和处置。
- `probing_sequence_1096.csv`：1096 行首次启动时间顺序。
- `canonical_parent_642_window_impact.csv`：冻结 canonical parent 影响。
- `atomic_canonical_734_window_impact.csv`：atomic canonical ID 影响。
- `raw_id_followup_queue_228.csv`：119 个高风险加 109 个未闭合设置的 index/canary 队列。
- `shared_index_observed_20260830.json`：共享 streaming/target index 的实际比较计数。
- `summary.json`：计数、输入 hash、campaign/family/lifecycle 汇总。

本审计没有启动 feature extraction、训练或 eval，也没有修改 registry 和 canonical CSV。
