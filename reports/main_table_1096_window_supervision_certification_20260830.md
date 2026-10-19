# 主表 1096 设置：window–supervision 错配认证

日期：2026-08-30  
范围：冻结的 1096 个 probing setting；不改 registry、canonical 表或实验产物。

## 最终认证

1. 如果“每一个实验类别”指 family：A1、A2、B1、B2、C1、C2、C3 七类中都存在已经确认采用错误 window 配对的实验。
2. 如果指 1096 个 setting 的每一格：不成立。结论为：
   - 862 格：确认采用 `shared-obs24` feature cache 与 `h124` manifest 的冲突协议；
   - 119 格：历史 per-prefix cache 与 loader/manifest 的 window 定义冲突，高风险，但缺 run-bound raw-ID index，不能升级为 confirmed；
   - 109 格：未闭环，既不能判错，也不能判 safe；
   - 6 格：manifest-exact safe。
3. 用户列出的六类只覆盖 1040 格。Wan/Cog campaign 另有 B1=56，加入后才是 1096。
4. “标签错误”应严格表述为：raw depth/mask/pose 通常对 manifest 指定帧本身是正确的，但 VFM feature 来自另一组 raw frames；训练把它们拼成同一个样本，造成错误的 `(X, y)` 或 `(X, condition, y)` 配对。feature tensor 本身也通常不是数值损坏，而是 sample identity/provenance 错误。

## window 协议为什么会错

Motion-uniform observation 序列先由相机轨迹生成，再按 `observations_per_window`、stride 和最多四个窗口选取 `obs_start`。不同 `obs_len` 会改变后部窗口的 start；因此同一个 `window_0001`/`window_0002`/`window_0003` 不保证代表同一批 raw frames。

冲突的两套协议是：

| 消费方 | prefix 8 | prefix 12 | prefix 16 | prefix 24 |
|---|---:|---:|---:|---:|
| shared feature cache | 24 | 24 | 24 | 24 |
| h124 manifest | 12 (=8+4) | 16 (=12+4) | 20 (=16+4) | 28 (=24+4) |

旧 loader/路径用 `(scene, window_id, prefix tail)` 找 feature，但这个 key 没有编码 `obs_len`/protocol hash，历史路径也没有在取 tensor 前断言 cache record 的 raw `indices` 与 manifest `streaming_indices` 完全相等。于是同名路径发生了语义碰撞。

## 实际 raw-ID 数据

现存 shared cache 与 canonical h124 manifest 的全量比较：

| Prefix | exact | mismatch | mismatch rate |
|---:|---:|---:|---:|
| 8 | 1134/2723 | 1589/2723 | 58.35% |
| 12 | 1362/2723 | 1361/2723 | 49.98% |
| 16 | 1709/2723 | 1014/2723 | 37.24% |
| 24 | 1694/2721 | 1027/2721 | 37.74% |
| 合计 | 5899/10890 | 4991/10890 | 45.83% |

Wan 与 MiniMax-H3 的 681/681 个 scene index 逐 record 相同，以上计数对两者完全一致。CogVideoX 当前可见 465 个 scene，raw-window records 与 Wan 相同，且可见子集的四个 prefix mismatch rate 分别为 58.26%、49.97%、37.39%、37.59%。V-JEPA2 与 LingBot 当前各只恢复出一个 scene，但都直接复现了四个 prefix 的非零错配；VideoMAE 当前没有可读的本地历史 index。

这证明的是“每个采用该协议的 setting 的训练 population 中含有大量错配样本”，不是“每个样本都错”。

## 各 family 实际消费的监督

| Family / probe | 直接进入 loss/head 的量 | raw supervision lineage | 错窗后的准确表述 |
|---|---|---|---|
| A1 depth | `vfm_feat`, `dmaps` | manifest raw frame 的 depth | feature–depth 错配 |
| A1 camera/pose | `vfm_feat`, intrinsics/extrinsics | manifest raw frame 的相机标定/位姿 | feature–pose 错配 |
| A1 mask | `vfm_feat`, `identity_ids` | manifest raw frame 的 instance-ID mask | feature–instance-mask 错配 |
| A2 view consistency | `vfm_feat`, `overlap_gt` | depth→pointmap，再结合 K/pose 计算 overlap | feature–geometric-overlap 错配；不是直接 depth/mask/action loss |
| B1 ego belief | `vfm_feat`, object mask, `hidden_obj_polar` | instance mask + depth pointmap + pose | mask 被施加到另一窗口 feature；feature–polar 错配 |
| B2 ego belief v2 | `vfm_feat`, `belief_query_feat`, `hidden_obj_polar` | query 是 manifest object mask 在 feature 上 masked-pool；polar 由 mask/depth/pose 派生 | query token 本身被异窗 mask 污染，同时 feature/query–polar 错配 |
| C1 action dynamics | input feature, 9D action, isolated target feature | action 由 manifest anchor/future pose 派生 | input–action–target 三元组错配；不直接使用 mask label |
| C2 path integration | input feature sequence, path actions, target feature sequence, valid mask | 连续相对 pose actions + exact raw-ID targets | input 与 action/未来 feature 序列错配 |
| C3 counterfactual | input feature sequence, anchor-relative actions, target sequence, overlap gate | pose action；depth/pose 派生 overlap validity | input 与 action/target/gate 错配；不直接使用 mask label |

因此不能写成“每类实验都同时把 depth、mask、pose、action 四类标签弄错”。各 family 只受其实际消费的监督影响。

按具体 probe 的 setting 统计如下：

| Probe | 总设置 | Numeric | 旧 canonical | Exact-safe | Confirmed | High-risk | Unresolved |
|---|---:|---:|---:|---:|---:|---:|---:|
| `streaming_depth`（depth/pose） | 168 | 140 | 90 | 0 | 136 | 0 | 32 |
| `streaming_mask` | 112 | 62 | 62 | 0 | 80 | 0 | 32 |
| `streaming_camera`（legacy pose-only） | 20 | 0 | 0 | 0 | 16 | 0 | 4 |
| `view_consistency` | 168 | 138 | 82 | 0 | 136 | 0 | 32 |
| `ego_belief`（B1） | 56 | 56 | 10 | 0 | 56 | 0 | 0 |
| `ego_belief_v2`（B2） | 168 | 151 | 112 | 0 | 136 | 23 | 9 |
| `action_dynamics`（C1） | 136 | 96 | 96 | 6 | 98 | 32 | 0 |
| `path_integration`（C2） | 136 | 92 | 92 | 0 | 104 | 32 | 0 |
| `counterfactual`（C3） | 132 | 98 | 98 | 0 | 100 | 32 | 0 |

## 按 family 的 1096 格认证

| Family | 总设置 | Numeric | 旧 canonical parent | Exact-safe | Confirmed | High-risk | Unresolved |
|---|---:|---:|---:|---:|---:|---:|---:|
| A1 | 300 | 202 | 152 | 0 | 232 | 0 | 68 |
| A2 | 168 | 138 | 82 | 0 | 136 | 0 | 32 |
| B1 | 56 | 56 | 10 | 0 | 56 | 0 | 0 |
| B2 | 168 | 151 | 112 | 0 | 136 | 23 | 9 |
| C1 | 136 | 96 | 96 | 6 | 98 | 32 | 0 |
| C2 | 136 | 92 | 92 | 0 | 104 | 32 | 0 |
| C3 | 132 | 98 | 98 | 0 | 100 | 32 | 0 |
| 合计 | 1096 | 833 | 642 | 6 | 862 | 119 | 109 |

结论：七类中均有 confirmed error；但 default8 的 224 格没有任何一格达到 exact-safe 或逐 scene confirmed 的证据门槛。

## Campaign × family

每个格子的格式为 `Confirmed / High-risk / Unresolved / Exact-safe`；四项之和为该 campaign-family 的 setting 数。

| Campaign | A1 | A2 | B1 | B2 | C1 | C2 | C3 |
|---|---:|---:|---:|---:|---:|---:|---:|
| default8 | 0/0/64/0 | 0/0/32/0 | — | 0/23/9/0 | 0/32/0/0 | 0/32/0/0 | 0/32/0/0 |
| wan_cog_layer_sweep | 56/0/0/0 | 56/0/0/0 | 56/0/0/0 | 56/0/0/0 | 18/0/0/6 | 24/0/0/0 | 20/0/0/0 |
| videomae_vjepa_layer_sweep | 80/0/0/0 | 40/0/0/0 | — | 40/0/0/0 | 40/0/0/0 | 40/0/0/0 | 40/0/0/0 |
| minimax_h3_layer_benchmark | 56/0/4/0 | 20/0/0/0 | — | 20/0/0/0 | 20/0/0/0 | 20/0/0/0 | 20/0/0/0 |
| lingbot_video_layer_benchmark | 40/0/0/0 | 20/0/0/0 | — | 20/0/0/0 | 20/0/0/0 | 20/0/0/0 | 20/0/0/0 |

Campaign 汇总：

| Campaign | 设置 | Numeric | Canonical | Exact-safe | Confirmed | High-risk | Unresolved |
|---|---:|---:|---:|---:|---:|---:|---:|
| default8 | 224 | 223 | 192 | 0 | 0 | 119 | 105 |
| wan_cog_layer_sweep | 292 | 240 | 80 | 6 | 286 | 0 | 0 |
| videomae_vjepa_layer_sweep | 280 | 203 | 203 | 0 | 280 | 0 | 0 |
| minimax_h3_layer_benchmark | 160 | 112 | 112 | 0 | 156 | 0 | 4 |
| lingbot_video_layer_benchmark | 140 | 55 | 55 | 0 | 140 | 0 | 0 |

## 错误是否真实进入训练

862 个 confirmed setting 全部有 `train_started_at`：

- 612：train and eval complete；
- 199：train complete、eval 未开始；
- 16：train partial/interrupted；
- 35：train failed。对最后 35 格，只有启动证据，不能保证已经消费过有效 minibatch。

因此至少 827 个 confirmed setting 完成或部分执行了错误协议下的训练；604 个已经产出 numeric result；444 个曾是旧 canonical parent。这不是仅存在于代码里的潜在 bug。

## default8 的严格边界

default8 使用历史 `FEAT_STREAMING_BY_PREFIX`：

- A1 64、A2 32：loader 记录的 `obs_len=p` 与名义 per-prefix cache 长度一致，未观察到长度冲突；但历史 raw-ID index 不可读，所以不是 exact-safe。
- B2：23 格 loader `obs_len=24` 对 cache `p`，高风险；9 格未观察到长度冲突但仍未 exact 证明。
- C1：28 格动态 `p+1` 对 cache `p`，4 格 h124 `p+4` 对 cache `p`，全为高风险。
- C2：16 格动态 `p+4`、16 格 h124 `p+4`，都对 cache `p`，全为高风险。
- C3：22 格动态 `p+4`、10 格 h124 `p+4`，都对 cache `p`，全为高风险。

不能把 shared-cache 的 45.83% mismatch rate 机械套到 default8；也不能因长度相等就把 A1/A2/部分 B2 判 safe。

## C 类 target 的独立问题

Wan/MiniMax 的 flat target union 对 canonical h124 的 future coverage：

- p8：2724/2724 三个 horizon 全覆盖；
- p12：1800/2724 全覆盖；剔除 `48da625e8c_w003` 后为 1799/2723；
- p16：2723/2723；
- p24：2721/2721。

p8/p16/p24 target 能找到不等于 C probe 安全，因为 input feature 仍错窗。p12 还叠加明确 target 缺失；loader 对缺失 target 写零张量并将 valid 置 false，loss 只消费 valid 项，所以有效样本量与分布发生偏移。Cog/V-JEPA2/VideoMAE/LingBot 的历史 target index 当前没有同等级的全量逐 scene 证据。

## 六格 exact-safe 的含义

六格是 Wan p8 C1 的 layers `{0,5,10,15,25,29}`。它们是后来绑定到 frozen parent 的 manifest-exact versioned recovery lineage，不代表最初的 shared-cache run 安全。

## 证据等级

- A：Wan/MiniMax 现存 shared cache 的 681-scene 全量 raw-ID index + 历史 extractor/run-root/manifest lineage；exact-safe 另有 Dataset canary。
- B：CogVideoX 465-scene可见子集的实际 index + extractor/run-root/manifest lineage。
- C：V-JEPA2/LingBot 的单 scene 直接复现，或 VideoMAE 的 extractor 调用 + shared root + run-bound h124 manifest；不是全量历史 object 的逐字节法证。
- D：default8 的历史 per-prefix index 缺失；只有结构冲突或“未观察到长度冲突”。
- E：MiniMax A1 camera 的 4 格 root 未恢复且训练未开始。

`aoss://.../FEAT_STREAMING` 是可变路径，目前没有每个历史 AOSS object 的 version ID/hash 与本地镜像逐文件闭环。因此 A/B/C 的区别必须保留，不能把 862 格描述成同一强度的逐样本法证。

## 首次启动顺序（按 ledger timestamp）

- default8：A1 depth → A1 mask → A2 → B2 → C1 → C2 → C3。
- wan_cog_layer_sweep：C3 → A2 → A1 depth → B1 → B2 → C2 → C1。
- videomae_vjepa_layer_sweep：C2 → A1 depth → C3 → B2 → C1 → A2 → A1 mask。
- minimax_h3_layer_benchmark：B2/C2 → A1 depth → A2 → C1 → C3 → A1 mask → legacy A1 camera。
- lingbot_video_layer_benchmark：C3/C2 → C1/B2 → A1 depth → A1 mask → A2。

这里是实际 first-start 顺序，不是设计文档中的逻辑顺序。

## 可复查输入和逐格证据

- setting-level 1096 审计：`results/analysis/main_table_1096_window_audit/window_audit_1096.csv`
- chronological ledger：`results/analysis/main_table_1096_window_audit/probing_sequence_1096.csv`
- 642 canonical parent 影响：`results/analysis/main_table_1096_window_audit/canonical_parent_642_window_impact.csv`
- 734 atomic canonical 影响：`results/analysis/main_table_1096_window_audit/atomic_canonical_734_window_impact.csv`
- raw-ID follow-up queue：`results/analysis/main_table_1096_window_audit/raw_id_followup_queue_228.csv`
- 现存 shared index 计数：`results/analysis/main_table_1096_window_audit/shared_index_observed_20260830.json`
- 可复现审计逻辑：`scripts/audit_main_table_1096_windows.py`
