# Canonical V2 probing 任务设计与 A1 接入检查

检查日期：2026-09-22。范围：工作区的任务实现、配置、评估器，以及主线 `main_v2_probe_v2_816_20260917` 的清单、冻结队列和提交脚本引用的 `runtime/main_v2_probe_v2_r5/source`。同时检查 timestep canonical-v2 入口，但不将它等同于十模型主线。

本次为检查：没有修改训练代码、冻结配置、队列、checkpoint 或结果表，没有提交训练任务。没有重新核验远程任务和全部特征的实时可用性。

## 结论

- C1/C2/C3 有显著重叠。建议保留两个科学问题：动作条件下的直接未来预测，以及递归动作组合。将当前 C3 改造为直接预测基线/动作敏感性评估，暂不作为独立的反事实能力证据。
- A1 的 depth + pose 与 depth + movement 可以并列，分别测观察序列中的相机配置与相邻相机变换；depth 是共同辅助任务。工作区实现和配置已具备，但正式 canonical 主线的队列和冻结 runtime 尚未接入 movement。
- C 的 horizon 标签、C2 指标解释，以及冻结 runtime 的独立评估字段存在需要优先处理的问题。

## C1/C2/C3 实际测量什么

| 任务 | 实际输入与目标 | 独立价值与边界 |
|---|---|---|
| C1 `action_dynamics` | prefix 特征 + anchor→h1 的 GT 相对相机变换，预测 isolated future pooled feature | 单步动作条件 readout；属于训练 probe 的条件预测能力，不能直接归因于 backbone 自身执行动作 |
| C2 `path_integration` | prefix 初始化 GRU 状态，依次输入 anchor→h1、h1→h2、h2→h4，预测同三个终点特征 | 可以测递归更新，但当前是三个 waypoint 转移；h4 不是四次等间隔递归 |
| C3 `counterfactual` | 同一 prefix + anchor→h1/h2/h4 的三个变换，预测同三个终点特征 | 与 C1 的多 horizon 版本高度重叠，可作为 C2 的直接终点预测对照 |

证据：`configs/experiment/inscene15k_streaming/{action_dynamics,path_integration,counterfactual}_dino_v1.yaml`；`vidfm3d/data/components/inscene15k_dataset.py:1954`；三个对应 probe 类。冻结 runtime 也使用相同的 C3 head 和动作构造方式。

## 需要优先处理的发现

### 1. C3 的 counterfactual/intervention 命名超出当前设计能支持的结论

候选目标来自同一条实际轨迹的 h1/h2/h4，不是固定 horizon、从同一状态执行不同动作得到的匹配分支。因此更准确的名称是多终点动作条件预测/候选动作匹配。对静态场景仍有价值，但不能仅凭该指标宣称因果干预能力。

另外，`probe_counterfactual.py:63-69` 把所有候选动作放进同一个无 mask Transformer，并添加固定 slot positional embedding：

- 一个候选的输出能读取其他候选动作，和文件头声称的 independently 不一致。
- 固定 h1/h2/h4 顺序使 slot 本身提供 horizon 信息，即使动作恒定也可输出不同预测。
- C1 保留 pooled temporal tokens，C2/C3 对整个 prefix 再取时间均值；C2 与 C3 使用不同 head。当前分数差异同时包含读出结构差异。

CPU 合成检查（随机初始化的小尺寸同类 head，seed=7、eval 模式、dropout=0）：只修改候选 3 的动作，候选 1 输出最大变化为 0.145607；三个动作全置零后，slot 1/2 输出仍相差 0.028753；排列动作后对应输出的置换等变误差为 0.024414。这验证了架构允许候选耦合和 slot 捷径，不表示已测得训练后模型实际利用捷径的程度。

建议：直接预测时每次只输入一个候选动作，复用共享 predictor；若显式输入 horizon，则基线也输入同样 horizon。增加候选排列检查、同 horizon 替代动作、独立训练 visual-only/action-only 和 same-domain anchor-copy 基线。仅移除 positional embedding 还不能消除候选之间互相读取。

### 2. C2 的 pose/drift 指标不是连续轨迹位姿回归误差

`vidfm3d/eval_diag.py:649-667` 在每个样本自身的有效 target feature 集合（通常三个）中检索，然后取被检索目标的 GT 位姿，计算 translation error。GT 对应候选只要检索正确，误差就是零；不测旋转。`drift_rate` 是该终点检索误差除以稀疏 waypoint 路径长度，`loop_closure_error` 也依赖这样的候选检索。

所以应称为 retrieval-derived translation error，并作为附属指标；它不是传统 trajectory drift/SE(3) integration accuracy。若要保留 path integration 主张，需要共享一步 updater、多步 rollout、与 direct endpoint predictor 的配对比较，以及长 horizon/动作顺序/组合测试。

### 3. canonical horizon 与记录字段不一致

选目标时 `inscene15k_dataset.py:2078` 用 canonical observation offset；记录时 `:1962` 和 `:2605` 用原始帧索引差。

实际 manifest 示例 `00777c41d4_w000_p8`：anchor raw index=45；canonical h1/h2/h4 的 raw index 分别是 48/49/57。当前记录值会是 3/4/12，h2 反而可能进入 `h4` 指标。

`eval_diag.py` 按这个字段生成 `pose_error_h*` / `cf_cos_h*`；`scripts/build_canonical_results.py:259,324` 只取 h1/h2/h4。后果不仅是缺失，也可能错分组。应分别保存 `action_horizons=[1,2,4]` 和 `raw_frame_offsets=[3,4,12]`。这是元数据/评估问题，不等于未来目标选错。

### 4. 当前提交引用的冻结 runtime 缺少独立 C 评估器需要的字段

`runtime/main_v2_probe_v2_r5/source/vidfm3d/data/components/inscene15k_dataset.py:2281-2295` 的 C fast path 只返回训练所需特征、动作和有效 mask，缺少 `action_horizons`、`start_extrinsic`、`target_extrinsics`；C3 还缺 `counterfactual_overlap`。

同一快照的 `vidfm3d/eval_diag.py:361-366,398-404` 直接读取这些字段。该 fast path 无 train/eval 区分，直接使用快照自带 evaluator 会发生字段缺失。训练/Lightning validation loss 能正常运行，并不代表独立科学指标评估链可用。需新版本补齐评估字段或提供经过验证的独立 eval runtime；本次未实际运行远程 checkpoint 评估。

### 5. C3 的过滤和候选数会影响公平比较

C3 额外使用 overlap≥0.05 筛选，而 C2 不使用；`intervention_validity` 在有效候选≥2 时才计算。两候选与三候选的随机基准不同，因此不能把不同样本集合上的 validity 直接横比。

应报告有效样本/候选数和对应 chance baseline，优先在 C2/C3 的共同有效集合上比较。C1 的零动作/打乱动作测试是敏感性诊断，不能替代重新训练的 visual-only/action-only 对照；当前普通 C2/C3 evaluator 也未形成同样完整的控制矩阵。

## A1 两个并列版本的准备状态

| 层面 | 检查结果 |
|---|---|
| 工作区 YAML | 10 份 movement 配置：九个具名 VFM + direct-VLM 路由，全部继承对应 depth 配置 |
| Hydra 配置合成 | 10/10 通过；数据集表达式、data 设置、trainer、seed、optimizer 与对应 depth 配置一致 |
| active heads/loss | depth + camera → depth + movement；dmap=1，camera=0，movement=1 |
| 工作区模型与 loss | `RelativeMovementHead`、`ProbeModelPA`、训练 loss 分派均接入 |
| 工作区评估 | 两种版本均输出 `adjacent_movement_*`；pose 先转换为相同 S−1 条有向相邻边 |
| 已有单元测试 | `tests/test_a1_relative_movement.py`：5 passed，覆盖动作约定、6D rotation、输出形状/反传、head 参数量、完美预测 loss/metric |
| 主线 816 项清单 | `streaming_movement`=0；`streaming_depth`=136 |
| 主线去重后 queue_760 | `streaming_movement`=0；`streaming_depth`=128 |
| 当前提交引用的冻结 runtime | 无 movement YAML/head/loss/evaluator 接入；不能仅向旧 runtime 添加配置名后就运行 |
| timestep canonical 入口 | `build_tstep_probe_matrix.py:53` 未列 movement；`run_tstep_probe_group.sh:225` 的 pixel-task batch 分支也未列 movement |

这意味着单独使用当前工作区、提供正确数据/特征路由可准备 smoke run，但不能宣称正式主线的两个并列 A1 已完整部署。本次没有真实 canonical 数据上的 train→checkpoint→eval 验证。

补充：816 清单另有 MiniMax-H3 的 20 项 `streaming_camera`，去重队列中也保留了这 20 项；其 atomic output 是 `A1-camera`。它们不是 movement，不能作为第二条 A1 已存在的证据。重构新任务表时应明确它们是 pose-only 消融，避免和 depth+pose 的 camera 输出混记。

### A1 解释与对照注意点

- 当前所谓 absolute pose 实际是以 prefix 尾帧为参考的 pose，非全局世界坐标绝对定位。
- pose head 的 9D 是 translation + quaternion + FoV；movement 的 9D 是 rotation 6D + translation。参数量相同，但监督内容、输出数 S/S−1 和 loss 尺度不完全相同；不要称为只有标签名称变化的严格等价消融。
- 相机输出之间的公平比较应使用相同的 `adjacent_movement_*`，不能直接比较历史 all-pairs AUC 和相邻边 movement AUC。
- 两个版本各自保留 depth 结果，并用 variant 字段区分。共享 backbone 的不同辅助监督会影响 depth，不能合并成同一个 A1-depth 结果或取较好值。
- 目前 movement 对整段已观察 prefix 的相邻边读出，不是预测未知未来动作；其输入也没有 GT action。这与 C 的条件任务不同。

## 建议的新任务组织

1. A1-pose：depth + tail-referenced camera pose。
2. A1-movement：depth + adjacent camera movement，与 A1-pose 平行报告。
3. C-direct：共享 predictor 的动作条件直接未来预测，可覆盖 h1/h2/h4；报告 action gain、匹配基线和同场景目标区分。
4. C-rollout：递归动作组合，以同 horizon 的 C-direct 为对照；如果使用 h1/h2/h4 的三个 waypoint 转移，应明确标注；若研究四步递归，需输入 h0→h1→h2→h3→h4 的四个动作，h3 无需额外 isolated target 监督。
5. 原 C3 的候选动作匹配指标并入 C-direct/C-rollout 的评估，不单独宣称 counterfactual ability；严格反事实实验待固定 horizon 的分支数据和控制协议具备后再列主任务。

先修正 horizon/评估字段，再冻结新任务 schema 和包含 movement 的新 runtime/队列。旧冻结队列与历史结果保留原身份，避免悄悄改变正在运行任务的含义。

## 验证限制

单元测试在 CPU 上完成，有 CUDA 初始化环境警告；不代表 GPU 或远程存储链已验证。Hydra 检查使用占位数据根，只验证配置组合及继承关系。所有正式队列统计均为本地冻结产物统计，不是实时训练完成率。

## 追查补充：C2 几何、C1 多 horizon 和 A2

### C2 的平移计算还有具体几何错误

`vidfm3d/eval_diag.py:500-509` 直接计算 world-to-camera 外参 `E=[R|t]` 的 t 差，并用 t 折线长度作为路径长度。相机中心应为 `c=-R^T t`，不同 R 对应的 t 不能直接解释为同一参考系内相机位置。

直接调用仓库函数的 CPU 合成检查：两个相机中心都为 `(1,0,0)`，朝向分别为单位旋转与绕 z 轴旋转 90 度；代码报告 `_translation_error=1.4142135`，实际中心距离为 0。因此需要修正几何计算，而不只是修改指标名称。C 的外参还经过 anchor pointmap 尺度归一化，现有数值不能直接标为米；loop 判据 0.1 也不是统一的 0.1 米。

### C1 当前不是混合 horizon

工作区和冻结 runtime 的 10 个 C1 YAML 都为 `action_horizons: [1]`；检查的 `frozen_queue_r2/queue_760.jsonl` 中 76 个 C1 条目没有 horizon override。

单独改 YAML 为 `[1,2,4]` 无效：工作区 `inscene15k_dataset.py:1920-1929` 仍取 `target_feats[0]`、`extrinsics[1]` 和 `target_global[0]`；冻结 runtime 的 C1 分支同样只取第一个。

建议训练样本为 `(prefix, anchor→target action, isolated target, canonical_horizon, raw_frame_offset)`，h1/h2/h4 均匀采样或显式展开。保持 C1 每次独立预测一个 endpoint，不复用 C3 的候选联合注意力。评估分别报告三个 horizon 和宏平均；明确等 optimizer steps 或等 target-supervision budget 的比较口径，防止展开数据导致训练预算混淆。已有目标缓存是否覆盖每个实际 owner/配置仍需 gate 检查，协议定义本身已有 h1/h2/h4。

### counterfactual 可作为固定 C1/C2 的评估协议

训练 C1/C2 后固定 checkpoint，在 held-out 场景中固定同一 prefix，对有真实对应 isolated target 的候选 action 独立预测。C2 对每条备选路径从相同初始状态重新 rollout；不能把 anchor→h1/h2/h4 直接变换当成连续增量动作传给 C2。

只扰动动作并观察输出变化测的是敏感性。进一步要求预测分别靠近对应分支 GT，才支持正确的条件响应。任意打乱/取反动作通常没有对应真实目标，不能用原目标的退化当作该反事实目标的预测正确率。

对于静态场景，可以把不同相机 endpoint 的真实观测用作 counterfactual viewpoint query 的监督，未必必须在数据中实际走过每条分支；这依赖静态场景、合法视点、相机内参等条件一致，并且不支持动态世界因果推断。若要讲固定时间内动作导致不同未来，则需匹配 horizon/动作预算的分支数据。现有同轨迹 h1/h2/h4 可以先做多目标动作匹配，但 horizon 与候选身份混淆，需要单独披露。

### A2：当前可作为投影视野重叠 readout，但不能直接等同于共同可见表面一致性

已核对工作区与冻结 runtime，核心标签/head/loss 逻辑一致。

1. **标签缺少遮挡检查。** `spatial_diag.py:62-74` 仅检查源点有效、投影落在目标图像内和正深度，没有与目标深度图做一致性检查，也没有按目标落点排除无效深度。对称取 min 不能修复遮挡。合成输入中，两张同内参/外参、所有像素深度分别为 1 和 2 的互不一致表面仍输出 overlap=1；该人为构造用于证明缺少表面一致性约束，不是自然静态场景误标率估计。
2. **单个输出承受不同含义的监督。** `probe_ext_module.py:109-129` 同时对高 overlap（≥0.4）/低 overlap（≤0.05）做硬二分类，并对全体 pair 做权重 0.5 的 soft BCE。硬正标签把输出推向 1，真实 overlap=0.4 的 soft 标签把它推向 0.4。因此 sigmoid 不能同时解释为校准的 overlap ratio 和硬分类概率。全部 pair 都为硬样本的简化情形下，r=0.4 的最优输出是 `(1+0.5*0.4)/1.5=0.8`，不是 0.4。任务可保留，但应明确分类为主，或分开分类/回归 head。
3. **标签对称，head 不保证对称。** `probe_view_consistency.py:52-55` 拼接 `[zi,zj,zi*zj,abs(zi-zj)]`，交换 i/j 会改变前两部分。可使用对称输入，或对两个方向的 logit 取平均；评估只计 i<j 的独立无序 pair。
4. **需要控制时间邻近/外观捷径。** 当前全 prefix 的帧对可能仅凭时间邻近、相似外观就能得到较好重叠预测。应加入 train-fitted 时间间隔先验、无需训练的 pooled feature cosine，以及按间隔分层的结果。GT pose-only 可作诊断上界，需明确它使用特权输入。没有数据统计前，不能断言已有结果实际走了这些捷径。
5. **跨 prefix 比较改变了 pair 集合。** p8 有 28 个无序 pair，p24 有 276 个，长间隔/负例比例可能改变。全体 pair 指标反映上下文与任务组成的共同变化；如研究上下文长度的效果，应另外在相同 raw frame pair 上配对比较。置信区间按 scene/window 聚类，不能将同场景 pair 当独立样本。

A2 已正确排除对角线，独立 evaluator 已有 balanced accuracy、AUROC、PR-AUC 和正负样本计数。训练阈值来自配置，而 evaluator 的 0.4/0.05 写死；当前默认一致，未来改阈值需要一并修正。A2 的 GT 几何用于监督，probe 输入只有特征，未发现直接把 GT pose/depth 当作 head 输入的问题。全 prefix 上的 pair readout 合理，但不能宣称在每个早期帧时刻都满足单帧级在线因果性。

以上补充仍为审查和 CPU 合成验证，没有修改实验实现或任务队列。
