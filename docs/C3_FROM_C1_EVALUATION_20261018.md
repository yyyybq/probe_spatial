# C3-from-C1 评估（主目录）— 2026-10-18

配套文档：[C 探针协议](../protocols/C_PROBES.md)、[canonical 正式评估](EVALUATION_PLAN_20261017.md)、
[timestep 正式评估](TSTEP_EVALUATION_20261018.md)。所有 commit 与 tag 只在本地，未 push。

## 结论

- **180 个 C1 checkpoint 的 C3-from-C1 评估全部完成并通过验收**（本次实际执行通过）。代码是发布用的主目录。
- **协议 C3（120 个 canonical C1）**：120 项的 `intervention_validity` 95% 区间下限全部高于机会水平 1/3。
  各模型平均值在 0.35（Aether）到 0.62（DINO）之间。`intervention_margin` 多数接近 0：预测随动作变化的幅度很小。
- **h1 训练的诊断（60 个）**：55 个区间下限高于 1/3，平均值 0.37 到 0.40；它们不是协议 C3，单独报告。
- **主目录与冻结包一致**：C3 运行里 114,240 个候选的 C1 预测与之前冻结包的 C1 评估比较，114,224 个逐位相同，
  其余 16 个只差浮点舍入（相对差不超过 1.1e-7）；目标全部相同。
- 这些是同一轨迹上的终点组成的动作集，结果是多动作敏感性诊断，不是反事实结论。

## 定义与范围

按 `protocols/C_PROBES.md`，C3 不单独训练，而是对冻结的 C1 一步预测器做“动作敏感性”评估：同一锚点（前缀末帧）
在同一条轨迹上的 h1、h2、h4 三个终点组成一个动作集，每个候选动作分别送入 C1，看预测是否更接近与该动作配对的目标。
这些是同一轨迹上的终点，不是恢复状态后执行的反事实分支，所以只能称为多动作敏感性诊断，不能用因果措辞。

主指标 `intervention_validity`：每个动作集内，预测与配对目标的匹配准确率；3 个候选时机会水平为 1/3。
另记 `intervention_validity_chance_adjusted`、`intervention_margin`、`target_to_prediction_sensitivity_pearson` 等。

评估对象是已经做过 v2 评估的全部 C1 checkpoint，共 180 个，按训练时的时间间隔分两类：

| 类别 | 训练时间间隔 | 数量 | 说明 |
|---|---|---:|---|
| `c3_from_c1`（协议 C3） | h1、h2、h4 | 120 | canonical V2 的 C1，验证集本身就按三个间隔展开，评估在训练分布内 |
| `h1_trained_diagnostic` | 只有 h1 | 60 | 44 个 timestep C1 与 16 个 MiniMax 全层 C1；评估时把 `action_horizons` 设为 `[1,2,4]` |

**h1 训练的 60 个不是协议 C3**：它们从未见过 h2、h4 的动作，结果只说明在训练外的动作幅度上 C1 的反应，
要与协议 C3 分开报告，不能放进同一张表比较。

## 做法

- **代码**：发布用的主目录，冻结为评估包 `runtime/main_eval_v2_from_main_206c5de`（commit `206c5de`，305 个文件，
  tag `39f6d18`）。主目录与冻结训练包在数据加载与预测上一致，见[主目录发布一致性](MAIN_TREE_RELEASE_PARITY_20261017.md)。
- **评估项**（`scripts/build_eval_c3_from_c1_items_20261018.py`）：每项从该 checkpoint 已有的 C1 评估项派生，
  保留训练环境、覆盖参数、训练记录与 checkpoint，只加 `++evaluation_protocol=c3_action_sensitivity`（h1 训练的
  另加 `action_horizons=[1,2,4]`）。180 项的实验配置在主目录与训练时逐字节相同。
- **运行器**：`artifacts/c3_from_c1_eval_20261018_c3/run_eval_single.py`，由 r2 运行器派生，只多两处：追加评估项里的
  评估参数、读取 C3 的 summary 文件。
- **自带核对**（`scripts/check_c3_vs_c1_predictions_20261018.py`）：C3 运行中每个候选的 C1 预测与目标，逐条与该
  checkpoint 之前的 C1 评估（冻结训练包）比较。h1 训练的只有 h1 候选有对应记录。
- **验收**同其他评估：v2 评估器、主指标有限、非有限预测为 0、有有效动作集、summary 哈希一致。

## 结果

本次实际执行通过（2026-10-18）。

- **canary**：9 项（协议 C3 五个模型；诊断四项，含 timestep 与 MiniMax 全层），全部通过；协议 C3 的 5 项与 C1 评估逐位相同。
- **正式评估**：180 项（含 canary 重跑），45 个单卡作业，0 失败，180 项全部 ACCEPTED，非有限预测为 0。

**协议 C3**（每项 276 个动作集，每集 3 个候选；机会水平 1/3）

| 模型 | 项数 | 平均 validity | 最低 | 最高 | 去机会后平均 | 区间下限高于 1/3 | 平均 margin | 平均 Pearson |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| DINO | 4 | 0.620 | 0.601 | 0.646 | 0.429 | 4 | 0.0292 | 0.78 |
| VideoMAE | 16 | 0.509 | 0.435 | 0.566 | 0.263 | 16 | −0.00002 | 0.56 |
| Qwen2.5-VL | 4 | 0.478 | 0.461 | 0.499 | 0.217 | 4 | −0.0026 | 0.63 |
| V-JEPA2 | 16 | 0.457 | 0.397 | 0.566 | 0.186 | 16 | 0.0013 | 0.49 |
| Fast3R | 4 | 0.446 | 0.434 | 0.460 | 0.169 | 4 | −0.0005 | 0.32 |
| Wan | 28 | 0.425 | 0.406 | 0.457 | 0.137 | 28 | −0.0008 | 0.34 |
| LingBot | 20 | 0.415 | 0.403 | 0.429 | 0.123 | 20 | −0.0050 | 0.36 |
| CogVideoX | 24 | 0.393 | 0.376 | 0.413 | 0.090 | 24 | −0.0002 | 0.37 |
| Aether | 4 | 0.355 | 0.344 | 0.362 | 0.032 | 4 | −0.0001 | 0.19 |

**h1 训练的诊断**（不是协议 C3）

| 来源 | 模型 | 项数 | 平均 validity | 最低 | 最高 | 区间下限高于 1/3 |
|---|---|---:|---:|---:|---:|---:|
| canonical MiniMax 全层 | MiniMax-H3 | 16 | 0.374 | 0.331 | 0.431 | 11 |
| timestep | CogVideoX | 24 | 0.389 | 0.377 | 0.402 | 24 |
| timestep | MiniMax-H3 | 8 | 0.384 | 0.371 | 0.395 | 8 |
| timestep | Wan | 12 | 0.401 | 0.384 | 0.416 | 12 |

MiniMax 全层的 16 项每项 171 个动作集，评估清单与其他 canonical run 不同，见[正式评估](EVALUATION_PLAN_20261017.md)。

**与 C1 评估的逐条核对**

| 类别 | 项数 | 结果 |
|---|---:|---|
| 协议 C3 | 120 | 828 个候选全部逐位相同 |
| timestep 诊断 | 44 | h1 候选全部逐位相同 |
| MiniMax 全层诊断 | 16 | 每项 171 个 h1 候选中 170 个逐位相同，1 个相对差不超过 1.1e-7 |

不同的那一条每项都是 C1 评估中最后一个批次里单独的样本（171 为奇数，批大小 2）；C3 运行把三个候选放在一起批处理，
矩阵形状不同带来浮点舍入差。MiniMax 的 C1 预测数值很大（最大约 2.9e5），所以绝对差最大 0.031，相对差约 1e-7。

**读法**：几乎所有模型的匹配准确率都高于机会水平，但 `intervention_margin` 接近 0，说明换动作时预测只移动很少，
主要是方向略偏向正确目标；DINO 的 margin 与 Pearson 明显最高。这是同一轨迹终点上的诊断，不能写成反事实或因果结论。

## 复现命令（服务器，`/mnt/umm/users/yinbaiqiao/probe_spatial`）

```bash
PY=/mnt/umm/users/yinbaiqiao/.conda/envs/probe-spatial/bin/python
$PY scripts/build_eval_v2_packs_20261017.py --revision m1 --commit 206c5de
python3 scripts/build_eval_c3_from_c1_items_20261018.py --write
$PY scripts/submit_eval_v2_canonical_20261017.py --revision c3 --mode canary --keys <key ...>
$PY scripts/submit_eval_v2_canonical_20261017.py --revision c3 --mode production --per-job 4
python3 scripts/export_eval_v2_canonical_20261017.py --revision c3 --mode production
$PY scripts/check_c3_vs_c1_predictions_20261018.py --mode production
```

结果表：`results/c3_from_c1_eval/c3/production_metrics.csv`；核对：`results/c3_from_c1_eval/c3/production_c1_parity.json`。
