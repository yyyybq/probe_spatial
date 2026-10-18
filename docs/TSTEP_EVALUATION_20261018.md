# timestep 探针的 v2 正式评估 — 2026-10-18

配套文档：[canonical 正式评估](EVALUATION_PLAN_20261017.md)、[代码版本对齐](CODE_ALIGNMENT_20261017.md)。
所有 commit 与 tag 只在本地，未 push。

## 结论

- **316 个 timestep run 已用 v2 评估器在各自的训练代码上完成正式评估，316 项全部通过验收**（本次实际执行通过）。
  验收条件同 canonical：v2 评估器、主指标有限、非有限预测为 0、有评分记录、summary 哈希一致。
- 范围是 `opt_v3`、`pack_v4_1` 两个冻结包训练的 284 个 run，加上 32 个 B2 fp32 重训。
- **身份说明**：两个 timestep 冻结包只有 10-17 的快照身份（当时没有记录哈希清单），结果表要注明。
  32 个 B2 重训的代码是已提交的 `e4e4886`。
- **没有覆盖**：128 个 MiniMax 第 20/25 层 run（等 stage 5 重训）、单独训练的旧 C3、2026-09-07 的 88 个 run。
- **C3-from-C1**（2026-10-18 补做）：44 个 timestep C1 只用 h1 训练，按诊断方式评估（评估时用 h1/h2/h4 动作集），
  不是协议 C3，见 [C3-from-C1 评估](C3_FROM_C1_EVALUATION_20261018.md)。

## 范围

清单 `metadata/release_inventory/timestep.csv` 共 576 个 timestep run（Wan、CogVideoX、MiniMax-H3 × 8 种探针 ×
timestep × 层 × prefix）。本轮评估 316 个：

| 来源 | 训练代码 | 评估项 | 说明 |
|---|---|---:|---|
| `tstep_probe_opt_v3` 三个 run id | 冻结包 `code/tstep_probe_opt_v3` | 254 | 清单状态 NOT_VERIFIED |
| `tstep_probe_pack_v4_1` 四个 run id | 冻结包 `code/tstep_probe_pack_v4_1` | 30 | 清单状态 NOT_VERIFIED |
| 2026-10-16 B2 fp32 重训 `tstep_probe_b2_fp32_20261016_r1` | 主目录 `e4e4886` | 32 | 替代 32 个权重非有限的 B2 run |

探针：A1 depth、A1 mask、A2、B1、B2、C1、C2。不在本轮：

- 128 个 MiniMax 第 20/25 层 run：特征缓存溢出，等 stage 5 用 bf16 特征重训；
- 单独训练的 `counterfactual`（旧 C3 协议）。按 `protocols/C_PROBES.md`，C3 是在 C1 checkpoint 上的评估，
  需要主目录的 C1 动作集数据集，冻结包里没有；canonical 评估同样没有包含 C3；
- 2026-09-07 的 88 个 `tstep_probe_seeded_v1` run：训练代码无法恢复。

## 做法

与 canonical 相同：**在训练时的代码上评估，只把评估器换成 v2**（非有限预测计为无效）。

- **评估包**（`runtime/tstep_eval_v2_from_*`，均已冻结并记为 tag）：

| 评估包 | 基础 | 替换 / 补丁 | tag commit |
|---|---|---|---|
| `tstep_eval_v2_from_tstep_probe_opt_v3` | `code/tstep_probe_opt_v3` 的树 | 换 `eval_diag.py`、`eval_a1_streaming.py`、`loss.py`（`994a8e6`）；C 快速路径补 C2 评估字段（同 canonical r2） | `8fbf9e1` |
| `tstep_eval_v2_from_tstep_probe_pack_v4_1` | `code/tstep_probe_pack_v4_1` 的树 | 同上 | `a9cb990` |
| `tstep_eval_v2_from_main_e4e4886` | commit `e4e4886` 的 `vidfm3d/`、`configs/` | 同样换三个评估文件（与 `e4e4886` 只差一行注释） | `6294d04` |

  这两个 timestep 冻结包当时没有记录哈希清单，身份是 10-17 记下的快照 tag。建包时逐文件核对
  `runtime/<包>/source` 仍与 tag 相同。结果表里这些行要注明“快照身份”。
- **评估项**（`scripts/build_eval_v2_tstep_items_20261018.py`）：逐个 run 读取它自己的训练覆盖参数
  （`.hydra/overrides.yaml`），去掉日志、路径、worker 数等执行参数；按训练启动脚本重建环境变量，
  并与该 run 的 `run_manifest.json` 核对数据根目录、场景清单及其 SHA-256、流式与目标特征根目录。
  冻结包 run 使用 fp16 直通与按探针的快速路径；B2 重训使用 fp32 特征、主目录原路径，与训练时一致。
  316 项全部通过核对，0 个问题。
- **划分与指标**：`eval_split=val`。主指标同 canonical；B1 用 `mean_az_err_deg`，另记仰角误差与对数距离误差。

## 进度与结果

本次实际执行通过（2026-10-18）。

- **canary**：13 项，覆盖 7 种探针、3 个评估包和 3 个模型，13 个单卡作业，全部通过验收。
- **正式评估**：316 项（含 canary 的 13 项重跑），79 个单卡作业，每作业 4 项，0 失败，316 项全部 ACCEPTED。

| 探针 | `opt_v3` | `pack_v4_1` | B2 重训 | 合计 |
|---|---:|---:|---:|---:|
| A1 depth | 36 | 12 | | 48 |
| A1 mask | 33 | 15 | | 48 |
| A2 | 44 | | | 44 |
| B1 | 42 | 2 | | 44 |
| B2 | 12 | | 32 | 44 |
| C1 | 44 | | | 44 |
| C2 | 43 | 1 | | 44 |

- **可重复性**：canary 的 13 项在正式评估里又跑了一次。11 项主指标逐位相同；另 2 项是 A1 mask，
  差 7.2e-7 与 6.5e-6。这是 identity loss 用 CUDA `scatter_add_` 求和的已知运行间噪声，见
  [主目录发布一致性](MAIN_TREE_RELEASE_PARITY_20261017.md)第四节。
- C1 每项 276 条记录（canonical 是 828 条）：timestep 的 C1 配置只用时间间隔 1（`action_horizons: [1]`），
  canonical 的 h124 配置用 1、2、4，验证时每个前缀展开成三条。

## 复现命令（服务器，`/mnt/umm/users/yinbaiqiao/probe_spatial`）

```bash
PY=/mnt/umm/users/yinbaiqiao/.conda/envs/probe-spatial/bin/python
$PY scripts/build_eval_v2_packs_20261017.py --revision t1 --commit 994a8e6
$PY scripts/build_eval_v2_packs_20261017.py --revision t1m --commit 994a8e6
python3 scripts/build_eval_v2_tstep_items_20261018.py --write
$PY scripts/submit_eval_v2_canonical_20261017.py --revision t1 --mode canary --keys <key ...>
$PY scripts/submit_eval_v2_canonical_20261017.py --revision t1 --mode production --per-job 4
python3 scripts/export_eval_v2_canonical_20261017.py --revision t1 --mode production
```

结果表：`results/tstep_eval_v2/t1/production_metrics.csv`。
