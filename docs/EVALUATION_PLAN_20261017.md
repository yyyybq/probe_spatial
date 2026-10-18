# 正式评估：为什么还没有可发布结果，以及补齐方案 — 2026-10-17

配套文档：[代码版本对齐](CODE_ALIGNMENT_20261017.md)。所有 commit 与 tag 只在本地，未 push。

## 结论

- **canonical V2 的 checkpoint 从来没有做过正式评估。** 训练结束后每个 run 目录里只有 checkpoint、
  训练日志与收据，没有评估输出。清单里的 “validation” 指训练前的输入 canary，不是对训练好的探针打分。
  所以 865 个训练完成、692 个身份核验通过，可发布结果仍是 0。
- **timestep 只评估过 A1、A2、B2 三类，而且评估代码与训练代码不是同一版本。** 2026-09-23 的评估战役
  `tstep_eval_a1a2b2_20260923_r1` 用当时主目录的 v1 评估器，训练用的是冻结包 `tstep_probe_opt_v3`，
  因此那 136 行只算“匹配但未验收”。B1、C1、C2、mask 没有评估，C3 是旧协议。
- **另一个原因是 v1 评估器会给发散的模型打出看似正常的分数**（NaN logits 变成类别 0），必须换成 v2。
- **正式评估本身不贵。** canary 实测每个 checkpoint 2.5–4.4 分钟；676 个约 40–50 GPU·h。
- **现状（2026-10-18）**：canonical 合并表 804 项全部通过验收。
  - 676 个已核验 run：r2 一次通过 668 项；另 8 项是评估项绑错了 owner overlay，按训练回执改正后用 r3 重评，全部通过。
  - r4 补评 128 项，全部通过：16 个 B2 fp32 重训、96 个 MiniMax 全层六探针 run、16 个 MiniMax 第 30 层 run。
  - 合并表：`results/canonical_v2_eval_v2/final/production_metrics.csv`（见下文“正式评估”与“补充评估 r4”）。

## 做法

1. **评估用训练时的代码。** 主目录的数据加载路径与训练时不同（见对齐文档），所以评估在训练冻结包的
   副本上运行，只替换 3 个文件为 v2：`vidfm3d/eval_diag.py`、`vidfm3d/eval_a1_streaming.py`、
   `vidfm3d/utils/loss.py`（来自 commit `994a8e6`）。这样的评估包有 3 个，均已冻结并记为 tag：

| 评估包 | 基于训练包 | commit |
|---|---|---|
| `main_v2_eval_v2_from_main_v2_probe_v2_r7_single_v3` | `code/main_v2_probe_v2_r7_single_v3` | `2495bc5` |
| `main_v2_eval_v2_from_main_v2_probe_v2_aetherfix_r1` | `code/main_v2_probe_v2_aetherfix_r1` | `6ab0c5c` |
| `main_v2_eval_v2_from_main_v2_probe_minimax_bf16_v2` | `code/main_v2_probe_minimax_bf16_v2` | `c84da76` |

2. **评估项从训练项派生。** 每个评估项复制训练项的实验配置、科学参数与执行 manifest，按训练运行器
   重建环境（owner overlay、fp16 直通设置），绑定 checkpoint 路径与大小；运行器在 GPU 节点上先逐项核对
   哈希，再计算 checkpoint 的 SHA-256 写进结果。
3. **划分与指标沿用 2026-09 的正式评估口径**：`eval_split=val`，不用训练时的 W&B 日志。

| 探针 | 评估器 | 主指标 |
|---|---|---|
| A1 depth | `eval_a1_streaming` | `mean_loss_depth`（另记 `mean_Auc_30`） |
| A1 mask | `eval_a1_streaming` | `mean_loss_identity` |
| A2 view consistency | `eval_diag` | `balanced_acc` |
| B2 ego belief v2 | `eval_diag` | `mean_ang_err_deg` |
| C1 action dynamics | `eval_diag` | `global_R@1`（另记无动作、打乱动作、末帧对照） |
| C2 path integration | `eval_diag` | `global_R@1` |

4. **验收**（`scripts/export_eval_v2_canonical_20261017.py`）：v2 评估器版本、主指标有限、非有限预测为 0、
   至少有一条被评分的记录、summary 哈希与收据一致。不满足的行标为 REJECTED 并写明原因。

## 覆盖范围

- 已为 692 个已核验 canonical run 中的 **676 个** 生成评估项（656 个 r7 包，20 个 Aether 包）。
- 10-17 未覆盖的三组（16 个 MiniMax L30 A/B16、MiniMax 全层战役、16 个 canonical B2 重训）已在
  2026-10-18 用 r4 补评，见下文“补充评估 r4”。MiniMax 全层战役里 36 个不属于六探针协议的 run 不评估。

## 进度（2026-10-17）

**r1 canary**（V-JEPA2 L11 p12 的 6 个探针 + Aether L1 p12 A2，7 个单卡作业，本次实际执行）

- 第 1 次 7 个全部在启动时失败：checkpoint 文件名里的 `=` 没有加引号，Hydra 无法解析参数。
  修复后先用 Hydra 只组合配置的模式核对了 7 个评估项，再提交第 2 次。
- 第 2 次 6 个通过、C2 失败：

| run | 主指标 | 值 | 评分/载入 | 用时 |
|---|---|---:|---:|---:|
| V-JEPA2 A1 depth | `mean_loss_depth` | 0.313（`mean_Auc_30` 0.081） | 276/276 | 3.0 min |
| V-JEPA2 A1 mask | `mean_loss_identity` | 2.92 | 276/276 | 2.8 min |
| V-JEPA2 A2 | `balanced_acc` | 0.921 | 276/276 | 2.6 min |
| V-JEPA2 B2 | `mean_ang_err_deg` | 37.6（95% 区间 32.7–42.9） | 171/276 | 3.5 min |
| V-JEPA2 C1 | `global_R@1` | 0.0157（无动作 0.0048，打乱动作 0.0109） | 828/828 | 2.5 min |
| Aether A2 | `balanced_acc` | 0.684 | 276/276 | 4.4 min |

  所有行非有限预测为 0，checkpoint 的 SHA-256 已记录。B2 的 171/276 与 2026-09 timestep 评估的有效比例一致
  （无可见目标的样本不计分）。

**C2 失败原因与 r2 修复**

- 训练包的 C 探针快速取样路径 `_getitem_feature_action_diag` 只产出训练要用的字段；v2 评估器算 C2 还需要
  相机位姿和时间间隔（`target_extrinsics`、`start_extrinsic`、`action_horizons`），用于终点位姿误差、漂移率和
  分时间步误差。主目录后来给这个路径补上了这些字段，但主目录又缺 owner overlay，不能直接用。
- r2 评估包只在数据加载器里改两处：总是重新计算几何量（即标签缓存未命中时执行的同一段代码），并按主目录的
  算法补上这三个字段。训练输入由未改动的代码产生。MiniMax 包的这个函数与另两个不同，本轮不涉及。

| r2 评估包 | 基于 | commit |
|---|---|---|
| `main_v2_eval_v2r2_from_main_v2_probe_v2_r7_single_v3` | `code/main_v2_probe_v2_r7_single_v3` | `2d818ac` |
| `main_v2_eval_v2r2_from_main_v2_probe_v2_aetherfix_r1` | `code/main_v2_probe_v2_aetherfix_r1` | `2379275` |

**r2 canary**（4 个作业，全部通过，本次实际执行）

- V-JEPA2 C2：`global_R@1` 0.0242，终点位姿误差 0.121；Aether C2：`global_R@1` 0.0024。
- V-JEPA2 C1 与 A2 在 r2 上的主指标和全部次指标与 r1 逐位相同，说明补丁不影响这两条路径、评估可复现。

**正式评估**：r2 上 676 个评估项，每作业 4 个，共 169 个单卡作业（08:41–12:45 UTC，本次实际执行）。

- 导出结果：**668 项 ACCEPTED**（v2 评估器、主指标有限、非有限预测为 0、有评分记录、summary 哈希一致），
  **8 项 FAILED**，0 项 REJECTED。表：`results/canonical_v2_eval_v2/r2/production_metrics.csv`。
- 8 项失败全是 VideoMAE 前缀 24 的 C1 与 C2（层 5、11、17、23 各 2 项），**原因是评估项绑错了 overlay，
  不是数据丢失**（10-17 晚的初步判断“数据缺失”有误，2026-10-18 更正）：
  - 这 8 个 checkpoint 由 9 月 23 日的重试运行器 `runtime/main_v2_probe_v2_vm24_targetfix_r1/run_single.py`
    训练，overlay 是 `probe552_validation_40_20260923_r2/overlay/manifest.json`（sha256 `291cebd5…`），
    目标特征读的是规范目录 `FEAT_TARGET_CANONICAL_V2_GAP_DELTA_V1` 的副本，这 2,724 个对象现在都在 AOSS 上，
    全量 SHA 核对一致。
  - 生成评估项的脚本按输出目录匹配运行器。重试运行器把 checkpoint 写进原 r7 战役的目录，回执却写在自己的目录，
    所以脚本找到的是 r7 运行器和它的 overlay（`9671fc95…`）。r7 overlay 指向的旧目录
    `FEAT_TARGET/videomae/` 在 2026-09-19 到 09-21 之间已和其他旧命名空间一起被删除，评估因此读不到文件。
- **改正（本次实际执行通过）**：
  - 用作业读出全部训练回执（root 权限文件），逐项核对评估项与训练回执的 overlay、训练项和代码包哈希：
    668 项一致；不一致的正是这 8 项，且只差 overlay 与运行器。证据：
    `artifacts/release_audit_20261018/overlay_binding_audit/BINDING_AUDIT.json`。
  - 生成脚本改为按 checkpoint 的 run id 在训练提交记录里找运行器，并要求与训练回执一致
    （`--revision r3 --receipts …`）。用新逻辑重建全部 676 项：668 项与 r2 逐字节相同，这 8 项只在 overlay、
    运行器和对应环境变量上不同。
  - 只为这 8 项写了 r3 评估项（`eval_v2_canonical_20261017_r3`，同一 r2 评估包），canary 2 项后正式评估 8 项，
    全部通过验收；层 11 的两项与 canary 数值相同。
- 合并表 `results/canonical_v2_eval_v2/final/production_metrics.csv`：676 项全部 ACCEPTED，
  其中 8 行来自 r3（`eval_revision` 列）。

## 补充评估 r4：B2 重训与 MiniMax（2026-10-18）

本次实际执行通过。r2 没有覆盖的三组 canonical run，用同样方法评估（训练代码 + v2 评估器，逐项按训练记录核对）：

| 组 | run 数 | 训练代码 / 评估包 | 训练记录核对 |
|---|---:|---|---|
| B2 fp32 重训（替代 16 个权重非有限的 CogVideoX L0/5/10 与 Aether L1 B2） | 16 | r7 / Aether 包，`main_v2_eval_v2r2_from_*` | 训练回执的状态、训练项、代码包、overlay 全部一致 |
| MiniMax-H3 全层战役，六探针部分 | 96 | `main_v2_probe_minimax_bf16_v2`，新建 `main_v2_eval_v2r2_from_main_v2_probe_minimax_bf16_v2`（同 r2 的 C 路径补丁，tag `b3df00a`） | 训练回执一致 |
| MiniMax-H3 第 30 层 A/B16 | 16 | 同上 | 正式训练命令（环境变量与覆盖参数）逐项复原，与 run 的 `run_manifest.json` 一致 |

- 生成脚本 `scripts/build_eval_v2_canonical_extra_items_20261018.py`，评估项在 `eval_v2_canonical_20261017_r4`。
  训练回执由 SCO 作业读出（`artifacts/release_audit_20261018/overlay_binding_audit/training_receipts_extra.jsonl`）。
- 不在本轮：MiniMax 全层战役中的 20 个 `streaming_camera` 与 16 个单独训练的 `counterfactual`（不属于六探针协议）。
- 一项清单记录过时：`minimax_h3 … l50 p24 streaming_mask` 的清单记的是 10-15 仍在训练时的 epoch 95，
  这个 run 已训练完成（epoch 99，回执 SUCCEEDED），评估用的是最终 checkpoint，并在评估项里注明。
- **可比性注意**：MiniMax 全层战役的训练与评估清单是 `frozen_queue_r2`（p12：训练 1,523、验证 171 个样本），
  其他 canonical run（包括 MiniMax 第 30 层）用的是 `execution_manifest_r3_receipt_union`（训练 2,445、验证 276）。
  所以 MiniMax 全层的数字是在较小的验证集上得到的，不能与其他模型的 canonical 数字直接并列。

结果（本次实际执行通过）：canary 10 项（两组 B2 重训、MiniMax 全层每种探针 1 项、第 30 层 B2 与 A1 mask），
全部通过；正式评估 128 项、32 个单卡作业、0 失败，**128 项全部 ACCEPTED**。canary 的 10 项在正式评估中重跑，
8 项主指标逐位相同，2 项 A1 mask 差不超过 1.1e-7（CUDA `scatter_add_` 的运行间噪声）。

| 组 | 评估项 | 通过 | 每项验证样本 |
|---|---:|---:|---:|
| B2 fp32 重训 | 16 | 16 | 276 |
| MiniMax 全层（六探针） | 96 | 96 | 171 |
| MiniMax 第 30 层 A/B16 | 16 | 16 | 276 |

合并表 `results/canonical_v2_eval_v2/final/production_metrics.csv` 现为 **804 项，全部 ACCEPTED**
（r2 668 项、r3 8 项、r4 128 项，见 `eval_revision` 列）。

## 复现命令（服务器 `/mnt/umm/users/yinbaiqiao/probe_spatial`）

```bash
/mnt/umm/users/yinbaiqiao/.conda/envs/probe-spatial/bin/python scripts/build_eval_v2_packs_20261017.py --revision r2 --check
python3 scripts/build_eval_v2_canonical_items_20261017.py --revision r2 --all-verified
/mnt/umm/users/yinbaiqiao/.conda/envs/probe-spatial/bin/python scripts/submit_eval_v2_canonical_20261017.py --revision r2 --mode production
python3 scripts/export_eval_v2_canonical_20261017.py --revision r2 --mode production
# 2026-10-18 改正：核对绑定，重建并重评 8 项，合并
python3 scripts/audit_eval_item_binding_20261018.py --revision r2
python3 scripts/build_eval_v2_canonical_items_20261017.py --revision r3 --keys <8 个 key> \
    --receipts artifacts/release_audit_20261018/overlay_binding_audit/training_receipts.jsonl --write
/mnt/umm/users/yinbaiqiao/.conda/envs/probe-spatial/bin/python scripts/submit_eval_v2_canonical_20261017.py --revision r3 --mode production --per-job 2
python3 scripts/export_eval_v2_canonical_20261017.py --revision r3 --mode production
python3 scripts/export_eval_v2_canonical_final_20261018.py
# 2026-10-18 r4：B2 重训与 MiniMax
$PY scripts/build_eval_v2_packs_20261017.py --revision r2m --commit 994a8e6
$PY scripts/build_eval_v2_canonical_extra_items_20261018.py --write
$PY scripts/submit_eval_v2_canonical_20261017.py --revision r4 --mode production --per-job 4
python3 scripts/export_eval_v2_canonical_20261017.py --revision r4 --mode production
python3 scripts/export_eval_v2_canonical_final_20261018.py
```

（`$PY` 为 `/mnt/umm/users/yinbaiqiao/.conda/envs/probe-spatial/bin/python`。）

训练回执是 root 权限文件，由 SCO 作业用 `artifacts/release_audit_20261018/overlay_binding_audit/dump_training_receipts.py`
读出（命令见同目录 `job_command.txt`）。

结果表：`results/canonical_v2_eval_v2/r2/production_metrics.csv`。

## 待你决定

1. **timestep**：已按 (a) 执行（2026-10-18）：为 `tstep_probe_opt_v3`、`tstep_probe_pack_v4_1` 和
   B2 重训用的 `e4e4886` 建评估包，评估 316 个 run，见 [timestep 正式评估](TSTEP_EVALUATION_20261018.md)。
   2026-09-07 的 88 个 run 没有可恢复的训练代码，不进入本轮。
