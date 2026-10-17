# 代码发布独立审核 — 2026-10-15

本次在 [2026-10-11 审查](RELEASE_AUDIT_20261011.md) 之后独立复核 `probe_spatial`
（远端 `/mnt/umm/users/yinbaiqiao/probe_spatial`，分支 `d-series-paper-ready`，
HEAD `2f19ecb`，工作区 747 个未提交变更）。相邻目录 `VAGEN-Lite` 是另一项目
（InteriorGS QA），未改动。审核沿实际运行路径进行：数据选择 → 特征读取 →
probe 训练 → 权重选择 → 独立评估 → 指标聚合 → 导出；重点核对 10-11 之后的
代码变化与实验产物。本次没有重训任何实验，没有改写任何历史指标文件；
新评估器输出带 `evaluator_version` 字段，不覆盖旧结果。

术语：**probe** 是在冻结基础模型表征上训练的轻量诊断模型；**checkpoint**
是保存的模型权重文件；**非有限值** 指 NaN 或 ±inf。

## 结论

1. **10-11 之后主目录只有 1 处源码改动**（`vidfm3d/utils/feature_layers.py`，
   10-13，为 MiniMax 文件名加入时间步后缀），与 MiniMax 战役冻结的 runtime
   副本一致，予以保留并补全。其余 10-11 后的新文件是 timestep 评估脚本、
   MiniMax/LingBot 提交与迁移脚本，以及与本项目无关的 `spatial_paired_*`、
   `spatial_six_tasks_*`、`sites/`、`paired_*` 数据集目录（InteriorGS 空间
   配对题，应从公开源码树排除）。
2. **发现并修复一类会把无效训练记成"成功"的缺陷**：特征缓存或权重出现
   NaN 时，损失函数把 NaN 损失置零，训练照常跑完并写 `TRAIN_COMPLETE`；
   两个评估器随后把 NaN 预测记为 0 损失、AUC 0、"全部判负"或"第 0 个
   bin"。timestep 库存里 80 个"退化"评估行与此直接对应。本次在训练模块
   加入输入有限性拦截，评估器改为标记并排除非有限预测并计数。
3. **权重张量首次逐文件检查**（登录节点 CPU，只读）：timestep 和 canonical
   V2 两张清单中可加载的小型 checkpoint 全部检查，大型 A1 checkpoint 抽样。
   结果见 [实验清单](EXPERIMENT_INVENTORY_20261015.md)：CogVideoX、Aether、
   MiniMax 的 B2（ego_belief_v2）训练存在整批 NaN 权重，这些训练之前被记为
   "完成且身份核验"。它们需要重训，不能靠重新评估修复。10-16 补充：这些 B2
   都在 fp16 特征直通下训练；CogVideoX L30/35/40 以 fp32 直通重跑后权重有限，
   canonical MiniMax 读 bf16 缓存的 B2 也有限。重训只需改直通精度，不需重提特征。
4. **LingBot C40 已完成**（10-11 时未确认），**MiniMax-H3 全层战役 132 项
   中 129 项完成、3 项仍在运行**（2026-10-15 23:00 UTC 调度器与本地产物
   一致）。战役包含 36 项六探针协议之外的训练（`streaming_camera` 20、
   `counterfactual` 16），已在清单中单独标注。
5. 普通 CPU 测试基线 446 通过；修复后 463 通过（新增 17 项回归测试）。
   没有删除任何测试；7 个依赖冻结产物的测试模块仍为显式可选。

## 一、代码问题、依据与修复

| # | 位置 | 问题 | 依据 | 修复 | 验证 |
|---|---|---|---|---|---|
| 1 | `vidfm3d/utils/loss.py` `check_and_fix_inf_nan`；`VideoProbeLitModule.model_step`、`ProbeExtensionLitModule.model_step` | 损失函数把 NaN/inf 损失替换为 0。特征缓存非有限时训练不会报错，权重变 NaN 仍写 `TRAIN_COMPLETE`，日志显示零损失 | timestep `minimax_h3 t500 L20 p8 streaming_depth` 终点 checkpoint 355,144,435/355,144,437 个参数非有限；官方评估 `mean_loss_depth=0, Auc_30=0`；同层 A2 预测 100% NaN；同层 L15 权重全部有限 | 新增 `require_finite_input`，两个训练模块在前向前检查 `vfm_feat`/`input_feat*`/`target_feat*`，发现非有限值即抛 `NonFiniteInputError` 并指出样本 `scene_path` | `tests/test_nonfinite_handling.py`（4 项） |
| 2 | `vidfm3d/eval_a1_streaming.py` | 非有限预测经同一损失函数被记为 0 损失；AUC 对 NaN 比较得 0 | 同上 16 个 MiniMax depth 评估行 loss=0、AUC=0 | 评估器 v2：每条记录带 `input_finite`/`prediction_finite`，非有限预测损失记 NaN 不参与均值；摘要新增 `n_nonfinite_prediction`、`n_valid_prediction`、`evaluator_version`，`invalid_manifest_ids` 不再恒为空 | 同上（2 项） |
| 3 | `vidfm3d/eval_diag.py` B2/B1/A2/C1 记录与 `_summarize` | NaN logits 的 `argmax` 返回 0，B2 被记为恒定 bin 0 的角误差（Cog 与 MiniMax 四个 prefix 的值完全相同）；A2 的 NaN 概率 `>0.5` 为假，被记为全部判负（balanced_acc=0.5、neg_acc=1） | Cog/MiniMax timestep B2 `val_predictions.pt` 中 `pred_bin` 唯一值为 1 个（0），`logd_err` 为 NaN；对应 checkpoint 99.7% 参数非有限；Wan B2 有 35 个不同 bin | 记录加 `prediction_finite`；`_summarize` 对 A2/B1/B2/C1 排除非有限预测并输出 `n_scored`、`n_nonfinite_pred`；`_valid_evaluation_record` 把非有限预测计入无效；摘要加 `evaluator_version`、`n_nonfinite_pred_records` | 同上（3 项） |
| 4 | `vidfm3d/eval_diag.py` `_summarize` A2 | 正负样本阈值硬编码 0.4/0.05，与训练模块 `view_overlap_pos/neg_threshold` 超参脱节 | 代码阅读 | 阈值作为参数，`main` 从已加载模块的 `hparams` 读取并写入摘要 | 同上（1 项） |
| 5 | `vidfm3d/utils/feature_layers.py` | `lingbot` 不在时间步后缀集合，`feature_layer` 覆盖时会找 `feature_layer10.sft`，磁盘文件是 `feature_t250_layer10.sft`；`minimax_h3`/`lingbot` 无注册规格，缺省通道数回落 1536、网格回落 18×32；无时间步时硬编码 749（这两个模型是 250） | 磁盘文件名 `FEAT_STREAMING/lingbot/scannetpp/48da625e8c/window_0003/prefix_000007/feature_t250_layer*.sft`；缓存张量形状 MiniMax `(T,15,26,5376)`、LingBot `(T,30,52,2048)`；提取器文档字符串 | 注册 `minimax_h3`（5376 通道，50 层，15×26 网格，t=250）与 `lingbot`（2048 通道，48 层，30×52 网格，t=250）；`default_feature_timestep` 从规格推导；`resolve_feature_layers.py` 增加 `default_timestep` 字段；`run_streaming_probe_sweep.sh` 的 `T` 缺省改为按模型查注册表 | 同上（3 项）+ `bash -n` |
| 6 | `vidfm3d/models/video_probe_module.py` `training_step` | `train_viz_freq // 20` 为 0 时取模报错 | 代码阅读 | `max(..., 1)` | 既有测试通过 |
| 7 | `vidfm3d/utils/feature_layers.py` 10-13 未提交改动 | 需确认是否与实验实际使用的实现一致 | `runtime/main_v2_probe_minimax_bf16_v2/source` 中同函数相同 | 保留 | 对比完成 |

历史运行使用的是冻结 runtime 副本，这些修复不会改变任何已完成训练的权重
或已导出的指标；它们只阻止新的无效训练，并让重新评估时的无效样本被计数
而不是记为满分。

## 二、澄清的接口、配置与实验定义

- **MiniMax-H3 fp16 特征缓存溢出，但只影响 timestep 实验**（10-16 修正）：
  fp16 最大值 65,504。GPU 任务 `pt-et0ezc14` 读取 AOSS 上 timestep 命名空间
  `FEAT_STREAMING_CANONICAL_V2_SEEDED_V1`（3 个验证场景 × 2 个 prefix）：
  MiniMax L15 全部有限（最大绝对值约 2.9 万），**L20/L25 在 t500 与 t1000 的
  24 个文件全部含 inf**（每文件 8–60 个，最大有限值 65,280）；对应的
  `FEAT_TARGET_CANONICAL_V2_SEEDED_V1` 目标缓存 12 个采样文件中 10 个含 inf。
  CogVideoX（最大约 1.6 千）与 Wan（约 23）正常。timestep MiniMax L20/L25 的
  128 个训练因此无效。
  canonical t=250 的情况不同：10-15 当天我用登录节点本地 `FEAT_STREAMING/minimax_h3`
  的 fp16 文件（L20 每文件 59 个非有限、L49 55,527 个）推断 canonical MiniMax
  L≥20 也不可靠，这个推断是错的——正式 canonical 运行（L30 A/B16 与全层战役）的
  `run_manifest.json` 与 `START_RECEIPT.json` 记录的输入是
  `FEAT_STREAMING/FEAT_TARGET_CANONICAL_V2_MINIMAX_BF16_V2`，即团队 9-23 起用
  bf16 存储重提的缓存。GPU 任务 `pt-ljbe3zu0`（10-16）采样该命名空间 L10/20/30/40/49
  的 30 个 streaming 文件与 12 个 target 文件：全部 bf16、全部有限，最大绝对值
  达 3.7×10^6（第 40 层），说明 bf16 存储是必要的，也说明 canonical MiniMax
  训练的输入没有溢出。10-15 清单对这 128 个 canonical run 的"缓存溢出阻塞"标记
  已撤回，改记 `BF16_V2_CACHE_SAMPLE_FINITE_42_FILES`。
  原始记录：`artifacts/release_audit_20261015/feature_finiteness/`。
  修复方向：timestep 的 L20/L25 按已有的 BF16 路线（`prepare_feature_for_storage`
  只存在于 `runtime/main_v2_minimax_bf16_v3_extractor/source`，主目录没有）重提并重训；
  LingBot 本地 5 层缓存全部有限（最大约 40），不受影响。方案见
  [MiniMax 重提与重训计划](MINIMAX_REEXTRACT_RETRAIN_PLAN_20261016.md)。
- **MiniMax-H3 全层战役的探针集合**：132 项 = L20/L40/L50 各 32 项、L10 20 项、
  L30 16 项；每层包含 `streaming_camera`（单独相机头）和 `counterfactual`
  （旧独训 C3）各 4 项。两者都不属于 canonical V2 的六探针协议（协议规定
  A1 姿态由 `streaming_depth` 内的相机头给出，C3 为复用 C1 权重的评估）。
  清单把这 36 项标为 `canonical_v2_minimax_all_layers_20261011_noncanonical_probe`，
  不计入主表格子。逻辑 L50 对应磁盘层 49。
- **B2 的特征传输精度**：`PROBE_FP16_PASSTHROUGH=1` 让 fp16 特征直到 GPU 才
  升精度。9-28 的 `single_gpu_cog12_ego_fp32` 已对 CogVideoX L30/35/40 的
  B2 改用 `PROBE_FP16_PASSTHROUGH=0`，但更早的 CogVideoX L0/5/10 和 Aether L1
  的 B2 没有重跑，其 checkpoint 仍为 NaN。本次没有重新训练去证明精度是唯一
  原因；重训前应用一个 cell 做对照。
- **评估划分**：所有已有评估均为 `val`，`eval_diag.py` 对 `test` 需显式冻结
  manifest，不能更名。
- **公开入口脚本的时间步**：`scripts/run_streaming_probe_sweep.sh` 过去对所有
  模型默认 `T=749`；MiniMax-H3/LingBot 的缓存是 t=250。现按注册表取默认值，
  显式 `T=` 仍优先。
- **README 中的安装与测试命令**未变；`vidfm3d/eval_a1_streaming.py` 此前从未
  被 git 跟踪（仍为未跟踪新文件），发布提交必须显式加入。

## 三、测试整理

- 基线：`python -m pytest -q` 446 通过、1 项 artifact 用例取消选择（305 秒）。
  修复后 463 通过（278 秒），新增 `tests/test_nonfinite_handling.py` 17 项。
- 逐类判断（74 个测试模块、约 420 个用例）：
  - 覆盖真实错误或重要行为：核心回归（probe 合约、指标加权、训练入口、
    seed、split、AOSS 错误、离线缓存、噪声种子、C3-from-C1、D-series、
    SE3、validity）保留。
  - 只重复实现细节：`tests/windowfix_v2/` 与多数 `test_tstep_*`、
    `test_wan_target_*` 是集群运维脚本（提特征排队、AOSS 同步、提交清单）
    的合成输入测试。它们能在 CPU 上无数据运行，保护的是曾经出错的运维合约，
    本次保留；它们不证明 GPU 或集群执行成功。
  - 随机数据未参与计算：复核了使用 `torch.randn` 的 9 个模块，随机张量均
    被传入被测函数，未发现新的无效用例。
  - 依赖作者机器文件：7 个 artifact 模块已在 `conftest.py` 隔离；另有 5 个
    模块把 `/mnt/umm/...` 路径仅作为字符串常量写入合成清单，不触碰文件系统。
  - 对应不存在的功能、完全重复：未发现。`test_offline_hf_cache_paths.py`
    与 `windowfix_v2/test_offline_hf.py` 测的是同一模块的不同合约（缓存目录
    优先级 vs 快照解析），不重复。
- 未删除任何测试。

## 四、防御性检查的取舍

- 保留：exact target 身份、window/prefix 身份、owner catalog fail-closed、
  temporal validity、`eval_split` 白名单。这些防止的是科研结论错误。
- 新增：训练前输入有限性检查。代价是每批一次归约；收益是避免再出现跑满
  100 个 epoch 的 NaN 训练（本次至少 16 个 MiniMax A1 训练、44 个 B2 训练
  属此类）。
- 未改：`check_and_fix_inf_nan` 对损失的置零逻辑（来自上游 VidFM3D，影响
  历史复现）；`datamodule` 用 `eval()` 解析数据集字符串（上游设计，公开后
  应在 README 标注只接受受信配置）。

## 五、公开发布相关检查

- **凭据**：源码、配置、脚本、文档中未发现 AWS/HF/GitHub 令牌、AOSS 密钥或
  代理凭据模式；`~/.bashrc` 中的代理凭据不在仓库内。`.test-tstep-sync-*/petreloss.conf`
  是测试生成的合成文件，已被 `.gitignore` 排除。
- **个人绝对路径**（公开前需改为环境变量或文档示例）：
  `features/run_inscene15k.py` 的 `VFM_DEFAULTS` 中 MiniMax/LingBot 模型路径；
  `features/minimax_h3/minimax_h3_feature.py`、`features/lingbot/lingbot_feature.py`
  的默认模型路径；`vidfm3d/eval_a1_streaming.py` 文档字符串；
  `tests/test_tstep_sync_loop.py`、`tests/test_wan_target_h124_sco_execution.py`、
  `tests/windowfix_v2/test_aoss_streaming_reference_gate.py` 的字符串常量。
  `scripts/` 中 307 个文件含集群路径，属运维历史。
- **不应进入公开源码树的目录**：`spatial_paired_20261014_v1/`、
  `spatial_paired_20261015_v2/`、`spatial_six_tasks_2026101*_v*/`、`sites/`、
  `paired_release_tools/`、`paired_v2_inventory/`、`analysis/interiorgs_v1_flow_audit/`
  （InteriorGS 问答数据，属另一项目）；`runtime/`、`model_cache/`、`robocasa365/`
  已被忽略。
- **许可**：`THIRD_PARTY.md` 所列缺口（Fast3R/DUSt3R、VGGT、Aether 的许可文件）
  未变化，仍是发布阻塞项。
- **发布提交**：554 个未跟踪文件中包含正式源码（`vidfm3d/eval_a1_streaming.py`、
  `vidfm3d/utils/aoss_fs.py`、`vidfm3d/utils/action_validity.py` 等）和全部
  新测试；`git add -u` 会遗漏它们。

## 六、本次实际执行的验证

| 检查 | 结果 |
|---|---|
| 基线 `python -m pytest -q`（修复前） | 446 通过、1 取消选择，305 秒 |
| 相关模块 7 个（修复后） | 69 通过 |
| 全量 `python -m pytest -q`（修复后） | 463 通过、1 取消选择，278 秒 |
| `bash -n scripts/run_streaming_probe_sweep.sh` | 通过 |
| checkpoint 张量有限性扫描（CPU，只读） | 见实验清单；原始记录 `artifacts/release_audit_20261015/ckpt_finite_sweep.jsonl` |
| LingBot C40 / MiniMax 全层战役本地产物核验 | `artifacts/release_audit_20261015/campaign_local_verify_20261015.json` |
| 调度器状态查询（267 个任务 id） | `artifacts/release_audit_20261015/campaign_job_status_20261015.json`，观测 2026-10-15 22:5x UTC |
| AOSS 上 MiniMax timestep 特征有限性（GPU 节点，1 卡 zoetrope） | 任务 `pt-et0ezc14`（首次 `pt-8p9mw0tc` 因命令引号在启动时失败）；记录 `artifacts/release_audit_20261015/feature_finiteness/submission_record.json` |
| AOSS 上 canonical MiniMax BF16_V2 缓存有限性（GPU 节点，1 卡 zoetrope，10-16） | 任务 `pt-ljbe3zu0`：42 个采样文件全部 bf16 且有限；结果 `artifacts/release_audit_20261015/feature_finiteness/bf16_v2_feature_finiteness_20261016.jsonl` |

环境：`~/.conda/envs/probe-spatial`（Python 3.11.15、PyTorch 2.6.0、
Lightning 2.6.1），CPU、`HF_HUB_OFFLINE=1`、`WANDB_MODE=disabled`。

## 七、未执行与限制

- 没有在干净环境安装、没有运行 GitHub Actions、没有真实多 GPU 训练。
- 大型 A1 checkpoint（约 4.2 GB）只抽样检查；全部检查需要在计算节点进行。
- 登录节点不能访问 AOSS，timestep 特征是否含 NaN 由 GPU 任务回答；在其
  结果出来之前，"MiniMax L20/L25 特征损坏"仍是最可能但未直接证实的解释。
- 没有重新评估任何 checkpoint；v2 评估器的数值需要在重新运行后以新版本号
  单独记录。
