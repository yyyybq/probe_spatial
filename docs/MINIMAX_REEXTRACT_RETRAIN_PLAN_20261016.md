# MiniMax-H3 timestep 特征重提与 176 个 run 重训：提交清单与资源估算 — 2026-10-16

> **执行状态（2026-10-16 14:10 UTC）**：阶段 0 完成；阶段 4 的 48 个 B2 重训运行中（首个已完成，权重有限）；阶段 1 canary 第 4 次通过；阶段 2/3 的 32 个分片已提交。见 [执行记录](MINIMAX_REEXTRACT_RETRAIN_EXECUTION_20261016.md)。

这是方案，不是执行记录：本文件撰写时**没有提交任何提取或训练任务**。
两次只读 GPU 审计（`pt-et0ezc14`、`pt-ljbe3zu0`）与一次登录节点 CPU 扫描是估算依据。

术语：**run** 指一个 (模型, 探针, 层, prefix, timestep, seed) 训练配置；**streaming 缓存**
指每个观察窗口前缀 `[I_0..I_t]` 一次前向得到的特征文件；**target 缓存** 指 C 类探针用的
单帧孤立目标特征；**bf16** 与 **fp16** 是两种 16 位浮点存储格式，bf16 的可表示范围
与 fp32 相同，fp16 最大 65,504。

## 结论与范围修正

| 项 | 10-15 版 | 10-16 修正后 |
|---|---|---|
| 需重提特征的范围 | timestep L20/L25 + canonical L20/30/40/49 | **仅 timestep**：`FEAT_STREAMING/FEAT_TARGET_CANONICAL_V2_SEEDED_V1` 的 MiniMax-H3 L20/L25，t500 与 t1000 |
| 需重训的 run | 256 | **176**：timestep MiniMax L20/L25 全部 128 个 + B2 权重 NaN 的 48 个 |
| canonical MiniMax L≥20（128 个） | 阻塞 | 不阻塞：它们读的是 bf16 `MINIMAX_BF16_V2` 缓存，采样 42 个文件全部有限 |

修正依据：正式 canonical run 的 `run_manifest.json`/`START_RECEIPT.json` 记录输入为
`aoss://baiqiao/InsScene-15K/FEAT_{STREAMING,TARGET}_CANONICAL_V2_MINIMAX_BF16_V2`；
GPU 任务 `pt-ljbe3zu0` 采样其 L10/20/30/40/49 共 42 个文件：全部 `torch.bfloat16`、
全部有限、最大绝对值 L20 4.8×10^5、L30 1.2×10^6、L40 3.7×10^6（远超 fp16 范围，
所以 bf16 存储是必要的）。timestep 的 `*_SEEDED_V1` 仍是 fp16：`pt-et0ezc14` 采样
L20/L25 的 24 个 streaming 文件全部含 inf，12 个 target 文件中 10 个含 inf。

## 一、需要重新提取的特征

| 字段 | 值 |
|---|---|
| 模型 | `minimax_h3`，`/mnt/umm/users/yinbaiqiao/Consistency/minimax-h3/models/MiniMax-H3`，输入 73 帧 480×832，prompt 与 seeded-v1 相同 |
| 层（逻辑 = 磁盘） | 20、25 |
| scheduler timestep | 500、1000 |
| 协议 | canonical-v2 seeded-v1（确定性噪声种子 `inscene_noise_seed_v1`，窗口契约不变），场景清单 `artifacts/tstep_canonical_v2/manifests/inscene_c_h124_prefix{8,12,16,24}.jsonl`（每份 2,721 行：train 2,445 / val 276，681 个场景） |
| 新命名空间（建议） | `FEAT_STREAMING_CANONICAL_V2_SEEDED_V1_MINIMAX_BF16_V1`、`FEAT_TARGET_CANONICAL_V2_SEEDED_V1_MINIMAX_BF16_V1`；存储契约沿用 `MINIMAX_H3_FINITE_BF16_V1`（逐张量有限性校验，bf16 落盘） |
| streaming 单位 | 每 timestep 10,884 个 prefix 前向（681 场景 × 4 窗口 × 4 prefix）；文件数 = 10,884 × 2 层 × 2 timestep = **43,536**；每文件 `(22,15,26,5376)` bf16 = 92.3 MB → **4.0 TB** |
| target 单位 | 每 timestep 681 场景 × 56 个目标帧（4 窗口 × 4 prefix × 3 horizons + 锚帧去重，与现有 C manifests 一致）；文件数 = 681 × 2 层 × 2 timestep = **2,724**；每文件约 235 MB → **0.64 TB** |
| 合计存储 | 约 **4.7 TB**（AOSS 对象存储；本地只需每分片的暂存空间） |

现有 fp16 的 t1000 target 缓存只覆盖 237 个场景（`logs/tstep_resume4/aoss_done/FEAT_TARGET_minimax_h3_t1000.txt` 711 行 = 237 × 3 层），即原来 t1000 的 C 类训练本身就缺目标；重提须覆盖全部 681 个场景。

### 吞吐依据（单卡 `N4lS.Iq.I80.1`，MiniMax-H3 bf16 路线的历史日志）

| 模式 | 来源 | 实测 |
|---|---|---|
| streaming | `minimax_layer30_priority_20260924_r1/streaming-0-of-8`（1 层 hook）：1,360 单位 = 85 场景，10:42→20:47 | 10.1 h / 85 场景 ≈ **7.1 min/场景**（16 个 prefix） |
| streaming | `minimax_bf16_v2_production_extract_v4/shard_*`（5 层 hook）105 个场景 | 均值 **552 s/场景**，最大 859 s |
| target | `minimax_layer30_priority_20260924_r1/target-0-of-8`：每场景 56 帧 | **1,610 s/场景**；85 场景 37.1 h（含上传） |

hook 的层数不改变前向成本，所以 L20+L25 一次前向即可。

### 提特征资源估算

| 阶段 | 计算量 | 作业形态 | 墙钟 |
|---|---:|---|---|
| streaming t500 + t1000 | 2 × 681 × ~500 s ≈ **190 GPU·h** | 2 timestep × 8 分片 × 1 卡 = 16 个作业 | 每作业约 12 h |
| target t500 + t1000 | 2 × 681 × 1,610 s ≈ **610 GPU·h** | 2 × 8 分片 × 1 卡 = 16 个作业（或 32 × 1 卡缩短墙钟） | 每作业约 38 h（16 分片时约 19 h） |
| 合计 | **约 800 GPU·h**（不确定性 ±30%：场景帧数差异、AOSS 上传、队列等待） | 32 个单卡作业 | 全部并行约 2 天；16 卡并行约 4 天 |

## 二、重训队列（176 个 run）

机器可读：`metadata/release_inventory/retrain_queue_20261016.csv`
（`python scripts/build_retrain_queue_20261016.py` 重建）。单 run 时长取已完成 timestep
同模型同探针的单卡中位数/第 90 百分位（`run_manifest.json`→`TRAIN_COMPLETE` 时间）。

| 队列组 | run 数 | 组成 | 前置条件 | 训练设置变化 | GPU·h 中位 / p90 |
|---|---:|---|---|---|---:|
| `B2_NONFINITE_WEIGHTS_FP16_PASSTHROUGH` | 48 | canonical：CogVideoX L0/5/10 × 4 prefix（12）、Aether L1 × 4（4）；timestep：CogVideoX L15/20/25 × t499/t999 × 4（24）、MiniMax L15 × t500/t1000 × 4（8） | 无（现有缓存有限） | `PROBE_FP16_PASSTHROUGH=0`，其余 override 完全相同 | 389 / 712 |
| `TSTEP_MINIMAX_L20_L25_FEATURE_OVERFLOW` | 128 | MiniMax L20/L25 × t500/t1000 × 8 探针 × 4 prefix（depth、mask、A2、B1、B2、C1、C2、旧独训 C3 各 16） | 第一节的重提完成并通过对齐审计 | 指向新 bf16 命名空间；`PROBE_FP16_PASSTHROUGH=0`；seed 42、manifest、预算不变 | 795 / 1,410 |
| 合计 | **176** | | | | **1,184 / 2,122** |

按探针的 128 个 timestep run 的单卡时长（MiniMax 中位 / p90）：depth 13.7 / 19.8 h、
mask 17.5 / 35.6 h、A2 1.0 / 1.5 h、B1 6.9 / 12.8 h、B2 7.0 / 13.1 h、C1 1.1 / 1.7 h、
C2 1.3 / 1.8 h、C3 1.2 / 1.8 h。B2 的 48 个 run 用 CogVideoX 中位 8.3 h（p90 15.2 h）。

训练前每个 cell 的 3 步协议 canary（现有 `probe_validation` 流程）约 5 min/cell，
176 个 cell 约 15 GPU·h。

## 三、提交清单（按阶段，含依赖）

**阶段 0：代码与冻结（CPU，约半天）**

1. 把 bf16 存储契约从冻结副本合并到主目录 `features/run_inscene15k.py`：
   `validate_finite_feature`、`prepare_feature_for_storage`、`MINIMAX_STORAGE_CONTRACT`、
   发布前 dtype/有限性校验、sidecar 字段 `feature_storage_contract/feature_storage_dtype/finite_check`，
   并在 `SEEDED_FEATURE_NAMESPACES` 中登记两个新命名空间（差异约 60 行，来源
   `runtime/main_v2_minimax_bf16_v3_extractor/source/features/run_inscene15k.py`）。
   补一个合成测试：MiniMax 张量含 inf 时拒绝落盘；正常张量以 bf16 落盘且 sidecar 记录 dtype。
2. 用 `scripts/build_minimax_bf16_production_route.py` 的方式生成新的路线回执：场景列表取
   seeded-v1 四份 manifest 的 681 个场景；`streaming_layers=[20,25]`、`target_layers=[20,25]`；
   每个 timestep 一份回执（t500、t1000）；记录源码树 sha256。
3. 冻结新的 runtime 副本（`runtime/tstep_minimax_bf16_v1_extractor/`），复制并改写
   `run_production_shard_v3.sh`：增加 `TIMESTEP`、`FEATURE_NAMESPACE`、`EXTRACT_MODE`
   环境变量；保留 launcher/verifier/route 三重 sha 校验与发布后 `verify_production_shard_v3.py`。
4. 冻结 176 个 run 的 item 文件：B2 组直接复用各自原 campaign 的 item（`single_gpu_*`、
   `tstep_probe_*` 的配置 override），只加 `PROBE_FP16_PASSTHROUGH=0` 环境；timestep 组
   复用 `artifacts/tstep_canonical_v2/preextract_matrix/groups.tsv` 的分组方式，替换
   `streaming_namespace/target_namespace`。

**阶段 1：提特征 canary（GPU，1 卡，约 30 min）**

5. 1 个场景、两个 timestep、两种模式各跑一次 → 校验：文件 dtype 为 bf16、全部有限、
   形状 `(22,15,26,5376)`、sidecar 含存储契约与噪声种子字段、`prefix_index.npy` 与
   seeded-v1 完全一致、`scripts/audit_tstep_canonical_alignment.py` 对新命名空间通过。

**阶段 2：streaming 重提（GPU）**

6. 16 个作业：`t500 × shard 0–7`、`t1000 × shard 0–7`，`--aec2-name=zoetrope`
   `--worker-spec=N4lS.Iq.I80.1 --priority=HIGHEST --quota-type=reserved`，镜像与挂载同现有脚本；
   每作业约 12 h；产物发布到 AOSS 新命名空间并写 `COMPLETE.json`。

**阶段 3：target 重提（GPU，可与阶段 2 并行）**

7. 16 个作业（或 32 个以缩短墙钟）：同上，每作业约 38 h（16 分片）/ 19 h（32 分片）。

**阶段 4：B2 重训（GPU，不依赖阶段 1–3，可立即开始）**

8. 48 个 cell 的 3 步 canary（现有 `probe_validation` runner，1 卡作业打包）。
9. 48 个单卡训练作业（`run_single.py` 模式，`PROBE_FP16_PASSTHROUGH=0`），
   canonical 16 个输出到新 run id（如 `main_v2_b2_fp32pt_20261017_r1`），timestep 32 个
   输出到 `tstep_probe_b2_fp32pt_20261017_r1`；不覆盖旧目录。

**阶段 5：timestep MiniMax L20/L25 重训（GPU，依赖阶段 2、3 全部 PASS）**

10. 对齐审计 PASS 后，16 个组（2 timestep × 2 层 × 4 prefix）× 8 探针：按现有
    `scripts/run_tstep_probe_group.sh` 一组一个 8 卡节点（`N4lS.Iq.I80.8`，每卡一个探针），
    16 个作业，每作业以最慢的 mask 为界约 18–36 h；或改为 128 个单卡作业。
11. 可选：组内的 16 个旧独训 `counterfactual` 是协议外的遗留对照，可不重训（省约 20 GPU·h）。

**阶段 6（本方案不含）**：对新 cell 跑带 `evaluator_version` 的 A1/A2/B2 官方评估与
C1/C2/C3 匹配评估，再进入验收。

### 资源汇总

| 项 | GPU·h（中位） | GPU·h（p90 / 上界） | 作业数 | 墙钟（并行充足时） |
|---|---:|---:|---:|---|
| 提特征（含 canary） | ~800 | ~1,050 | 33 | 2–4 天 |
| B2 重训 + canary | ~400 | ~725 | 48 + canary | 1–2 天，可立即开始 |
| timestep 重训 + canary | ~810 | ~1,425 | 16（8 卡）或 128（1 卡） | 1.5–3 天，依赖提特征 |
| 合计 | **~2,000** | **~3,200** | | 串行依赖下约 4–6 天 |

不确定性来源：集群排队（10-12 曾有 95 个作业排队半天）、AOSS 上传、mask 探针尾部时长
（p90 是中位的 2 倍）。GPU·h 为单卡占用时间之和，8 卡节点按 8 倍折算节点时。

## 四、需要你决定的事项（推荐项放第一）

1. **target 范围**：(a) 与现有 C manifests 一致的每场景 56 帧（推荐，与 timestep C1/C2/C3
   协议对齐，约 610 GPU·h）；(b) 只提训练必需帧（需先改 manifest 并重新冻结协议，节省有限）。
2. **B2 对照 cell**：(a) 直接以 `PROBE_FP16_PASSTHROUGH=0` 重训 48 个（推荐；CogVideoX
   L30/35/40 与 bf16 缓存的 MiniMax B2 已证明该设置下权重有限）；(b) 先跑 1 个 cell 的
   fp16/fp32 直通对照（约 8 GPU·h，多等约 1 天）。
3. **顺序与资源池**：(a) 阶段 4 立即在 h800/zoetrope 混合提交，阶段 2/3 在 zoetrope
   并行（推荐）；(b) 全部串行以减少同时占卡。
4. **旧独训 C3 的 16 个 run**：(a) 一并重训保持 8 探针矩阵完整（推荐，约 20 GPU·h）；
   (b) 不重训，timestep 矩阵改为 7 探针并在清单注明。
5. **命名空间与 dtype**：(a) 本文件所列新命名空间 + bf16（推荐，与 canonical 一致）；
   (b) fp32 存储（存储翻倍到约 9.4 TB，无必要）。

## 五、明确不做

- canonical MiniMax（L30 A/B16 与全层战役 132 个）：不重提、不重训；只需正常的指标导出与验收。
- MiniMax L15、Wan、CogVideoX 的 A1/A2/C 类 timestep run：缓存有限、权重有限，不动。
- 不删除旧的 fp16 缓存与 NaN checkpoint；新结果写新目录与新命名空间。

## 六、可重复命令

```bash
python scripts/build_retrain_queue_20261016.py          # 重建 176 行队列
python scripts/update_release_inventory_20261015.py      # 读取 artifacts/release_audit_20261015/ 的证据
python scripts/summarize_release_inventory.py
```

证据：`artifacts/release_audit_20261015/feature_finiteness/`（两次 GPU 审计与提交记录）、
`artifacts/release_audit_20261015/ckpt_finite_sweep.jsonl`、
`artifacts/main_v2_probe_v2_816_20260917/minimax_layer30_priority_20260924_r1/`（吞吐日志）、
`artifacts/main_v2_probe_v2_816_20260917/minimax_bf16_v2_production_route_v2/route_receipt.json`。
