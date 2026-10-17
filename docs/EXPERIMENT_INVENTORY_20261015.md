# 实验清单与发布剩余工作 — 2026-10-15（10-16 修正）

**10-16 修正**：10-15 版把 canonical V2 中 MiniMax-H3 第 20 层及以上的 128 个 run 标为
"特征缓存溢出阻塞"，依据是登录节点本地的 fp16 `FEAT_STREAMING/minimax_h3` 文件。
这些正式运行实际读取的是 bf16 命名空间 `FEAT_STREAMING/FEAT_TARGET_CANONICAL_V2_MINIMAX_BF16_V2`
（见各 run 的 `run_manifest.json` 与 `START_RECEIPT.json`）；GPU 任务 `pt-ljbe3zu0`
采样该命名空间 42 个文件，全部 bf16 且有限（最大绝对值达 3.7×10^6）。阻塞标记已撤回，
MiniMax L30 A/B16 恢复 VERIFIED，canonical VERIFIED 计数从 676 回到 **692**。
fp16 溢出只影响 timestep 的 `*_SEEDED_V1` 命名空间（L20/L25，含 target 缓存），该结论不变。
重提/重训方案见 [MINIMAX_REEXTRACT_RETRAIN_PLAN_20261016.md](MINIMAX_REEXTRACT_RETRAIN_PLAN_20261016.md)，
重训队列见 `metadata/release_inventory/retrain_queue_20261016.csv`（176 个 run）。

本清单在 [2026-10-11 清单](RELEASE_READINESS_20261011.md) 的分组与定义上更新，
依据为 2026-10-15 的本地产物核验、一次调度器快照（22:5x UTC）、首次 checkpoint
权重张量扫描和一次只读 GPU 特征审计。机器可读表：
[canonical_v2.csv](../metadata/release_inventory/canonical_v2.csv)、
[timestep.csv](../metadata/release_inventory/timestep.csv)、
[analyses.csv](../metadata/release_inventory/analyses.csv)（未变）；10-11 的原表保存在
`metadata/release_inventory/archive_20261011/`。重建命令：

```bash
python scripts/update_release_inventory_20261015.py
python scripts/summarize_release_inventory.py
```

术语：**run** 指一次训练配置的产物；**checkpoint** 指保存的权重文件；
**非有限** 指 NaN 或 ±inf；**VERIFIED** 在 canonical V2 中只表示训练身份与输入
协议门禁通过，不表示指标已验收；**YES_WITH_SCOPE** 表示在声明的
validation/development 范围内可作为结果候选。`run`、`package`、`scope` 是不同
统计单位，不相加。

## 结论

| 分组 | 本次统计对象 | 训练结束 | 权重张量检查 | 身份与协议核验 | 可发布结果 |
|---|---|---:|---|---:|---:|
| Canonical V2 | 868 个 run（原矩阵 736 + MiniMax 全层战役 132） | 865 | 已加载 588 个（含 20 个大型抽样）：572 有限、16 非有限；280 个大型 A1 未检查 | 692 VERIFIED；16 DEGENERATE；160 NOT_VERIFIED | 0 |
| Timestep | 576 个 run | 576 | 已加载 438（432 小型 + 6 大型抽样）；127 个非有限；138 个大型未检查 | 160 DEGENERATE；416 NOT_VERIFIED | 0 |
| 其他 analysis | 7 个 package、617 个 run、6 个未完成 scope | 不变 | 未做（各包有自己的冻结 receipt/哈希） | 不变 | 6 个 package YES_WITH_SCOPE |

两个主要新事实：（1）**B2（ego_belief_v2）在 CogVideoX、Aether、MiniMax-H3 上存在
整批 NaN 权重的"已完成"训练**（fp16 特征直通下训练；fp32 直通或 bf16 缓存的 B2 有限）；
（2）**timestep 的 MiniMax-H3 fp16 特征缓存（`*_SEEDED_V1`）在 L20/L25 溢出**，该
128 个训练全部无效；canonical 的 MiniMax 运行读 bf16 缓存，不受影响（10-16 修正）。
这两类都不能靠重新评估修复。

## 一、Canonical V2

### 完成与核验

| 范围 | run | 训练结束 | 身份+协议 VERIFIED | 说明 |
|---|---:|---:|---:|---|
| main552（8 模型） | 552 | 552 | 536 | LingBot C40 于 10-11/10-12 完成（40/40，权重有限，身份 MATCH）；CogVideoX B2 L0/5/10 与 Aether B2 L1 共 16 个 checkpoint 99.8% 参数非有限，改为 DEGENERATE |
| Wan168 | 168 | 168 | 140 | 28 个 A2 的逐例 3-step 门禁与代表性放行冲突，维持 NOT_VERIFIED |
| MiniMax L30 A/B16 | 16 | 16 | 16 | 权重有限、身份匹配、canary PASS；输入为 bf16 `MINIMAX_BF16_V2` 缓存（采样有限） |
| MiniMax 全层战役（六探针） | 96 | 95 | 0 | L10 16、L20 24、L30 8（C）、L40 24、L50 24；1 个 `l50 p24 streaming_mask` 仍在 H800 运行；无逐例 canary 记录，故 NOT_VERIFIED；输入同为 bf16 缓存 |
| MiniMax 全层战役（非六探针） | 36 | 34 | 0 | `streaming_camera` 20、`counterfactual` 16；2 个仍在运行；不计入主表 |
| 合计 | 868 | 865 | 692 | |

调度器快照（2026-10-15 22:5x UTC）：LingBot 40 个恢复任务 SUCCEEDED；MiniMax
战役 132 个有效任务中 129 个 SUCCEEDED、3 个 RUNNING（均在 H800），与本地
TRAIN_COMPLETE.json、正常结束日志和 checkpoint 一致。

### 权重张量扫描（首次）

扫描对象是两张清单列出的 1,272 个 checkpoint：全部 880 个小于 600 MB 的
ProbeExtension 权重（A2/B1/B2/C1/C2/C3）与 26 个按模型×探针抽样的大型 A1
权重（depth/mask，约 4.2 GB）。结果：

- canonical V2：16 个非有限，全部是 B2——CogVideoX L0/L5/L10 各 4 个 prefix、
  Aether L1 的 4 个 prefix；CogVideoX B2 L30/35/40（9-28 以 `PROBE_FP16_PASSTHROUGH=0`
  重跑的 12 个）有限；MiniMax L30 B2 有限；Wan、VideoMAE、V-JEPA2、DINO、F3R、
  Qwen、LingBot 的 B2 全部有限。20 个抽样大型 A1 权重全部有限。
- 这 16 个 run 在 10-11 清单中记为 COMPLETE/VERIFIED；现改为
  `correctness_status=DEGENERATE, result_status=INVALID_NONFINITE_WEIGHTS`。

### MiniMax-H3 特征缓存：fp16 旧缓存溢出，canonical 使用的 bf16 缓存有限

登录节点本地 fp16 缓存 `FEAT_STREAMING/minimax_h3/.../48da625e8c/window_0003/prefix_000007/`：
L10 全部有限（最大绝对值 20,480）；L20 每文件 59 个非有限值、L30 13 个、L40 1,655 个、
L49 55,527 个（占 0.12%），有限值的最大绝对值 65,280（fp16 上限 65,504）。这些是
9-23 之前的旧命名空间；canonical 正式运行没有读它们。正式运行读取的
`FEAT_STREAMING/FEAT_TARGET_CANONICAL_V2_MINIMAX_BF16_V2`（存储契约
`MINIMAX_H3_FINITE_BF16_V1`，提取时逐张量做有限性校验）采样 42 个文件全部 bf16
且有限，最大绝对值 L20 4.8×10^5、L30 1.2×10^6、L40 3.7×10^6、L49 3.6×10^5。
因此 canonical 的 148 个 MiniMax run 记 `BF16_V2_CACHE_SAMPLE_FINITE_42_FILES`，
不再阻塞；LingBot 同场景 5 层缓存全部有限（最大约 40）。
注意：该 bf16 存储策略只存在于冻结副本 `runtime/main_v2_minimax_bf16_v3_extractor/source`，
主目录的 `features/run_inscene15k.py` 仍一律写 fp16，需要合并（见计划文档）。

### 仍需工作

- 28 个 Wan A2：定验收口径（逐例 3-step 还是代表性 one-batch），补证据。
- 全部 692 个 VERIFIED run：从现有 checkpoint 导出身份匹配的正式指标（使用带
  `evaluator_version` 的新评估器，非有限预测会被计数而不是记零），冻结
  validation 聚合。
- MiniMax 战役 L10 的 20 个 run：补逐例协议 canary 或说明以 plan 中的远端特征
  证据替代。

## 二、Timestep

| 模型 | 完训 | 非有限权重（已检查小型 ckpt） | 特征溢出但权重有限/未检查 | 官方评估候选（未验收） | 仅训练日志 | 旧独训 C3 |
|---|---:|---:|---:|---:|---:|---:|
| Wan | 192 | 0 | 0 | 72 | 96 | 24 |
| CogVideoX | 192 | 24（全部 B2） | 0 | 48（A1、A2） | 96 | 24 |
| MiniMax-H3 | 192 | 103（B2 24 + L20/L25 的 A2/B1/C1/C2/C3 79） | 33（L20/L25 的 A1 depth/mask 32 + B1 1） | 16（L15 A1、A2） | 32 | 8 |
| 合计 | 576 | 127 | 33 | 136 | 224 | 56 |

- GPU 任务 `pt-et0ezc14`（1 卡 zoetrope，2 分钟，只读）检查 AOSS 上 3 个验证
  场景 × 2 个 prefix：MiniMax L15（t500/t1000）12 个文件全部有限；**L20/L25
  24 个文件全部含 inf**（每文件 8–60 个，最大有限值 65,280）；CogVideoX L15/L25
  与 Wan L25 正常。这解释了 10-11 记录的 16 个零损失 depth、16 个 NaN A2 和
  63 个非有限训练指标。
- 10-11 的"80 个退化官方评估行"现分为两类：CogVideoX/MiniMax 的 48 个 B2 行是
  NaN 权重（argmax 返回 bin 0 造成跨模型完全相同的角误差）；MiniMax L20/L25
  的 32 行是输入溢出。它们都是**无效**结果，不是"接近随机"的负结果。
- 其余 136 个候选（Wan 72、CogVideoX A1/A2 48、MiniMax L15 A1/A2 16）仍需源版本
  恢复、有效样本契约与导出；72 个 depth 摘要已含 AUC_30。
- 没有 timestep 行可发布；也没有理由重训 Wan 的 192 个 run。

## 三、其他 analysis

与 10-11 相同：六个结果包（Independent P8、continuous-action relational V2、
SE3 final r3、probe-validity paper、D-series Wan D1/D2、D-series V-JEPA2 D1/D3）
按 validation/development 范围可发布；V-JEPA2 D2 缺三种子汇总与统一导出；
C3-from-C1 V1、B2 P16 四模型扩展、Active representation MVP、SeeSE3 replication
缺完整结果。本次未重新核验这些包（它们有各自的冻结哈希审计），也未发现与之
矛盾的新证据。

## 四、发布剩余工作（按性质分类）

**A. 只需整理或导出**

1. 发布提交：显式 `git add` 未跟踪的正式源码与测试（`vidfm3d/eval_a1_streaming.py`、
   `vidfm3d/utils/aoss_fs.py`、`vidfm3d/utils/action_validity.py`、
   `vidfm3d/utils/owner_catalog.py`、`vidfm3d/utils/paper_validity.py`、
   `vidfm3d/data/components/c1_action_set_dataset.py`、
   `vidfm3d/models/components/movement_head.py`、`features/noise_seed.py`、
   `features/offline_hf.py`、`features/canonical_gap_worklist.py`、`tests/`、
   `protocols/`、`metadata/`、`docs/`）；排除 `spatial_paired_*`、
   `spatial_six_tasks_*`、`sites/`、`paired_*`、`analysis/interiorgs_v1_flow_audit/`。
2. 把个人绝对路径改为环境变量/示例（`features/run_inscene15k.py` VFM_DEFAULTS、
   `features/minimax_h3/*.py`、`features/lingbot/*.py`、评估器文档字符串、3 个
   测试常量）。
3. 六个 analysis 包：按 10-11 清单的范围声明整理下载索引与哈希。
4. 72 个 timestep depth 摘要中已有的 AUC_30 直接导出（Wan 24、CogVideoX 24、
   MiniMax L15 8，其余 16 行无效）。

**B. 需要补评估**

5. 692 个 VERIFIED canonical run 的正式指标导出与验收（新评估器，`val`）。
6. timestep 136 个候选的源版本恢复与验收；mask/B1/C1/C2 的 matched 评估
   （仅 Wan 与 CogVideoX，以及 MiniMax L15）。
7. 当前 C3：在 matched C1 checkpoint 上做 evaluation-only（Wan/CogVideoX 各 24，
   MiniMax L15 4）。
8. V-JEPA2 D2 三种子聚合与统一 D-series 导出。

**C. 需要先诊断**

9. B2 NaN 的原因：fp16 特征直通（`PROBE_FP16_PASSTHROUGH=1`）加 bf16-mixed 训练；
   CogVideoX L30/35/40 改为 fp32 直通后有限，读 bf16 缓存的 MiniMax canonical B2
   也有限。重训设置定为 `PROBE_FP16_PASSTHROUGH=0`；计划中先跑 1 个对照 cell 确认。
10. 28 个 Wan A2 的验收口径。

**D. 已有证据表明确实需要重训**（队列：`metadata/release_inventory/retrain_queue_20261016.csv`，176 个 run）

11. canonical B2：CogVideoX L0/5/10 与 Aether L1，共 16 个 run（权重 NaN；特征缓存
    有限，只改直通精度）。
12. timestep B2：CogVideoX 24 + MiniMax L15 8，共 32 个 run（同上）。
13. timestep MiniMax L20/L25 全部 128 个 run（2 timestep × 2 层 × 8 探针 × 4 prefix）：
    先把 `*_SEEDED_V1` 的 L20/L25 streaming 与 target 缓存按 BF16 路线重提到新命名
    空间，再重训。canonical MiniMax 不在此列（10-16 修正）。

**E. 不纳入本次发布的可选扩展**

14. C3-from-C1 V1 的 40 个 cell、B2 P16 四模型扩展、Active representation MVP、
    SeeSE3 replication、额外 seed、独立 test split。

## 五、证据与时间

| 证据 | 路径 | 观测时间 |
|---|---|---|
| 调度器快照（267 个任务） | `artifacts/release_audit_20261015/campaign_job_status_20261015.json` | 2026-10-15 22:5x UTC |
| 战役本地产物核验 | `artifacts/release_audit_20261015/campaign_local_verify_20261015.json` | 2026-10-15 22:5x UTC |
| checkpoint 张量扫描 | `artifacts/release_audit_20261015/ckpt_finite_sweep.jsonl` | 2026-10-15 22:4x–23:0x UTC |
| AOSS timestep 特征有限性（GPU 任务 pt-et0ezc14） | `artifacts/release_audit_20261015/feature_finiteness/aoss_feature_finiteness_20261015.jsonl` | 2026-10-15 23:00–23:02 UTC |
| AOSS canonical BF16_V2 缓存有限性（GPU 任务 pt-ljbe3zu0） | `artifacts/release_audit_20261015/feature_finiteness/bf16_v2_feature_finiteness_20261016.jsonl` | 2026-10-16 05:52–05:5x UTC |
| 本地 MiniMax/LingBot t250 缓存检查 | `artifacts/release_audit_20261015/local_t250_*_feature_finiteness*.log` | 2026-10-15 23:0x UTC |
| 10-11 原表 | `metadata/release_inventory/archive_20261011/` | 2026-10-11 |

本清单没有重训、重评估或改写任何历史指标文件；所有新状态字段都带 10-15 标注，
10-11 的判断仍可在归档表中查到。
