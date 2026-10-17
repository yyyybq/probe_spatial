# 代码版本对齐：结果批次、冻结包与主目录 — 2026-10-17

范围：`/mnt/umm/users/yinbaiqiao/probe_spatial`，2026-10-17 在 fujian 上只读核对。
证据目录：`artifacts/release_audit_20261017/code_alignment/`（逐文件 diff、`pack_commits.tsv`）。
所有 commit 与 tag 只在本地，未 push。

## 结论

- **canonical V2 与 2026-09 timestep 的结果都不是主目录代码训练的。** 它们来自 `runtime/` 下的冻结包，
  这个目录被 `.gitignore` 排除，此前也不在 git 历史里。运行清单里的 git commit 一律记为 "unknown"。
- **现在每个产出结果的冻结包都对应一个本地 commit 和 tag `code/<包名>`**，见下表。其中 5 个包的
  身份能用当时记录的哈希逐一核对；2 个 timestep 训练包当时没有记录完整身份，只能作为 10-17 的快照。
- **主目录与冻结包是两条并行的开发线。** 5 个包与主目录都在同样 15 个文件上不同。主目录多了后来的修复与
  新功能；冻结包里有主目录没有的实现，其中特征选择用的 owner overlay 和标签缓存会影响能否按原样复现。
  所以主目录目前不能原样重跑或评估 canonical 结果。
- **评估必须用训练时的冻结包代码。** 加载 checkpoint 的模型代码在两边兼容；但数据加载路径不同，
  要得到与训练一致的样本与标签，评估应在冻结包上运行，只替换为 v2 评估器。

## 一、结果批次 → 冻结包 → commit

| 结果批次 | 行数（清单） | 冻结包 | commit | 身份核对 |
|---|---:|---|---|---|
| canonical V2 主矩阵、Wan、Qwen/F3R、DINO、VideoMAE/V-JEPA2、CogVideoX、LingBot 及各次重试 | 696 | `main_v2_probe_v2_r7_single_v3` | `0cfc903` | `source.sha256` 296 个文件全部一致 |
| canonical V2 Aether L1 | 24 | `main_v2_probe_v2_aetherfix_r1` | `06600ee` | `source.sha256` 297 个文件全部一致 |
| canonical MiniMax 全层战役 132 + L30 A/B16 | 148 | `main_v2_probe_minimax_bf16_v2` | `ad63dad` | `source.sha256` 288 个文件全部一致；包目录里另有 28 个冻结后写入的 canary 输出，不计入 |
| canonical MiniMax 的 bf16 特征（`*_MINIMAX_BF16_V2`） | — | `main_v2_minimax_bf16_v3_extractor` | `6dea88e` | 树摘要与 8 条提交命令记录的 `7183cd9b…` 一致 |
| timestep `tstep_probe_opt_v3` 三个 run id | 430 | `tstep_probe_opt_v3` | `756334e` | **未核对**：当时没有记录身份，10-17 快照 |
| timestep `tstep_probe_pack_v4_1` 四个 run id | 58 | `tstep_probe_pack_v4_1` | `eebc426` | **部分核对**：部署清单记录了 3 个文件的哈希，全部一致 |
| timestep `tstep_probe_seeded_v1/…20260907` | 88 | 无冻结包：2026-09-07 直接从主目录运行 | 无 | **无法恢复**：主目录此后已改动且从未提交 |
| 2026-10-16 timestep B2 重训（32） | — | 主目录 | `e4e4886` | 10-16 07:04 UTC 之后 `vidfm3d/`、`configs/`、`protocols/` 没有文件改动 |
| 2026-10-16 canonical B2 重训（16） | — | `r7_single_v3` / `aetherfix_r1` | `0cfc903` / `06600ee` | 运行器按 sha 绑定这两个包 |
| 2026-10-16 MiniMax bf16 timestep 重提 | — | `tstep_minimax_bf16_v1_extractor_r2` | `212e890`（另见 `41cce9c`） | `source.sha256` 13 个文件全部一致 |

另有 1 行 canonical（特征恢复范围）和 1 行 timestep（协议无效）没有对应训练。

未覆盖：其余模型的 canonical 特征缓存由哪些提特征包产生，本次没有映射。

## 二、冻结包与主目录的差异

5 个训练包与主目录的不同文件相同，下面以 `r7_single_v3` 为例，括号内为“仅主目录 / 仅冻结包”的行数。
逐文件 diff 在 `artifacts/release_audit_20261017/code_alignment/diffs/<包名>/`。

**主目录新增，不改变已训练模型的训练语义**

- `vidfm3d/models/components/probe_path_integration.py`（+82/−23）：重构为 initialize/update/decode 三步，
  参数与计算相同，旧 checkpoint 可直接加载。
- `vidfm3d/models/video_probe_module.py`（+86/−50）：可选的 movement 头、非有限输入检查、验证损失按批大小加权、
  test 阶段的指标名改为 `test/`。
- `vidfm3d/models/probe_ext_module.py`（+98/−98）：非有限输入检查、空批次指标结构统一、`compile=True` 被拒绝；
  `overlap_n_used` 指标的定义改变（只影响记录的指标）。
- `vidfm3d/models/components/probe_pixalign.py`（+13）、`vidfm3d/data/video_probe_datamodule.py`（+1）：
  movement 头与 C1 动作集数据集的接入。
- `vidfm3d/train.py`（+19/−21）：`seed=0` 现在也会设种子（原代码会跳过）；test 阶段改用最佳 checkpoint；
  `torch.load` 的兼容处理改为显式参数。canonical 都用 seed 42，不受 seed 修复影响。
- `vidfm3d/utils/loss.py`（+259）：`require_finite_input` 与 movement 损失。
- `vidfm3d/eval_diag.py`（+519/−34）、`vidfm3d/eval_a1_streaming.py`（+151/−21）：v2 评估器，
  把非有限预测计数为无效，不再按 0 或类别 0 记分。
- `vidfm3d/utils/feature_layers.py`（+51/−4）：登记 MiniMax-H3 与 LingBot。
- `configs/model/probe_ext.yaml`、`vidfm3d/utils/vis_utils.py`：注释与延迟导入。

**冻结包有、主目录没有，影响复现**

- **owner overlay**（`vidfm3d/utils/probe_owner_overlay.py`，环境变量 `PROBE_OWNER_OVERLAY_MANIFEST`、
  `PROBE_OWNER_LOCAL_MIRROR_ROOT`）：canonical 训练通过它为每个样本选定特征文件；运行器把特征根设为
  `aoss://baiqiao/PROBE816_OWNER_OVERLAY_ONLY`，即全部特征都经由 overlay。主目录删除了它，换成
  owner catalog（`INSCENE_OWNER_CATALOG`）。目前没有任何 owner catalog 文件，两者是否等价未验证。
- **标签缓存与元数据缓存**（`INSCENE_LABEL_CACHE_ROOT`、`INSCENE_METADATA_CACHE_MAX_SCENES`、
  `INSCENE_PROBE_TASK`），以及只加载图像或深度的快速路径、A2 overlap 的专用取样路径
  `_getitem_overlap_diag`。主数据加载器里这些都不存在（相对 `r7_single_v3`：+226/−543 行）。
  2026-09-10 的“主目录实现 vs 优化包”短训练 canary 中，Wan t999 L15 p8 的 B2、depth、mask 三例参数不一致，
  等价性没有逐位证实。
- `probe_ext_module.py` 的 `transfer_batch_to_device`：只把当前探针需要的张量拷到 GPU，影响速度，不影响结果。

**行为不同，需要知道**

- `vidfm3d/utils/aoss_fs.py`：冻结包把远程查询失败当作“文件不存在”，主目录改为报错。核对了 692 个已核验
  canonical run 中有记录的 668 个 `aoss_metrics.jsonl`，查询错误事件为 0；另 24 个没有该文件。
  冻结包还把 AOSS 配置默认指向个人路径，主目录改为 `~/.petreloss.conf`，SCO 作业需显式设 `AOSS_CONF`。

## 三、对发布与评估的含义

1. **评估**：在每个训练包上评估它自己训练的 checkpoint，只把 `eval_diag.py` 与 `eval_a1_streaming.py`
   换成主目录的 v2 版本，做法同冻结包运行器（按 sha 绑定、先 canary）。这样样本选择与训练一致，
   评分又避开了 v1 评估器的非有限值问题。
2. **发布代码**，需要你决定：
   (a) 推荐：以主目录为发布代码，补回 owner overlay（或证明 owner catalog 等价）与标签生成路径，
   用 canary 证明在相同 checkpoint 上与冻结包给出相同预测；结果表同时记录训练用的 `code/<包名>` commit；
   (b) 直接发布冻结包作为“结果代码”，主目录作为后续开发版本，需要说明两者差异；
   (c) 只发布主目录，不保证与已发表数字逐位一致。
3. **timestep**：2026-09-07 的 88 个 run 没有可恢复的代码版本；`opt_v3` 与 `pack_v4_1` 只有快照。
   这些 run 若要进入发布结果，需要在说明里标注，或用已提交的代码重训。
