# Q00：历史 1096 个非 diffusion-step 配置逐行 reconciliation

生成日期：2026-08-28（UTC）  
模式：严格只读对账；本轮没有启动训练、feature extraction、eval 或提交任务，也没有修改 registry/canonical。

## 结论先行

1096 universe 可以唯一冻结为 `experiments/task_status_1096_before_expand_20260822.csv`（SHA256 `9c7d6807b701de4b052819971a5c011e29c717e1c3e958ec9ae23116cdcfe779`）。三份 1096 行机器表的 full identity 集合完全相等；当前 1260 行 ledger 的 164 个 `expand20260822_` 新增项已全部排除。1096 中没有 t499/t999/t500/t1000 或 diffusion-step sweep；历史默认 t749/t250 保留在逐行 `default_timestep` 字段中。

按目标 metric contract 计，已有数值结果 **833/1096**：642 个 canonical-valid parent configurations、31 个 legacy B2 angle summaries、以及 162 个 Wan/Cog provisional rows（其中 2 个 parent 已有后续 canonical，故并集为 833）。F3R B2 l24/p24 的旧 `top1` 实际为 n/a，不是辅助数值，因此“任何 exact finite numeric”同样是 **833/1096**；完全没有任何数值的是 **263** 个配置（262 个无 registry representation + 该 F3R 行）。canonical-valid 是 **642/1096**，不是 761；761 是 atomic canonical ID 数量。

当前可直接 packaging 的严格数量是 **0**。31 个 default B2 虽有目标数值，但 split manifest 未绑定、样本量不一致且协议未冻结，必须先过 Wave E 的 split/protocol audit；通过后才转 Wave A。严格 checkpoint + manifest 条件下可 eval-only 的是 **99** 个；83 个 checkpoint 中断/无效，35 个 VideoMAE p24 缺 feature，237 个以 manifest/provenance/config 为首要阻塞。

## 1. Universe 冻结、来源和版本比较

| 路径 | 行数 | SHA256 | git 状态 | 判定 |
|---|---|---|---|---|
| experiments/task_status_1096_before_expand_20260822.csv | 1096 | 9c7d6807b701de4b052819971a5c011e29c717e1c3e958ec9ae23116cdcfe779 | ?? experiments/task_status_1096_before_expand_20260822.csv | 冻结 source of truth |
| experiments/expected_experiments.csv | 1096 | 25c1feb114d3f9a92d17e08e961185950559bde6942e36f564ba656ae1c47992 | ?? experiments/expected_experiments.csv | 同一 full-identity 集；较少生命周期字段 |
| experiments/registry_reconstructed_candidate.csv | 1096 | e8227598b9a4e28f328f223538e044aa0b821466ea229d31e560bb31fa556c61 | ?? experiments/registry_reconstructed_candidate.csv | 同一 full-identity 集；reconstruction candidate |
| experiments/task_status_1096.csv | 1260 | 92b86c63c2fcc8062b36a6e44dd27a6acf5c02e5699ee591f9042d7d87bcb899 | ?? experiments/task_status_1096.csv | 含 164 个 post-freeze 扩展，不属于本轮 |

冻结 full identity 是 `(campaign, model, model_variant, layer, prefix, probe, horizon, seed, protocol)`；`universe_id` 直接使用源表确定性的 `experiment_id`。1096 个 ID 和 1096 个 full identities 均唯一。源表没有原生 protocol 列，因此本报告显式加 namespace `historical_candidate_v1_val_protocol`；它是历史协议标签，不等于自动科学可比。

选择 pre-expand task ledger 而不是 reconstruction candidate 的依据是 `docs/FILE_MAP.md` 将 task ledger 标为 authoritative、candidate 标为 non-authoritative；pre-expand 文件又保留 48 个生命周期/provenance 字段。当前 1260 ledger 中原 1096 行的 48 个字段逐项相同，只有 164 个 `expand20260822_` additions（default8 156、wan_cog 8；Cog 132、Wan 32）。

生成 provenance：`scripts/reconstruct_historical_experiments.py`（`9aafc1ebb6a0e07bf94d5ea81e3cdf67c79b2776e637e611fbba12438fb5de05`）生成 expected/reconstruction candidate；`scripts/build_task_status_ledger.py`（`d847603ea590a9b329506962345af654603f4cde14955c3c6b688e9ee5f18748`）生成生命周期 ledger；历史说明位于 `archive/consolidation_history/20260817/historical_reconstruction_report.md`（`2c8b1359fcfd04587fce4b7aee9653ef9fc44ba6f25f41270acc3281c44551f8`）。这些关键文件当前均为 untracked 工作树文件，不能仅以 git commit 声称可复现；后续每次执行必须复制入口到 run provenance 并冻结 SHA。

历史脚本不能直接重用其状态：ledger 的 canonical/eval lookup 只用 `(model, layer, prefix, probe)`，忽略 campaign/model_variant/protocol；本报告完全重做 full-identity crosswalk。

## 2. Registry 与 canonical 精确对账

- Registry 有 955 IDs；928 个精确映射到 frozen universe 的 834 个 parent rows，27 个是 post-freeze Wan/Cog IDs，不属于 1096。
- 因为 A1 depth/camera atomic children 和少量 variant/collision IDs，928 个 in-universe registry IDs 比 parent rows 多 94 个。
- 所以真正没有任何 registry representation 的是 **262 个配置**。`1096 - 955 = 141` 只是不同粒度相减；正确恒等式是 `262 - 94 - 27 = 141`，不存在可列出的“141 个 missing configurations”。
- 761 个 canonical IDs 中，734 个映射回 642 个 frozen parent rows（732 个 primary mappings + 2 个 same-parent alternate mappings）；27 个是同一批 post-freeze extras。642 个 parent 的 expected atomic metrics 均齐全。
- 194 个 registry INCOMPLETE IDs 全部映射回 universe：162 个 Wan/Cog provisional + 32 个 default B2。它们对应 192 个 only-incomplete parents；另 2 个 parent 同时有后续 exact canonical A1 depth/camera IDs。

忽略 registry `model_variant` 后有 **21 collision groups / 42 COMPLETE IDs**。本报告保留每个 ID、campaign、variant 和 provenance，0 个被标成 exact duplicate alias。完整列表在 `universe_summary.json.variant_collision_groups` 和逐行 CSV 的 collision 字段中。

其中两组 Wan collision 映射到同一 frozen parent，但没有合并 artifact：B1 Wan l29/p24 选择原 `wan_cog_layer_sweep` 的 56.21144057 为 primary，`eval_20260817` 的 57.74463957 单列 alternate；B2 Wan l0/p24 同理选择 80.49305772，77.51404363 单列 alternate。逐行 CSV 的 `primary_registry_id` / `alternate_registry_id`、metric/raw/checkpoint alternate 字段完全分离；科学可比性只针对 primary variant。

## 3. 每个 task family 的覆盖矩阵

| Family | Universe | Canonical-valid | 目标数值 | Noncanonical | Models | Layers | Prefixes | Campaigns | 状态分布 |
|---|---|---|---|---|---|---|---|---|---|
| A1 | 300 | 152 | 202 | 148 | aether:8, cogvideox:40, dino:8, f3r:8, lingbot:40, minimax_h3:60, qwen2_5_vl:8, videomae:48, vjepa2:48, wan:32 | l-1:8, l0:32, l1:8, l5:24, l10:28, l11:16, l15:8, l17:16, l20:36, l23:32, l24:8, l25:8, l29:4, l30:24, l35:4, l40:24, l48:8, l50:12 | p8:75, p12:75, p16:75, p24:75 | default8:64, lingbot_video_layer_benchmark:40, minimax_h3_layer_benchmark:60, videomae_vjepa_layer_sweep:80, wan_cog_layer_sweep:56 | CANONICAL_COMPLETE:152, CHECKPOINT_INVALID_OR_INTERRUPTED:50, CHECKPOINT_READY_EVAL_ONLY:13, FEATURE_PARTIAL_OR_MISSING:10, MANIFEST_OR_PROVENANCE_BLOCKED:75 |
| A2 | 168 | 82 | 138 | 86 | aether:4, cogvideox:36, dino:4, f3r:4, lingbot:20, minimax_h3:20, qwen2_5_vl:4, videomae:24, vjepa2:24, wan:28 | l-1:4, l0:20, l1:4, l5:16, l10:16, l11:8, l15:8, l17:8, l20:16, l23:16, l24:4, l25:8, l29:4, l30:12, l35:4, l40:12, l48:4, l50:4 | p8:42, p12:42, p16:42, p24:42 | default8:32, lingbot_video_layer_benchmark:20, minimax_h3_layer_benchmark:20, videomae_vjepa_layer_sweep:40, wan_cog_layer_sweep:56 | CANONICAL_COMPLETE:82, CHECKPOINT_INVALID_OR_INTERRUPTED:10, CHECKPOINT_READY_EVAL_ONLY:15, FEATURE_PARTIAL_OR_MISSING:5, MANIFEST_OR_PROVENANCE_BLOCKED:56 |
| B1 | 56 | 10 | 56 | 46 | cogvideox:32, wan:24 | l0:8, l5:8, l10:8, l15:8, l25:8, l29:4, l30:4, l35:4, l40:4 | p8:14, p12:14, p16:14, p24:14 | wan_cog_layer_sweep:56 | CANONICAL_COMPLETE:10, MANIFEST_OR_PROVENANCE_BLOCKED:46 |
| B2 | 168 | 112 | 151 | 56 | aether:4, cogvideox:36, dino:4, f3r:4, lingbot:20, minimax_h3:20, qwen2_5_vl:4, videomae:24, vjepa2:24, wan:28 | l-1:4, l0:20, l1:4, l5:16, l10:16, l11:8, l15:8, l17:8, l20:16, l23:16, l24:4, l25:8, l29:4, l30:12, l35:4, l40:12, l48:4, l50:4 | p8:42, p12:42, p16:42, p24:42 | default8:32, lingbot_video_layer_benchmark:20, minimax_h3_layer_benchmark:20, videomae_vjepa_layer_sweep:40, wan_cog_layer_sweep:56 | CANONICAL_COMPLETE:112, CHECKPOINT_INVALID_OR_INTERRUPTED:11, CHECKPOINT_READY_EVAL_ONLY:1, FEATURE_PARTIAL_OR_MISSING:5, MANIFEST_OR_PROVENANCE_BLOCKED:39 |
| C1 | 136 | 96 | 96 | 40 | aether:4, cogvideox:4, dino:4, f3r:4, lingbot:20, minimax_h3:20, qwen2_5_vl:4, videomae:24, vjepa2:24, wan:28 | l-1:4, l0:16, l1:4, l5:12, l10:12, l11:8, l15:4, l17:8, l20:16, l23:16, l24:4, l25:4, l29:4, l30:8, l40:8, l48:4, l50:4 | p8:34, p12:34, p16:34, p24:34 | default8:32, lingbot_video_layer_benchmark:20, minimax_h3_layer_benchmark:20, videomae_vjepa_layer_sweep:40, wan_cog_layer_sweep:24 | CANONICAL_COMPLETE:96, CHECKPOINT_READY_EVAL_ONLY:28, FEATURE_PARTIAL_OR_MISSING:5, MANIFEST_OR_PROVENANCE_BLOCKED:7 |
| C2 | 136 | 92 | 92 | 44 | aether:4, cogvideox:4, dino:4, f3r:4, lingbot:20, minimax_h3:20, qwen2_5_vl:4, videomae:24, vjepa2:24, wan:28 | l-1:4, l0:16, l1:4, l5:12, l10:12, l11:8, l15:4, l17:8, l20:16, l23:16, l24:4, l25:4, l29:4, l30:8, l40:8, l48:4, l50:4 | p8:34, p12:34, p16:34, p24:34 | default8:32, lingbot_video_layer_benchmark:20, minimax_h3_layer_benchmark:20, videomae_vjepa_layer_sweep:40, wan_cog_layer_sweep:24 | CANONICAL_COMPLETE:92, CHECKPOINT_READY_EVAL_ONLY:32, FEATURE_PARTIAL_OR_MISSING:5, MANIFEST_OR_PROVENANCE_BLOCKED:7 |
| C3 | 132 | 98 | 98 | 34 | aether:4, cogvideox:4, dino:4, f3r:4, lingbot:20, minimax_h3:20, qwen2_5_vl:4, videomae:24, vjepa2:24, wan:24 | l-1:4, l0:16, l1:4, l5:12, l10:11, l11:8, l15:3, l17:8, l20:16, l23:16, l24:4, l25:3, l29:3, l30:8, l40:8, l48:4, l50:4 | p8:34, p12:34, p16:34, p24:30 | default8:32, lingbot_video_layer_benchmark:20, minimax_h3_layer_benchmark:20, videomae_vjepa_layer_sweep:40, wan_cog_layer_sweep:20 | CANONICAL_COMPLETE:98, CHECKPOINT_INVALID_OR_INTERRUPTED:12, CHECKPOINT_READY_EVAL_ONLY:10, FEATURE_PARTIAL_OR_MISSING:5, MANIFEST_OR_PROVENANCE_BLOCKED:7 |

Family/probe 的目标 metric contract：A1 depth=`mean_loss_depth`+`Auc_30`，A1 mask=`mean_loss_identity`，A1 camera=`Auc_30`，A2=`balanced_acc`，B1=`mean_az_err_deg`，B2=`mean_ang_err_deg`，C1/C2=`global_R@1`，C3=`intervention_validity`。B1 没有独立 family canonical CSV，只在 `all_results.csv`；逐行表显式记录这一点。

## 4. 唯一主状态与瓶颈

每一行恰好一个主状态；以下数量之和严格为 1096：

| 主状态 | 数量 |
|---|---|
| CANONICAL_COMPLETE | 642 |
| RAW_RESULT_READY_TO_PACKAGE | 0 |
| CHECKPOINT_READY_EVAL_ONLY | 99 |
| FEATURE_READY_TRAIN_AND_EVAL | 0 |
| FEATURE_PARTIAL_OR_MISSING | 35 |
| CHECKPOINT_INVALID_OR_INTERRUPTED | 83 |
| MANIFEST_OR_PROVENANCE_BLOCKED | 237 |
| NOT_IN_REGISTRY | 0 |
| EXACT_DUPLICATE_ALIAS | 0 |
| DEPRECATED_OR_PROTOCOL_INVALID | 0 |
| UNRESOLVED | 0 |

`NOT_IN_REGISTRY=262` 是正交属性，不会覆盖更有信息量的 checkpoint/feature/provenance 主状态。瓶颈分布：

| Bottleneck | 数量 |
|---|---|
| NONE | 642 |
| PACKAGING | 0 |
| EVAL | 99 |
| TRAINING | 0 |
| FEATURE_EXTRACTION | 35 |
| CHECKPOINT | 67 |
| MANIFEST | 73 |
| CONFIG_OR_CODE | 20 |
| PROVENANCE | 160 |
| DUPLICATE_OR_COLLISION | 0 |
| PROTOCOL_INVALID | 0 |
| UNKNOWN | 0 |

逐行证据含 registry/canonical IDs、summary/prediction、checkpoint epoch/step、期望 final epoch、clean marker、manifest/config/log/run、feature roots 与 provenance。路径存在只证明 artifact existence，不自动证明 protocol validity。

## 5. 三个重点审计

### Default B2 的 32 个 INCOMPLETE

31/32 有 exact `val_summary.json`、`mean_ang_err_deg` 和 `val_predictions.pt`，summary 绑定的 checkpoint 存在且为 epoch 49 final。它们的数值可信到“legacy run 产生了这个数”，但 run manifest 的 split manifest 为 null，n 随 prefix/model 从 67 到 198 不等，而 registry 声称 `configs/splits/inscene15k_v1.json`。因此当前是 `MANIFEST_OR_PROVENANCE_BLOCKED`，不是 Wave A。

唯一例外 `default8_ego_belief_v2_f3r_default_layer_benchmark_l24_p24_hall_s42`：只有 epoch 27 checkpoint，日志进入 epoch 28 后中断，resume 含 stall/EADDRINUSE，无 summary/prediction/clean stop；状态为 `CHECKPOINT_INVALID_OR_INTERRUPTED`，先做 B2 protocol reconstruction，再 clean train/eval，绝不是 eval-only。

### Wan/Cog 162 provisional

162/162 属于 1096，162/162 有可解析 target numeric、历史 run/checkpoint 记录，但 0 个 formal summary，lineage 均为 `PROVISIONAL_ONLY_NO_REPLACEMENT`。旧 callback/retention bug 意味真正 best checkpoint 多数从未保存；“TRULY_MISSING”是 true-best 缺失，不是完全没有 checkpoint。它们不能 packaging。157 个先进入 Wave E；5 个 `FOUND_NEARBY_ONLY` 进入 Wave F 人工容差/协议决策。两个 provisional parent 已有后续 canonical，主状态仍为 canonical，旧数值只保留为 lineage evidence。

### 262 个没有 registry representation 的配置

- 99：现有 manifest + strict final checkpoint，Wave B eval-only；eval 前仍需 live AOSS feature canary。
- 66：manifest/feature 可用，但 checkpoint 未到 final 或无 clean completion，Wave C clean train+eval。
- 42：V-JEPA/VideoMAE checkpoint 存在但历史 filtered manifest 路径已丢失；37 个 checkpoint strict-pass、5 个 strict-fail，统一先 Wave E reconstruct。
- 35：VideoMAE p24 × 5 layers × 7 probes 的 `prefix_000023` feature hole，Wave D。
- 20：MiniMax camera-only config 当前缺失且 filtered manifests 丢失；16 个历史 run 中断、4 个只声明 job。不能假设等同 joint depth+camera，先 Wave E 做 config/protocol audit。

## 6. 执行 Waves（本轮未执行）

`execution_waves.csv` 是逐配置精确列表；以下是聚合。Jobs 列为 feature/train/eval。训练点估使用历史成功运行的 probe median，低/高区间为 0.5×/2× 规划带；eval 点估 0.05 GPU-h/row、p90 0.13。feature 成本在 canary 前不伪造。

| Wave | 配置 | GPU jobs F/T/E | GPU-hours | 新增 checkpoint 盘 |
|---|---|---|---|---|
| Wave A | 0 | 0/0/0 | feature 0.00 (0.00–0.00); train 0.00 (0.00–0.00); eval 0.00 (p90 0.00) | ckpt 0.0–0.0 GiB; feature 0.0 GiB (budget 0.0) |
| Wave B | 99 | 0/0/99 | feature 0.00 (0.00–0.00); train 0.00 (0.00–0.00); eval 4.95 (p90 12.87) | ckpt 0.0–0.0 GiB; feature 0.0 GiB (budget 0.0) |
| Wave C | 66 | 0/66/66 | feature 0.00 (0.00–0.00); train 1553.22 (776.62–3106.44); eval 3.30 (p90 8.58) | ckpt 26.4–122.8 GiB; feature 0.0 GiB (budget 0.0) |
| Wave D | 35 | 2/35/35 | feature 0.75 (0.60–1.05); train 671.00 (335.50–1342.00); eval 1.75 (p90 4.55) | ckpt 14.0–65.1 GiB; feature 40.7 GiB (budget 45.0) |
| Wave E | 249 | 0/181/218 | feature 0.00 (0.00–0.00); train 3843.68 (1921.69–7687.36); eval 10.90 (p90 28.34) | ckpt 72.4–336.7 GiB; feature 0.0 GiB (budget 0.0) |
| Wave F | 5 | 0/5/5 | feature 0.00 (0.00–0.00); train 100.32 (50.15–200.64); eval 0.25 (p90 0.65) | ckpt 2.0–9.3 GiB; feature 0.0 GiB (budget 0.0) |

逐 wave 分布（完整 layer × 配置列表以 `execution_waves.csv` 为准）：

- Wave A: 0
- Wave B: models=lingbot:41, minimax_h3:18, wan:40; families=A1:13, A2:15, B2:1, C1:28, C2:32, C3:10; campaigns=lingbot_video_layer_benchmark:41, minimax_h3_layer_benchmark:18, wan_cog_layer_sweep:40; layers=l0:7, l5:7, l10:19, l15:7, l20:13, l25:6, l29:6, l30:11, l40:11, l48:9, l50:3; prefixes=p8:27, p12:32, p16:21, p24:19
- Wave C: models=lingbot:44, minimax_h3:10, wan:12; families=A1:34, A2:10, B2:10, C3:12; campaigns=lingbot_video_layer_benchmark:44, minimax_h3_layer_benchmark:10, wan_cog_layer_sweep:12; layers=l0:2, l5:2, l10:12, l15:2, l20:10, l25:2, l29:2, l30:11, l40:12, l48:8, l50:3; prefixes=p8:20, p12:28, p16:6, p24:12
- Wave D: models=videomae:35; families=A1:10, A2:5, B2:5, C1:5, C2:5, C3:5; campaigns=videomae_vjepa_layer_sweep:35; layers=l0:7, l5:7, l11:7, l17:7, l23:7; prefixes=p24:35
- Wave E: models=aether:4, cogvideox:96, dino:4, f3r:4, minimax_h3:20, qwen2_5_vl:4, videomae:14, vjepa2:36, wan:67; families=A1:90, A2:55, B1:44, B2:39, C1:7, C2:7, C3:7; campaigns=default8:32, minimax_h3_layer_benchmark:20, videomae_vjepa_layer_sweep:42, wan_cog_layer_sweep:155; layers=l-1:4, l0:31, l1:4, l5:27, l10:29, l11:4, l15:19, l17:4, l20:12, l23:33, l24:4, l25:23, l29:10, l30:15, l35:11, l40:15, l50:4; prefixes=p8:63, p12:68, p16:65, p24:53
- Wave F: models=cogvideox:2, wan:3; families=A1:1, A2:1, B1:2, B2:1; campaigns=wan_cog_layer_sweep:5; layers=l15:3, l25:1, l40:1; prefixes=p8:4, p24:1

- **Wave A：0 当前可直接 promotion。** 31 个 B2 只有在 Wave E 的 split/protocol/cardinality audit 通过后才转入 A。必须 staging copy + expected-ID-only diff + generalized atomic promotion + project consistency；不得直接写 protected tables。
- **Wave B：99。** 入口 `build_missing_eval_jobs.py` / A1 专用 builder；gate 是 strict final checkpoint、manifest SHA、feature root live canary、exact n/finite/identity、staging audit。任一不符即停止。
- **Wave C：66。** 复用 probe-fill chain 但必须改为只读入 frozen reconciliation、no-resume、strict clean-completion；现有宽松 done scanner 不可原样使用。
- **Wave D：35。** 5 个 layer cache groups 由一次共享五层 extraction 产生：1 个单 scene canary + 1 个 4×H800 full job，合计点估 0.75 H800-GPUh（0.60–1.05），新增 40.69 GiB（预留 45 GiB）。历史 runner 未传 `--num-frames 24`，而 extractor 默认 16 会静默过滤 p24；必须显式 `PREFIX_LENGTHS=24 --num-frames 24`，canary 看到 `prefix_000023` 五层后才可全量。历史成本证据为 `logs/videomae_streaming_extract_20260807_161245/supervisor.log`（`9923a2862d2512f64726b54c47b7e5ad6c062be44c18c97c05816f894fa372c9`）和 `logs/videomae_6115_repair_sco_20260809_163604/aoss_sync.log`（`b6bc8bc3fb9e9c4c19fff7e121074e0f626171782a4dd5a93d867c658b8b9d4a`）。
- **Wave E：249。** provenance reconstruction 本身 0 GPU。包括 155 provisional、31 B2 summaries、42 missing-manifest checkpoints、20 MiniMax camera、1 F3R B2。重建失败则保持 blocked，不启动下游；成功后转 A/B/C。
- **Wave F：5。** `FOUND_NEARBY_ONLY` 必须签名的人为 protocol/tolerance disposition；不得以相似文件名或忽略 variant 的方式合并。

每个 wave 的入口脚本目前多为 untracked。硬 gate：`plan 中脚本 SHA == frozen provenance copy SHA == launcher 记录 SHA`；否则停止。canonical eligibility 还要求 full identity、protocol、manifest/root/checkpoint hashes、finite metric、expected n 以及 staging-only promotion diff 全部通过。

## 7. 最短补齐路径与成本

在“31 个 legacy B2 经审计可 packaging”的最短路径假设下：

- 31 packaging promotions（0 GPU）；
- 287 clean train jobs；
- 423 eval jobs；
- 5 个 VideoMAE layer cache groups，由 2 个 feature jobs（1 canary + 1 shared full extraction）产生；
- 合计 **712 GPU jobs**（287 train + 423 eval + 2 feature）；
- 训练点估 **6168.22 GPU-h**，规划带 **3083.96–12336.44 GPU-h**；
- eval 点估 **21.15 GPU-h**，按历史 p90 **54.99 GPU-h**；
- feature extraction 点估 **0.75 H800-GPUh**（0.60–1.05），新增 **40.69 GiB**、预算 45 GiB；canary 不通过则停止，不能授权 full job；
- 已知总点估 **6190.12 GPU-h**，规划 envelope **3105.71–12392.48 GPU-h**（训练 + eval + feature；硬件代际混合，需按实际队列换算）；
- protected best+last checkpoint 预计新增 **114.8–533.8 GiB**；加 feature 点估后合计 **155.5–574.5 GiB**（按 45 GiB feature budget 则预留 159.8–578.8 GiB）。

如果 31 个 B2 legacy protocol 不能证明等价，最短路径再增加 31 个 exact-manifest eval jobs（点估 1.55 GPU-h，p90 4.03），但通常不需要重训其 epoch-49 checkpoint。

## 8. 哪些不能直接跑、哪些不应静默删除

- 不应先扩 sweep；先补齐这 1096。
- 不应直接运行旧 probe-fill done scanner；它会把“有 checkpoint/eval 或 END rc=0”误算为完成。
- 不应把 162 provisional 或 31 B2 summary 当 packaging-ready。
- 不应把 F3R B2 p24 当 eval-only。
- 不应把 MiniMax camera 静默视作 joint depth/camera alias。
- 不应合并 21 个 model_variant collision groups。
- 没有任何配置因“价值低”被删；deprecated/invalid 目前为 0，所有 1096 行均保留。

## 9. 保护文件完整性

生成脚本在写分析产物前后重新计算 registry 和全部 canonical CSV 的 SHA256，并要求完全一致。执行后哈希见 `universe_summary.json.protected_source_hashes_before/after`。本报告及五个分析表之外没有实验状态写入。

| Protected source | SHA256 before | SHA256 after | unchanged |
|---|---|---|---|
| experiments/registry.csv | efc533bbe6286090d6c111c9d272ffb28cf7477b937f799f312eedf2a944431e | efc533bbe6286090d6c111c9d272ffb28cf7477b937f799f312eedf2a944431e | YES |
| results/canonical/a1_results.csv | f7777676eafbc77dc8a8856e86a0901c98d6b5b3bdaddb568a975ef1c9f94d86 | f7777676eafbc77dc8a8856e86a0901c98d6b5b3bdaddb568a975ef1c9f94d86 | YES |
| results/canonical/a2_results.csv | 2613650bc7dcaa7a5bb5292c1953b9742657fbcb7967da3016b8c84c9f6ffa5f | 2613650bc7dcaa7a5bb5292c1953b9742657fbcb7967da3016b8c84c9f6ffa5f | YES |
| results/canonical/all_results.csv | 2a04a5e1f0165f1d7a06e7edcff38d956b845910ad95af3f7bbede3704133b17 | 2a04a5e1f0165f1d7a06e7edcff38d956b845910ad95af3f7bbede3704133b17 | YES |
| results/canonical/b2_results.csv | 0d4941709af83ba72820e4c463b96775d240096b44fca14bd4c5243e2dfa5f20 | 0d4941709af83ba72820e4c463b96775d240096b44fca14bd4c5243e2dfa5f20 | YES |
| results/canonical/c1_results.csv | b88a37e239d150d297a4e97838d93888d5b6d3d0780df510e65b7f916b2ade1a | b88a37e239d150d297a4e97838d93888d5b6d3d0780df510e65b7f916b2ade1a | YES |
| results/canonical/c2_results.csv | 8c96a38500420af11427ed506bc92d4e1fc9d6b636bc34f0c4baf2d279fd9524 | 8c96a38500420af11427ed506bc92d4e1fc9d6b636bc34f0c4baf2d279fd9524 | YES |
| results/canonical/c3_results.csv | 6cb2bec11abecb7cdf7549f14db84d58345e2bb5e9a0172f9435c80e5e2b7c5d | 6cb2bec11abecb7cdf7549f14db84d58345e2bb5e9a0172f9435c80e5e2b7c5d | YES |

## 10. 对九个执行问题的直接回答

1. **已有目标数值 833；任何 exact finite numeric 也是 833。** F3R B2 l24/p24 的 top1 是 n/a，不构成数值结果。
2. **Canonical-valid 642 个 parent configs。** 734 个 in-universe atomic IDs 映射到这 642 行，其中 732 个 primary、2 个 same-parent alternate 已分栏；另有 27 个 canonical IDs 在 frozen universe 外。
3. **完全没有任何数值 263 个。** 包括 262 个无 registry representation rows，加上有 INCOMPLETE registry 但没有任何 finite metric 的 F3R B2 l24/p24；逐项见 CSV 过滤 `auxiliary_numeric_present=false`。
4. **现在只差 packaging 的严格数量 0。** 31 个 B2 是“通过 protocol audit 后条件性 packaging”。
5. **只差 eval 99。** 精确列表为 `execution_waves.csv` 的 Wave B。
6. **formal 补齐计划需要 clean training 287。** 包括条件性通过 E/F gate 后的 replacement jobs；逐项看 `gpu_train_jobs=1`。
7. **需要重新抽 feature 35 个配置 / 5 个共享 VideoMAE cache groups。** 全部是 p24 的 5 layers × 7 probes。
8. **主状态 manifest/provenance blocked 237。** 另有 17 个以 checkpoint/config 为主状态但也必须先经过 Wave E/F gate，因此 Wave E/F 合计 254。
9. **最短路径：31 条件性 packaging + 287 train + 423 eval + 2 feature jobs（产出 5 cache groups）。** 共 712 GPU jobs；训练点估 6168.22 GPU-h、规划带 3083.96–12336.44，eval 21.15（p90 54.99），feature 0.75（0.60–1.05），总点估 6190.12、总规划 envelope 3105.71–12392.48 GPU-h。

## 11. 硬性验收

下面的 `NUMERIC_RESULT_PRESENT` 指“存在该 family 的目标 metric”；F3R B2 的 top1=n/a。`NEEDS_TRAINING`、`NOT_IN_REGISTRY`、collision 是正交执行/审计计数；唯一主状态的 partition 见第 4 节。

```text
UNIVERSE_SIZE: 1096
UNIVERSE_SOURCE: experiments/task_status_1096_before_expand_20260822.csv
UNIVERSE_SHA256: 9c7d6807b701de4b052819971a5c011e29c717e1c3e958ec9ae23116cdcfe779
DIFFUSION_STEP_SWEEP_EXCLUDED: true
CANONICAL_COMPLETE: 642
NUMERIC_RESULT_PRESENT: 833
REGISTRY_INCOMPLETE: 194
NOT_IN_REGISTRY: 262
READY_TO_PACKAGE: 0
EVAL_ONLY: 99
NEEDS_TRAINING: 287
NEEDS_FEATURE_EXTRACTION: 35
PROVENANCE_BLOCKED: 237
DUPLICATE_OR_COLLISION: 21 groups / 42 registry IDs; 0 exact duplicate aliases
DEPRECATED_OR_INVALID: 0
UNRESOLVED: 0
RESOLVED_TOTAL: 1096
COUNT_SUM_EQUALS_1096: true
```

逐行答案位于 `results/analysis/q00_1096/universe_1096_reconciled.csv`；454 个缺失/非 canonical 配置位于 `missing_or_noncanonical.csv`；精确 waves 与成本位于 `execution_waves.csv`。
