# MiniMax bf16 重提与 B2 重训 — 执行记录（2026-10-16）

计划：[MINIMAX_REEXTRACT_RETRAIN_PLAN_20261016.md](MINIMAX_REEXTRACT_RETRAIN_PLAN_20261016.md)。
用户批准的范围：完成阶段 0 的代码合并与冻结；跑阶段 1 的提特征 canary；立即提交阶段 4 的
48 个 B2 重训（`PROBE_FP16_PASSTHROUGH=0`）；阶段 2/3 的 32 个提特征作业在 canary 通过后提交。
阶段 5（128 个 timestep MiniMax L20/L25 重训）不在本次批准范围内。
本文件所有改动均未 commit、未 push。

状态标记：**本次实际执行通过** = 本次会话里运行并看到结果；**依据历史记录核验** = 读已有
日志或产物判断；**尚未执行** = 只写了代码或命令，还没跑。

## 结论（截至 2026-10-16 15:15 UTC，跟进会话）

- **80 个作业没有失败（本次实际执行通过，`sco acp jobs list` 15:14 UTC）。** B2 重训 48 个：
  2 个 SUCCEEDED、46 个 RUNNING。提特征分片 32 个：4 个 RUNNING、28 个 STARTING。
  STARTING 是排队：zoetrope 池被本批 B2 重训和其他用户的 8 卡作业占满，不是故障。
- **已结束的 2 个 B2 重训，最终 checkpoint 全部有限（本次实际执行通过）。** MiniMax t500 与 t1000，
  L15 p8，均为 epoch 49（`max_epochs=50` 的最后一个），各 18,532,225 个浮点参数，非有限 0 个。
  记录：`artifacts/release_audit_20261015/b2_fp32_retrain/final_ckpt_finite_20261016.jsonl`
  （脚本同目录 `b2_final_ckpt_finite_20261016.py`；重跑只追加新结束的 run）。范围：只含已写完成标记的 2 个 run。
- **正在跑的 4 个分片进度正常。** 15:00 UTC 时 t500 streaming s0/s2/s4 各完成 4–5 个场景，约 11 分钟一个场景，
  按此估算 streaming 分片约 16 h（计划估 12 h）；已写出的单元全部属于 fp16 原索引中本分片的场景，没有多出的单元。
- **合并、全量验证、阶段 5 的工具已写好并通过测试（尚未对生产数据执行）。** 详见下文
  “阶段 3 收尾与阶段 5 的工具”。全量 CPU 测试 487 passed、3 skipped、1 deselected（552 s）；
  3 个 skipped 是等待安装的新版组脚本的测试，在暂存副本里全部通过。
- **阶段 5 的配置与被替换的 128 个旧 run 一致（本次实际执行通过）。** 用组脚本的 override 重新组合 Hydra 配置，
  与旧 run 保存的 `config.yaml` 逐键比较：128/128 无差异（忽略运行名、路径、日志器、特征根、`autoresume`）。
  旧 run 来自 `tstep_probe_opt_v3_20260912_highest_r1`（101 个）与 `tstep_probe_pack_v4_1_20260914_prod_r1`
  / `canary_r3`（27 个），不是 09-09 的 MiniMax 战役。其中 17 个旧 run 是从更早的半程 checkpoint 续训的，
  新 run 从头训练。报告：`artifacts/release_audit_20261015/stage5_config_parity/config_parity_20261016.json`。
- **合并与阶段 5 还要等分片跑完。** 按现在的排队，最后一批分片（t1000 target）大约在 10-17 上午（UTC）
  才能拿到卡，target 分片按计划约 38 h，合并最早约在 10-18 夜间到 10-19。这只是估计，取决于集群排队。
- 本次跟进新增与修改的文件均未 commit、未 push，没有提交阶段 5 的任何作业。
- **阶段 5 已有条件批准（约 15:22 UTC）**：合并与全量验证 PASS、16 组干跑无误后，先提交 1 组 node8 试点，
  首个 checkpoint 有限后再提交其余 15 组。条件尚未满足（分片还在排队），见“需要你决定的事项”第 4 项。

## 结论（截至 2026-10-16 14:10 UTC）

- **阶段 1 canary 通过（第 4 次，本次实际执行通过）。** 场景 0 的 4 个步骤全部 OK，验证报告 PASS。
  68 个特征文件都是 bf16、全部有限；在原 fp16 有限的位置，新旧值完全相同，相对误差中位数与 p99 均为 0。
  原 fp16 文件 66/68 含 inf（共 1,413 个值），新文件最大绝对值达 5.5e5。结构文件与原文件逐字节一致。
- **阶段 2/3 的 32 个单卡分片已于 14:04 UTC 提交（作业号见下文表格）。**
- **第一个 B2 重训已结束**：MiniMax t500 L15 p8，跑满 50 个 epoch，最终 checkpoint 1,853 万个参数中非有限 0 个。

以下为 12:00 UTC 时的结论，保留作记录。

- **阶段 0 已完成（本次实际执行通过）。** bf16 存储契约已合入主代码。CPU 测试 472 个通过，
  其中 7 个是新增的存储契约测试、2 个是新增的同步循环测试；10-15 基线是 463 个。
  当前使用的冻结运行时是 r2（`runtime/tstep_minimax_bf16_v1_extractor_r2`），树摘要
  `5d2cb435ca2aeb22f7c23de1c60ef7a55344d900ecc018524356a6810ca423b2`。
  r1（`c27d44e8…b73c`）保留不动，只用于 canary 第 1、2 次。
- **阶段 4 的 48 个 B2 重训全部在 zoetrope 上运行。** canonical 16 个首次提交即运行；
  timestep 32 个首次提交全部在 1 分钟内失败，原因是作业以 root 身份运行、找不到 AOSS 配置文件。
  修复后第 2 次提交的 32 个正在运行。11:10 UTC 读到的 46 个 checkpoint（48 个 run 中的 46 个，
  另 2 个在读取时已被轮换删除）参数全部有限，所在 epoch 为 1–26。修复前同类 checkpoint
  约 99.7% 参数非有限。这是早期证据，最终 checkpoint 还需同样检查。
- **阶段 1 canary 前三次失败，第 4 次运行中。** 前两次失败都发生在第一个步骤（t500 streaming），
  没有发布任何特征。第 3 次的第一个步骤通过并发布了 32 个单元，第二个步骤没有启动。
  - 第 1 次（`pt-6g2hggcx`）：新命名空间的本地特征目录还不存在，同步循环启动即退出。
    包装脚本改为先建目录。
  - 第 2 次（`pt-fxzmatkf`）：提特征程序先写场景 0 的特征，场景结束才写 `prefix_index.npy`；
    r1 的同步循环把“特征齐、索引未到”的单元当作错误并终止整个步骤。2026-09 的战役没有触发，
    因为那些命名空间里已经有本地索引文件。修复后冻结为 r2（见下文）。
  - 第 3 次（`pt-wgghxqsc`，run tag `canary_20261016_r3`，r2 运行时）：t500 streaming 通过，
    发布 32 个单元，与原 fp16 索引的场景 0 层 20/25 条目数一致。随后 t500 target 拒绝启动：
    上传工具 `ads-cli` 在当前目录（即冻结树）写了一个空的 `syncRecord.txt`，冻结树校验发现多出的文件。
    修复：新增 `scripts/ads_cli_in_logdir.sh`，让上传工具在作业自己的日志目录里运行，
    包装脚本同时设置 `PYTHONDONTWRITEBYTECODE=1`；删除冻结树里的这个空文件后，r2 校验 PASS，摘要不变。
  - 第 4 次（`pt-vsbr4usy`，run tag `canary_20261016_r4`，r2 运行时）：运行中。它复用第 3 次已发布的
    t500 streaming 单元（同一冻结树产生），验证器会重新检查全部已发布文件。
- **阶段 2/3 尚未提交。** 提交脚本在 canary 报告为 PASS 之前会拒绝提交。32 个单卡分片作业的
  脚本和干跑已就绪。

## 术语

- **canary**：正式提交前的小规模试跑。这里指 1 个场景、2 个 timestep、2 种模式的提特征，
  加一份与原 fp16 缓存逐文件比对的验证报告。
- **done-index（完成索引）**：同步循环在把一个特征文件上传 AOSS 并确认后追加的一行相对路径。
  提特征程序据此跳过已完成的单元。
- **keep_local 结构文件**：`prefix_index.npy`（streaming）、`target_indices.npy` 与
  `target_indices.meta.json`（target）。它们上传 AOSS 后仍保留本地副本，训练读本地副本。
- **step lock（步骤锁）**：一个文本文件，写着拥有该提特征步骤的日志目录；别的日志目录启动同一步骤时会跳过。
- **`PROBE_FP16_PASSTHROUGH`**：1 表示 fp16 特征原样送到 GPU，0 表示在数据加载器里先转成 fp32。
  CogVideoX、Aether、MiniMax 的 B2 在 1 下训练成 NaN。

## 阶段 0：代码合并与冻结

| 文件 | 改动 |
|---|---|
| `features/run_inscene15k.py` | 新增 `MINIMAX_STORAGE_CONTRACT`、`validate_finite_feature`、`prepare_feature_for_storage`；5 处 `.half()` 落盘改为按 VFM 选择存储 dtype（MiniMax 为 bf16 并在转换前后检查有限性，其它 VFM 仍为 fp16）；发布前再查一次；sidecar 写入 `feature_storage_contract`、`feature_storage_dtype`、`finite_check`；登记两个新命名空间 |
| `features/minimax_h3/minimax_h3_feature.py` | 前向输出含 NaN/inf 时报错，不落盘 |
| `scripts/run_streaming_feat_extract_8gpu.sh` | 新命名空间加入白名单；seeded-v1 的协议、模型、prompt、done-index 命名约束同样适用于新命名空间；`*_MINIMAX_BF16_V1` 只允许 `VFM=minimax_h3` |
| `scripts/run_tstep_sync_loop.sh` | 识别新命名空间为 seeded；校验模式下检查存储 dtype：bf16 命名空间必须是 bf16 且有契约字段，其它命名空间必须是 fp16 |
| `scripts/run_tstep_probe_group.sh` | `PROBE_FP16_PASSTHROUGH` 可由环境覆盖（默认仍为 1，复现 2026-09）；新增 `AOSS_CONF` 默认值与可读性检查 |
| `tests/test_minimax_bf16_storage_contract.py` | 新增 7 个测试 |

- 测试（本次实际执行通过，登录节点 CPU）：针对性 28 个通过；全量 470 passed、1 deselected，用时 385 s。
- 冻结 r1（本次实际执行通过）：`runtime/tstep_minimax_bf16_v1_extractor/`，13 个文件，
  `source.sha256` 的摘要为 `23ad5b068b9f2b8073e67fd972fc682c37d6574a9e72e37c83a7bc1b71444839`。
  冻结时主树 git HEAD 为 `2f19ecb`，工作区有未提交改动。

### 阶段 0 补充：同步循环修复与 r2 冻结（canary 第 2 次失败之后）

- `scripts/run_tstep_sync_loop.sh`：快照里特征已齐、但场景结构索引（streaming 的
  `prefix_index.npy` 与 sidecar；target 的 `target_indices.*` 三个文件）尚不存在的单元，
  记为“等待中”，写入 `round_NNNN.pending.txt`，本轮不发布、不报错。提特征停止后若仍有等待中的单元，
  同步循环以非零退出并列出这些单元。索引存在但是符号链接或非普通文件，仍按原规则报错。
- `tests/test_tstep_sync_loop.py` 新增 2 个测试：运行中缺索引的单元被延后，索引出现后正常发布；
  target 缺结构文件且提特征已停止时失败、不上传、不删除。原有“停止状态下缺索引必须失败”的测试不变且通过。
- 测试（本次实际执行通过）：同步循环模块 28 个通过；全量 472 passed、1 deselected，用时 486 s。
- 冻结 r2（本次实际执行通过）：`runtime/tstep_minimax_bf16_v1_extractor_r2/`，13 个文件，
  树摘要 `5d2cb435ca2aeb22f7c23de1c60ef7a55344d900ecc018524356a6810ca423b2`，
  `source.sha256` 摘要 `5c97d4922d6d7ba8001ed2e979acc20f9eb1aa7fd1b3e2424abc6fc050bb72bc`。
  与 r1 相比只有同步循环不同。两个包装脚本与提交脚本已指向 r2；提交脚本只允许在还没有生产作业时更换冻结树，
  旧树记在 ledger 的 `tree_history`。

## 阶段 4：B2 重训（48 个，单卡，zoetrope）

唯一的科学设置变化是 `PROBE_FP16_PASSTHROUGH=0`。数据、manifest、配置、种子与原 run 相同。
新输出目录与旧目录分开，旧产物未改动。

### canonical 16 个

- 战役目录：`artifacts/main_v2_probe_v2_816_20260917/single_gpu_b2_fp32_20261016_r1/`
  （`items/` 为原 item 的逐字节拷贝，`plan.json`，`submissions.json`）。
- 运行器：同目录 `run_single.py`，sha256 `ccdc2e9d2ec19a9a6a26b73b9f516cbe2465b756b2a1b72cc44f13bbec3ae041`。
  CogVideoX 用 `runtime/main_v2_probe_v2_r7_single_v3` + cog96 overlay，Aether 用
  `runtime/main_v2_probe_v2_aetherfix_r1` + aether_target8 overlay，与原 run 相同。
- 每个作业先跑 2 步 canary，再跑正式训练。16 个 canary 全部 rc=0（本次实际执行通过）。
- 输出：`logs/inscene15k_streaming/runs/main_v2_single_gpu_b2_fp32_20261016_r1/production/…/<unique_key>/`。

| cell | p8 | p12 | p16 | p24 |
|---|---|---|---|---|
| Aether L1 | pt-bcje825e | pt-dmr3tft4 | pt-c6dudegv | pt-bhw1bj29 |
| CogVideoX L0 | pt-enb6vj25 | pt-yrv16mda | pt-g9urvx3l | pt-92903zr3 |
| CogVideoX L5 | pt-c4np1at7 | pt-kw90a1si | pt-bxmol5wi | pt-qyp6ocgx |
| CogVideoX L10 | pt-o9nrcu2f | pt-ulouhn57 | pt-weu6u1e8 | pt-sbvqz1ap |

### timestep 32 个

- 战役目录：`logs/tstep_probe_campaigns/tstep_probe_b2_fp32_20261016_r1/`。
  `matrix/groups.tsv` 从 2026-09 两个战役的矩阵逐行复制，只把 `tasks` 改为 `ego_belief_v2`；
  `matrix/provenance.json` 记录来源矩阵与严格审计文件的 sha256。8 个严格审计文件的
  sha256 与矩阵记录一致（本次实际执行通过）。
- 训练入口：`scripts/run_tstep_probe_group.sh`，`GPU_IDS_CSV=0`，`PROBE_FP16_PASSTHROUGH=0`。
- 输出：`logs/inscene15k_streaming/runs/tstep_probe_seeded_v1/tstep_probe_b2_fp32_20261016_r1/tp_<cell>_ego_belief_v2/`。
- 第 1 次提交（06:52 UTC）32 个全部失败：`aoss_client` 报 `ConfigFileNotFoundError(/root/.petreloss.conf)`。
  作业以 root 运行，家目录没有 AOSS 配置。修复：作业环境与组脚本固定
  `AOSS_CONF=/mnt/umm/users/yinbaiqiao/aoss/petreloss.conf`。旧作业号保存在 ledger 的
  `previous_attempts` 中。
- 第 2 次提交（约 07:05 UTC）的作业号：

| cell | p8 | p12 | p16 | p24 |
|---|---|---|---|---|
| CogVideoX t499 L15 | pt-tm01q8vx | pt-s1mfnsob | pt-vhfdiirr | pt-s1ybydz7 |
| CogVideoX t499 L20 | pt-jxstw15v | pt-gqxot0w0 | pt-194x43p2 | pt-s8lxwbc8 |
| CogVideoX t499 L25 | pt-o9fph89m | pt-e1baj8ex | pt-ri9o06q4 | pt-oa6xwxqr |
| CogVideoX t999 L15 | pt-mxpf4t79 | pt-0khpjrls | pt-9o5ij3di | pt-pex1i4u0 |
| CogVideoX t999 L20 | pt-su0oxwy1 | pt-s2ggrug8 | pt-xmw0rva0 | pt-xfjlbqbw |
| CogVideoX t999 L25 | pt-vi7k3veu | pt-zpfnhbsd | pt-kegrmlyj | pt-8507ha1r |
| MiniMax t500 L15 | pt-m19dq6xo | pt-5a19oiki | pt-tp0ds16c | pt-ao06rxg5 |
| MiniMax t1000 L15 | pt-769lqk6k | pt-plpqvuta | pt-qobkyqsd | pt-azc6nvej |

### 早期证据

- `artifacts/release_audit_20261015/b2_fp32_retrain/early_ckpt_finite_20261016T1110Z.jsonl`
  （脚本同目录 `b2_ckpt_finite_20261016.py`）：46 个 checkpoint，非有限参数 0 个，epoch 1–26。
  范围：读取时刻存在的 checkpoint，不是最终权重。

## 阶段 1：提特征 canary

- 范围：场景 0（`00777c41d4`），t500 与 t1000，streaming 与 target，层 20/25，1 张 H800。
  从冻结运行时执行，同步循环使用校验模式（`SYNC_TRUST_EXTRACT=0`，逐文件哈希并检查 sidecar 契约）。
- 第 1 次 `pt-6g2hggcx`（10:56 UTC 启动）：失败。冻结的同步循环启动时 `cd` 到
  `<输出根>/minimax_h3/scannetpp`，新命名空间里这个目录还不存在。启动器因此没有启动 GPU 进程。
  没有特征写出，done-index 为空，冻结树复核仍 PASS。日志：`logs/tstep_minimax_bf16_v1/canary_20261016/`。
- 修复：两个包装脚本在启动前创建该目录。包装脚本不在冻结树内，冻结树未改。
- 第 2 次 `pt-fxzmatkf`（run tag `canary_20261016_r2`，11:06–11:16 UTC）：失败。模型 11:10 加载完，
  场景 0 第一个前缀的两层特征写出后，同步循环的快照报 `missing/non-regular structural index`
  并退出，看门狗随即终止提特征进程。本地留下 2 个未发布的特征文件（root 所有、权限 0600），
  重跑会原子覆盖。日志：`logs/tstep_minimax_bf16_v1/canary_20261016_r2/`。
- 第 3 次 `pt-wgghxqsc`（run tag `canary_20261016_r3`，r2 运行时，11:58–12:18 UTC）：失败于第二步启动前
  的冻结树校验（原因与修复见“结论”）。第一步日志：`logs/tstep_minimax_bf16_v1/canary_20261016_r3/`。
- 第 4 次 `pt-vsbr4usy`（run tag `canary_20261016_r4`，r2 运行时，12:22–14:03 UTC）：**SUCCEEDED，报告 PASS**。
  提交脚本只允许复用场景 0、且由同一冻结版本发布的条目，复用条目数记在 ledger 的 `reused_published_units`
  （复用了第 3 次的 32 个 t500 streaming 单元）。冻结树在整个作业期间没有出现新文件。

| 步骤 | 文件数 | dtype | 形状 | 相对误差中位数 / p99 | 原 fp16 含 inf 的文件（值个数） | 新文件最大绝对值 |
|---|---:|---|---|---|---|---:|
| t500 streaming | 32 | bf16 | 22×15×26×5376 | 0 / 0 | 32（928） | 5.5e5 |
| t500 target | 2 | bf16 | 56×15×26×5376 | 0 / 0 | 2（48） | 2.6e5 |
| t1000 streaming | 32 | bf16 | 22×15×26×5376 | 0 / 0 | 30（424） | 4.2e5 |
| t1000 target | 2 | bf16 | 56×15×26×5376 | 0 / 0 | 2（13） | 1.2e5 |

  解释：MiniMax 以 bf16 计算，bf16 值在 fp16 的数值范围内可以被 fp16 精确表示，所以原 fp16 缓存在有限位置上
  与模型输出完全一致；bf16 存储同样无损，差别只在原来溢出为 inf 的位置。单个 target 文件出现过一次
  2.98e-6 的最大相对误差，来自 fp16 次正规数的舍入。范围：只有场景 0，不代表其余 681 个场景都无问题。
- 验证器 `scripts/verify_tstep_minimax_bf16_v1_canary.py` 在 canary 执行到验证步骤之前修订过一次：
  原 fp16 的 `prefix_index.npy` 是 object 数组，旧版用 `allow_pickle=False` 读取会直接崩溃。
  修订后结构文件改为逐字节比对，任何异常都记入报告而不是中断。修订前后 sha256：
  `92486af0…835f7d` → `5d5340de…90c7c5`。报告里记录验证器自身的 sha256。
- 通过条件（全部满足才 PASS）：新 done-index 的单元集合与原 fp16 索引在同场景、同层上一致；
  每个文件是 bf16、全部有限、形状与原文件相同；在原文件有限的位置，相对误差中位数 ≤ 1%、
  p99 ≤ 5%（分母下限 0.01）；sidecar 的契约、dtype、命名空间、协议 sha、载荷 sha256 正确，
  生成来源字段与原文件相同；结构文件与原文件逐字节一致，sidecar 除命名空间外一致，AOSS 副本等于本地副本。
- 报告路径：`logs/tstep_minimax_bf16_v1/canary_20261016_r4/verify_report.json`。

## 阶段 2/3：32 个单卡分片（14:04 UTC 已提交，运行中）

| 步骤 | s0 | s1 | s2 | s3 | s4 | s5 | s6 | s7 |
|---|---|---|---|---|---|---|---|---|
| t500 streaming | pt-koz8l4fe | pt-mg0ys38d | pt-dzr3r41g | pt-7cn7itzu | pt-16odcxpp | pt-057eetkd | pt-kjb1jji6 | pt-x648s51s |
| t500 target | pt-nqer8fgs | pt-nauh4wtt | pt-a898tesg | pt-tzde1jt7 | pt-95hns3is | pt-a9e2bkz9 | pt-wmg7b0tp | pt-n4lbl2pf |
| t1000 streaming | pt-pmknl998 | pt-yoc9imrh | pt-g70px4wx | pt-z2nxl1kk | pt-yz9312w4 | pt-txt7e5v8 | pt-0enw35gi | pt-txwj39yd |
| t1000 target | pt-lrz39wdf | pt-8d7vzmvc | pt-3daeyswc | pt-petfysgj | pt-0e5zet5z | pt-64xkqyir | pt-54lym2h5 | pt-jqxph5ac |

ledger：`logs/tstep_minimax_bf16_v1/submissions.json`（每个作业记录冻结版本、树摘要、分片包装脚本 sha256、
canary 报告 sha256）。下面是设计说明。

- 形状与计划一致：2 个 timestep × 2 种模式 × 8 个场景分片，每个作业 1 张卡，共 32 个。
- 每个分片有自己的本地根目录、done-index 与步骤锁，只共享 AOSS 命名空间（场景键不重叠）。
  原因：冻结的同步循环合并 done-index 时不加锁；结束时的验收会统计本地根下全部特征文件。
  多个作业共用一个根或一个索引，会丢索引行并让彼此验收失败。
- 分片 i（共 8 个）覆盖场景 `[682·i/8, 682·(i+1)/8)`，分片 0 从场景 1 开始，场景 0 由 canary 完成。
- 包装脚本：`scripts/run_tstep_minimax_bf16_v1_extract_shard.sh`。
  本地根：`InsScene-15K/tstep_minimax_bf16_v1_shards/t<T>_<mode>_s<i>of8/<命名空间>/`。
  分片状态：`logs/tstep_minimax_bf16_v1/production_20261016/shards/<分片>/`。
- 登录节点干跑（本次实际执行通过）：t1000 target 分片 0 的配置校验通过，场景范围 1:85。
- 预计耗时（沿用计划的实测吞吐）：streaming 每作业约 12 h，target 每作业约 38 h。
- 全部分片 PASS 之后的合并、审计与阶段 5 准备：脚本已在 15:10 UTC 写好并测试，见下一节；尚未对生产数据执行。

## 阶段 3 收尾与阶段 5 的工具（10-16 15:10 UTC 编写，测试通过，尚未对生产数据执行）

### 新增与修改的文件

| 文件 | 作用 |
|---|---|
| `scripts/consolidate_tstep_minimax_bf16_v1_shards.py`（新增） | `--check`（默认，只读，可在登录节点当进度表用）与 `--apply`（必须在 SCO 作业里以 root 运行）。检查每个分片：supervisor 日志以 OK 结尾且 `extract_fail=0 sync_fail=0 local_matching_sft_left=0`；分片 done-index 恰好等于 fp16 原索引在本分片场景范围内的层 20/25 单元，不缺不多，分片之间不重叠。场景顺序按提特征程序的规则重建（排序后的场景目录，数量必须是 682）。把 8 份分片索引与 canary 的场景 0 条目合并进正式索引；把结构文件复制到正式目录，t500 与 t1000 的两份必须逐字节相同，且与 fp16 原文件一致（.npy 逐字节，JSON 除 `feature_namespace` 外相同）。先写结构文件再写索引；原子写入、权限 0644；目标已存在且内容不同就拒绝；重复执行不改动任何文件；旧的正式索引备份在报告目录。 |
| `scripts/verify_tstep_minimax_bf16_v1_full.py`（新增） | 由 canary 验证器改写，全部 681 个场景：每个步骤的单元集合等于 fp16 原索引；**每个**文件的 sidecar（存储契约、bf16、finite_check、命名空间、协议、模型、模式、timestep、形状、生成来源）与 AOSS 上的载荷存在性；**每个**场景的结构文件与 fp16 原文件一致且等于 AOSS 副本；载荷按固定种子抽样（streaming 每层 16 个、target 每层 8 个）下载，检查 SHA-256、bf16、全部有限、形状、与 fp16 的数值偏差（容差同 canary）。报告写明范围：抽样以外文件的有限性依赖提特征程序的 finite_check 与同步循环的逐文件哈希。 |
| `scripts/submit_tstep_minimax_bf16_v1_consolidate.py`（新增） | 查询 32 个分片状态，在登录节点跑 `--check`；只有 32 个都 SUCCEEDED 且检查 PASS 时，`--submit` 才提交一个 1 卡作业：`--apply` 后接全量验证。记入提特征 ledger 的 `consolidation` 键，失败重试需 `--retry-reason`。默认干跑。 |
| `scripts/audit_tstep_canonical_alignment.py`（修改） | 新增 `NAMESPACE_PAIRS`：seeded-v1 对（原行为不变）与 MiniMax bf16 对（只允许 `minimax_h3`、层 20/25）。`audit` 与 `verify-artifact` 新增 `--streaming-namespace/--target-namespace`，默认仍是 seeded-v1；审计产物记录实际命名空间，验证时必须匹配。manifest 声明的特征根仍按 seeded-v1 检查（manifest 冻结未改）。 |
| `scripts/build_tstep_probe_matrix.py`（修改） | 接受上述两种命名空间对；bf16 对限定 `--vfms minimax_h3`、`--layers` ⊆ {20,25}；新增 `--tasks`（默认 9 个探针，保持原行为）；非默认命名空间的审计文件名带 `.seeded_v1_minimax_bf16_v1` 后缀，不会覆盖 seeded-v1 审计（其 sha256 记录在旧 groups.tsv 里）。 |
| `scripts/run_tstep_probe_group.sh.stage5`（暂存，尚未安装） | 组脚本新版本：接受 bf16 命名空间对（只允许 minimax_h3 层 20/25，且必须 `PROBE_FP16_PASSTHROUGH=0`，未设置时默认 0）；AOSS 根与 target 索引根按行里的命名空间推导；把命名空间传给 `verify-artifact`。seeded-v1 行为不变。**为何暂存**：30 个正在跑的 timestep B2 作业由 bash 执行旧的 `run_tstep_probe_group.sh`，bash 会在运行中继续从磁盘读脚本；原地覆盖会让这些作业在训练结束后读到错位的命令。挂载是 FUSE（quarkfs），跨节点“改名替换”的语义未知，所以等这些作业全部结束后再用改名方式安装：`mv scripts/run_tstep_probe_group.sh.stage5 scripts/run_tstep_probe_group.sh`。 |
| `scripts/submit_tstep_minimax_bf16_v1_retrain.py`（新增） | 阶段 5 提交脚本。合并作业的 apply 报告与全量验证都 PASS 之前只生成预览矩阵（写到 `launcher/preview_<UTC>/`，含审计），不会干跑或提交。PASS 之后：构建正式矩阵（审计写到 `artifacts/tstep_canonical_v2_seeded_v1_minimax_bf16_v1/audits/`），要求 16 组全部 `ready_full`、8 个探针、审计 verify 通过、与 09-09 MiniMax 矩阵的 manifest/探针/期望数一致、配置一致性检查 PASS，然后对每组跑 `DRY_RUN=1`。`--layout node8`（默认，16 个 8 卡作业）或 `--layout single`（128 个 1 卡作业）。`--submit` 必须带 `--approval "<谁、何时、原话>"`。按用户决定加了试点闸门：账本里已有作业时，再提交会先逐个读取已提交作业全部探针的最新 checkpoint（CPU），任一缺失或含非有限参数就拒绝，结果写到 `launcher/pilot_gate.json`。 |
| `scripts/check_tstep_minimax_bf16_v1_config_parity.py`（新增） | 见“结论”中的配置一致性检查。 |
| `tests/test_tstep_minimax_bf16_v1_stage5.py`（14 个）、`tests/test_consolidate_tstep_minimax_bf16_v1.py`（4 个）、`tests/test_verify_tstep_minimax_bf16_v1_full.py`（2 个） | 合成数据测试：bf16 审计通过且与命名空间绑定；seeded-v1 审计拒绝 bf16 根；bf16 对拒绝其他 VFM 与层 15；矩阵按 8 个探针生成 ready_full 行且审计文件名带后缀；组脚本干跑把 bf16 根传给全部 8 个探针并拒绝 `PROBE_FP16_PASSTHROUGH=1`；阶段 5 提交脚本两种布局都能出计划、干跑通过、无批准拒绝提交；试点闸门在缺 checkpoint 或含 NaN 时拒绝；合并脚本正常合并、幂等、拒绝 t500/t1000 结构文件不一致、拒绝缺单元与 supervisor 失败、拒绝与 fp16 不一致的 meta；全量验证器对有限 bf16 通过，对 NaN 载荷与错误 sidecar 失败。 |

- 修改前的三个脚本备份在 `/tmp/psv/backup_20261016_stage5/`（sha256：审计 `33bc3a94…`、矩阵 `efe9a4de…`、组脚本 `f3b82838…`）。
  审计与矩阵脚本用“写临时文件再改名”的方式替换；它们只在进程启动时读一次，正在跑的作业不受影响。
- 测试（本次实际执行通过）：暂存副本 `/tmp/psv/stage5_tree`（装好新组脚本）里相关的 28 个测试全部通过；
  主树全量 487 passed、3 skipped、1 deselected，552 s（之后新增的阶段 5 提交脚本测试在主树单独跑：10 passed、3 skipped）。
- 真实数据上的检查（本次实际执行通过）：
  - 场景 0（canary 已发布）的 bf16 严格审计，4 个 prefix 全部 `ok`（scope=canary，各 4 个样本，0 问题）；
    用 seeded-v1 默认命名空间审计同一批 bf16 文件则失败（5 类命名空间问题），说明绑定生效。
  - `consolidate --check`：682 个场景重建成功，canary 的场景 0 条目与 fp16 一致；正在跑的分片没有多出单元。
  - 阶段 5 预览：16 组全部 `wait_streaming`（每组 streaming 4/2721、target 1/681，只有场景 0），符合预期。

### 分片全部 SUCCEEDED 之后的步骤（尚未执行）

1. 登录节点：`python3 scripts/submit_tstep_minimax_bf16_v1_consolidate.py`（干跑，查看 32 个状态与 check 报告），
   PASS 后加 `--submit`。作业输出 `logs/tstep_minimax_bf16_v1/production_20261016/consolidation/r1/{apply_report,verify_full_report}.json`。
   这一步在已批准的阶段 2/3 范围内。
2. 确认 30 个 timestep B2 作业都已结束后，安装组脚本：`mv scripts/run_tstep_probe_group.sh.stage5 scripts/run_tstep_probe_group.sh`，
   再跑一次 `pytest tests/test_tstep_minimax_bf16_v1_stage5.py`（3 个 skipped 应变为通过）。
3. `python scripts/submit_tstep_minimax_bf16_v1_retrain.py`（默认 node8；可加 `--layout single`）：生成正式矩阵与 4 个严格审计、
   配置一致性检查、16 组（或 128 个）干跑，计划写到 `logs/tstep_probe_campaigns/tstep_probe_minimax_bf16_v1_r1/launcher/plan_<layout>.json`。
4. 阶段 5 已有条件批准（见“需要你决定的事项”第 4 项）：条件满足后先
   `... --submit --limit 1 --approval "<原话>"` 提交 1 组试点，首个 checkpoint 全部有限后再 `--submit` 其余 15 组。

### 阶段 5 提交方案（已有条件批准，尚未提交）

- 范围：MiniMax-H3，t500/t1000 × 层 20/25 × prefix 8/12/16/24 = 16 组，每组 8 个探针（streaming_depth、streaming_mask、
  view_consistency、ego_belief、ego_belief_v2、action_dynamics、path_integration、counterfactual），共 128 个 run，
  对应 `retrain_queue_20261016.csv` 里 `TSTEP_MINIMAX_L20_L25_FEATURE_OVERFLOW` 的 128 行。
- 设置：新 bf16 命名空间、`PROBE_FP16_PASSTHROUGH=0`；manifest、seed 42、配置、epoch（depth/mask 100，其余 50）、
  batch（depth/mask 1，其余 2）与旧 run 相同（已核对）；从头训练。run id `tstep_probe_minimax_bf16_v1_r1`，
  输出 `logs/inscene15k_streaming/runs/tstep_probe_seeded_v1/tstep_probe_minimax_bf16_v1_r1/tp_minimax_h3_t<T>_l<L>_p<P>_<探针>/`，不覆盖旧目录。
- 资源：计划估算约 810 GPU·h（中位）/ 1,425（p90）。node8 布局 16 个作业，每个以最慢的 mask 为界约 18–36 h；
  single 布局 128 个作业，排队更灵活，但每个作业各自缓存特征，AOSS 读取量约为 node8 的 8 倍。

## 复现命令（在服务器 `/mnt/umm/users/yinbaiqiao/probe_spatial` 下）

```bash
python3 scripts/freeze_tstep_minimax_bf16_v1_extractor.py --verify --release tstep_minimax_bf16_v1_extractor_r2
/mnt/umm/users/yinbaiqiao/.conda/envs/probe-spatial/bin/python -m pytest tests -p no:cacheprovider -q
/mnt/umm/users/yinbaiqiao/.conda/envs/probe-spatial/bin/python scripts/submit_b2_fp32_canonical_20261016.py
/mnt/umm/users/yinbaiqiao/.conda/envs/probe-spatial/bin/python scripts/submit_tstep_b2_fp32_retrain_20261016.py
/mnt/umm/users/yinbaiqiao/.conda/envs/probe-spatial/bin/python scripts/submit_tstep_minimax_bf16_v1_extract.py --canary
/mnt/umm/users/yinbaiqiao/.conda/envs/probe-spatial/bin/python scripts/submit_tstep_minimax_bf16_v1_extract.py --production
python3 scripts/build_retrain_submissions_20261016.py
# 10-16 跟进：最终 checkpoint 有限性（只追加新结束的 run）、合并进度/检查、合并作业、阶段 5 配置一致性与计划
/mnt/umm/users/yinbaiqiao/.conda/envs/probe-spatial/bin/python artifacts/release_audit_20261015/b2_fp32_retrain/b2_final_ckpt_finite_20261016.py artifacts/release_audit_20261015/b2_fp32_retrain/final_ckpt_finite_20261016.jsonl
python3 scripts/consolidate_tstep_minimax_bf16_v1_shards.py --report /tmp/psv/consol/check.json
/mnt/umm/users/yinbaiqiao/.conda/envs/probe-spatial/bin/python scripts/submit_tstep_minimax_bf16_v1_consolidate.py
/mnt/umm/users/yinbaiqiao/.conda/envs/probe-spatial/bin/python scripts/check_tstep_minimax_bf16_v1_config_parity.py --report artifacts/release_audit_20261015/stage5_config_parity/config_parity_20261016.json
/mnt/umm/users/yinbaiqiao/.conda/envs/probe-spatial/bin/python scripts/submit_tstep_minimax_bf16_v1_retrain.py
```

不带 `--submit` 时提交脚本只做校验并打印计划。作业状态：

```bash
sco acp jobs describe <job_id> --workspace-name=aigc -o json
```

逐 run 的作业号表：`metadata/release_inventory/retrain_submissions_20261016.csv`（176 行）。

## 需要你决定的事项（推荐项放第一）

1. **代码提交**：已决定并完成。用户选择 (a)，2026-10-16 提交到新分支 `minimax-bf16-reextract-20261016`，
   commit `41cce9c3185015e7bc11a6b24db93bb6d229195b`（25 个文件，+9,026/−185 行，未 push）。
   提交里的 12 个提特征文件与未改动的 `temporal_sampling.py` 共 13 个文件，与 r2 冻结运行时的 sha256 全部一致；
   在只含该 commit 的干净 worktree 中，`tests/test_minimax_bf16_storage_contract.py` 与
   `tests/test_tstep_sync_loop.py` 共 35 个测试通过（本次实际执行通过）。工作区其余未提交改动不在其中，
   包括本文件与其它 docs、`tests/conftest.py`、`pytest.ini`、训练代码与 10-15 审计的修复。
2. **生产分片的同步校验模式**：(a) 保持 `SYNC_TRUST_EXTRACT=0`，逐文件哈希与契约检查（推荐，
   canary 正在验证这条路径）；(b) 改为 1，只按文件存在与大小清点，更快但不复查 dtype。
3. **阶段 2/3 结束后的合并、审计与阶段 5**：(a) 分片全部 PASS 后由我编写合并与审计脚本，
   审计通过再请你批准阶段 5（推荐）；(b) 合并审计后直接提交阶段 5 的 128 个重训。
   10-16 15:10 UTC 起按 (a) 执行：脚本已写好并测试，等分片结束后运行；阶段 5 仍待批准（第 4 项）。
4. **阶段 5 的提交方式**：已决定（有条件批准）。用户 2026-10-16 约 15:22 UTC 选择 (a)，原话：
   “Approved for later: once consolidation and full verification PASS and the 16-group dry run is clean,
   submit stage 5 as a 1-group node8 pilot (--limit 1), then the other 15 groups after its first checkpoint is finite.”
   执行条件（全部满足才提交）：合并作业的 apply 报告与全量验证报告都是 PASS；
   `submit_tstep_minimax_bf16_v1_retrain.py` 的 16 组干跑、审计验证与配置一致性检查没有任何错误。
   先 `--limit 1 --approval "<上述原话>"` 提交 1 组 node8 试点；试点的首个 checkpoint（8 个探针各自的第一个）
   全部参数有限后，再提交其余 15 组。任一条件不满足就停下报告，不提交。
   当时的备选项：(b) 直接提交 16 个 8 卡作业；(c) 改为 128 个 1 卡作业（`--layout single`）。
5. **新版组脚本的安装时机**：(a) 等 30 个 timestep B2 作业全部结束后用 `mv` 安装（推荐）；
   (b) 现在安装，正在跑的作业可能在训练结束后读到错位的命令而被标为失败。
