# probing_spatial 当前风险处理与后续实验设计（2026-08-17）

本文件回答 5 个 follow-up 问题，并记录本次新增的审计实现。

新增脚本：

- `scripts/audit_current_risks.py`

新增审计输出目录：

- `logs/current_audit_20260817/README.md`
- `logs/current_audit_20260817/c_shortcut_audit.csv`
- `logs/current_audit_20260817/c_shortcut_summary_by_model.csv`
- `logs/current_audit_20260817/c_shortcut_audit.md`
- `logs/current_audit_20260817/b2_canonical_angle.csv`
- `logs/current_audit_20260817/b2_canonical_angle_formal_only.csv`
- `logs/current_audit_20260817/b2_best_angle_formal_by_vfm_prefix.csv`
- `logs/current_audit_20260817/b2_canonical_angle.md`
- `logs/current_audit_20260817/wan_cog_layer_formalization_summary.csv`
- `logs/current_audit_20260817/wan_cog_formal_best_by_probe_vfm_prefix.csv`
- `logs/current_audit_20260817/wan_cog_remaining_nonformal_jobs.tsv`
- `logs/current_audit_20260817/wan_cog_no_checkpoint_jobs.tsv`
- `logs/current_audit_20260817/wan_cog_layer_formalization.md`
- `logs/current_audit_20260817/new_backend_artifact_audit.csv`
- `logs/current_audit_20260817/new_backend_artifact_audit.md`
- `logs/current_audit_20260817/dirty_artifact_inventory.txt`
- `logs/current_audit_20260817/dirty_artifact_inventory.md`
- `logs/current_audit_20260817/experiment_design_notes.md`

## 1. 问题 1：C 系列高 R@1 是否真代表 spatial/action reasoning？

可以先做一轮基于现有 artifacts 的验证，已经实现为 `c_shortcut_audit.csv`。

审计定义：

- probe 指标：`global_R@1`
- shortcut/control baseline：`no_action_global_R@1`、`shuffled_action_global_R@1`、`last_observation_global_R@1`
- 只有 probe 的 `global_R@1` 严格超过三类 baseline，才标为 `action_beats_all_baselines=True`
- 只要任意 baseline 追平或超过 probe，就标为 `shortcut_risk_R@1=True`

当前结果：

- 共审计 96 行：C1/C2/C3 各 32 行
- C1 action_dynamics 有完整 R@1 shortcut baseline
- 只有 6 行超过所有 R@1 baseline
- 有 26 行存在 R@1 shortcut risk
- 23 行被 `last_observation/copy` baseline 追平或打败
- 9 行被 `no_action` 或 `shuffled_action` 追平或打败

因此，当前 C1 的结论应改成：

> C1 的部分 row，尤其是 Wan p12、CogVideoX p12/p16/p24、F3R p12、VideoMAE p24，提供了一些超越 shortcut baseline 的证据；但整体上高 R@1 不能直接解释为 action/spatial reasoning，因为多数模型/前缀会被 last-observation、no-action 或 shuffled-action control 追平或打败。

特别要小心 Qwen2.5-VL、DINO、VJEPA2、VideoMAE 的一些高 R@1：它们在 C1 中经常被 `last_observation` baseline 打败，说明 copy/smoothness 或 feature retrieval shortcut 很强。

下一步更强实验：

1. 固定 C 的 best config，在 held-out test split 上重跑 probe 和三类 shortcut baseline。
2. 增加 action-token ablation：zero action、trajectory 内 shuffle、跨 scene shuffle、reverse action order。
3. 增加 copy/smoothness baseline：last observation、linear feature extrapolation、prefix feature nearest-neighbor、scene-level retrieval。
4. 按 horizon 画曲线。copy/smoothness shortcut 应该在短 horizon 更强，真实 action reasoning 应该在长 horizon 和大位移时更有优势。
5. 按 motion magnitude / viewpoint change 分桶。真实 action reasoning 应该主要在静态检索不够用的桶里提升。

## 2. 问题 3：B2 无 GT spatial mask 的本质是什么？

本质是 identifiability 问题：没有 ground-truth spatial mask 时，B2 的 object-query localization 可能同时测到多种东西：

- 真实的 object-query 到 3D hidden location 绑定
- object/category prior，例如 chair/table 通常在某些区域
- scene/layout prior，例如房间类型和相机轨迹泄漏位置
- query wording bias
- visible-object bias
- dataset placement bias

所以 B2 当前要回答的不是简单的“有没有 spatial reasoning”，而是：

> 在没有 GT spatial mask 监督的条件下，object-query localization 的提升有多少来自真实 query-object-location binding，又有多少来自 category/query/scene/dataset prior？

可以设计以下实验讨论：

1. Query shuffle within scene：固定 scene/video，交换 object query。如果 probe 仍然高，说明 scene/category prior 很强；如果显著下降，才支持 query grounding。
2. Query-only baseline：只给 query/category，不给 visual feature。若分数高，说明 dataset/query prior 足够强。
3. Feature-only baseline：给 visual feature，mask 掉 query。若分数高，说明 scene/layout prior 足够强。
4. Category/room-balanced split：让 train/val/test 的 category 和 room/layout 分布更均衡，降低位置先验泄漏。
5. Same-category hard negatives：同类别物体放在不同方位/距离，避免 category alone 就能猜中。
6. Position-prior baseline：按 category、room type、prefix length 预测最常见 bearing/distance，不用视觉特征。
7. 小规模 synthetic / controlled subset：人为 counterbalance object location，用来做 bias stress test。

## 3. 为什么当前说是 validation，而不是 test？

你的理解有一半是对的：训练结束后拿权重做 evaluation，这确实是 evaluation/test-time procedure。但在 ML 报告里，“validation 还是 test”取决于 evaluation 用的是哪个 split，以及这个 split 有没有参与选择过程。

当前仓库证据显示主要结果来自 validation：

- B2 配置中 validation dataset 明确是 `split='val'`，例如 `configs/experiment/inscene15k_streaming/ego_belief_v2_cogvideox_v1.yaml`
- C eval runner 使用 `+eval_split=val` 和 `test=false`，见 `scripts/run_c_eval_slots.py`
- 多数汇总文件名是 `eval/val_summary.json`

因此目前更准确的说法是：

> 这些是训练后 evaluation 得到的 validation metrics，不是 frozen held-out test metrics。

建议改进成 frozen test protocol：

1. 用 train 训练。
2. 用 val 选择 model family / probe / layer / prefix / checkpoint / seed reporting rule。
3. 冻结所有选择，包括 best-layer 规则、prefix、checkpoint 选择规则、metric 主指标。
4. 只在冻结后，对 `split='test'` 跑一次 final evaluation。
5. 报告 test 均值和多 seed 方差；val 只作为 selection/ablation evidence。

如果需要，我建议新增一个 `scripts/run_frozen_test_protocol.py` 或扩展现有 eval runner，使它读取 frozen manifest，强制 `eval_split=test`，并输出 `test_summary.json` / `test_metrics.csv`，避免 val/test 混淆。

## 4. Wan/Cog layer sweep provisional 的问题是什么？能否彻底解决？

当前 artifact 里确实还混着 formal 和 provisional，不是叙述问题。

输入表：

- `logs/probing_jobs_20260725_ab/recovery_20260730/perf_analysis/evaluation_results_all_probes.tsv`

审计结果：

- 总 rows：213
- formal rows：51
- provisional rows：162
- class counts：`FOUND_EXACT=10`、`FOUND_EQUIVALENT=1`、`REPAIRED_BEST=40`、`FOUND_NEARBY_ONLY=5`、`TRULY_MISSING=157`
- B2 rows：56
- B2 formal-only rows：48
- B2 provisional rows：8

这说明：

- B2 基本可被 formal-only 表解决，但仍有 8 个 provisional rows
- ego_belief、streaming_depth、view_consistency 大多数仍是 provisional / missing class
- 因此 Wan/Cog layer sweep 不能整体写成 formal best

本次已经做的“彻底解决第一步”：

- `b2_canonical_angle.csv`：B2 全量 canonical angle 表
- `b2_canonical_angle_formal_only.csv`：B2 formal-only 表
- `b2_best_angle_formal_by_vfm_prefix.csv`：B2 formal-only best-by-prefix
- `wan_cog_formal_best_by_probe_vfm_prefix.csv`：所有 probe 的 formal-only best 表
- `wan_cog_remaining_nonformal_jobs.tsv`：仍不能 formal claim 的 rows
- `wan_cog_layer_formalization_summary.csv`：按 probe/vfm/prefix 的 formalization 状态

剩余要真正完成的工作：

1. 对 `wan_cog_remaining_nonformal_jobs.tsv` 中 rows 定位 exact/best checkpoint。
2. 能找到 checkpoint 的，重跑同一 eval protocol 并更新 all-probes table。
3. 找不到 checkpoint 的，明确标为 unrecoverable，不能进入 formal-best conclusion。
4. 所有 formal rows 只从 `*_formal_only.csv` 派生，不再从混合 TSV 直接写结论。

## 5. dirty workspace 与 LingBot/MiniMax-H3 后端 metrics 如何解决？

### Dirty workspace

本次生成了：

- `dirty_artifact_inventory.txt`
- `dirty_artifact_inventory.md`

当前 `git status --short` 显示：

- dirty entries：97
- modified/staged：22
- untracked：75

建议处理策略：

1. 小文件且复现实验必需的内容应进入 git：configs、scripts、manifests、docs、small CSV/Markdown summaries。
2. 大型 logs/checkpoints/features 不建议直接进 git；应保存在外部 artifact store，并在 git 中保留 manifest、路径、hash、生成脚本和版本说明。
3. 当前 `docs/`、`manifests/`、LingBot/MiniMax config/scripts 都未跟踪，复现前应该至少 commit 或 archive。
4. 不要在未确认的情况下清理 dirty workspace，因为其中有大量已有改动不是本次审计创建的。

### LingBot / MiniMax-H3

本次生成了：

- `new_backend_artifact_audit.csv`
- `new_backend_artifact_audit.md`

当前结论：

- MiniMax-H3 campaign 有 job manifests 和 monitor artifacts，但没有稳定 final metrics 汇总。
- LingBot/probe-fill campaign 有 job manifests 和 monitor artifacts，但 monitor 显示 max-hours reached 或 running，且没有 canonical final metrics。
- 这些后端现在只能写成“已有实现和 job artifacts”，不能写成最终性能结论。

要变成可报告结果，需要补齐以下协议：

1. 每个 backend/campaign 固定一个 `all.tsv` manifest。
2. 每个 job 完成后生成统一 eval summary。
3. 汇总脚本输出 canonical `metrics.csv` 和 `metrics.md`。
4. 对失败/超时 job 输出 failure manifest。
5. 报告只引用 canonical metrics，不引用 monitor log 或散落 job dirs。

本次没有启动新的训练或 GPU job；只做了本地 artifact 审计和 canonical 表生成。若要把 LingBot/MiniMax-H3 推到 final reportable state，下一步应先 repair/relaunch 失败或 running jobs，然后运行同一个 aggregate metrics 脚本生成正式结果表。
