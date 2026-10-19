# probing_spatial 全局主线重启报告

生成日期：2026-08-28 UTC。报告文件名按请求保留 `20260827`。本轮是严格的 metadata/result analysis：没有启动训练、没有抽 feature、没有提交作业、没有修改 `experiments/registry.csv` 或 `results/canonical/*.csv`。

## 执行结论

全局主线可以恢复，但必须先固定两个统计粒度：

1. **428/428 是 eval job-level campaign 完整性。**最终 promotion audit 记录 428 个 valid summaries、428 个 unique job keys、0 duplicates；Wan/p8/action_dynamics 的 6 个缺失 layer jobs 正是最后补齐的部分。
2. **当前 canonical universe 是 761 个 atomic primary-metric experiments。**`registry.csv` 共 955 个唯一 experiment IDs，其中 761 COMPLETE、194 INCOMPLETE；`all_results.csv` 恰好包含全部 761 个 COMPLETE IDs，字段逐项一致。

因此不能把“428 个 eval jobs”误写成“canonical 只有 428 行”。一个 eval job 可产生多个 atomic probe outputs；C2/C3 family tables 又展开多个 horizon/secondary metrics。

## 1. Source of truth 与精确完整性

| source | rows | unique experiment IDs | sha256 |
| --- | ---: | ---: | --- |
| `experiments/registry.csv` | 955 | 955 | `efc533bbe6286090d6c111c9d272ffb28cf7477b937f799f312eedf2a944431e` |
| `results/canonical/all_results.csv` | 761 | 761 | `2a04a5e1f0165f1d7a06e7edcff38d956b845910ad95af3f7bbede3704133b17` |
| `results/canonical/a1_results.csv` | 265 | 265 | `f7777676eafbc77dc8a8856e86a0901c98d6b5b3bdaddb568a975ef1c9f94d86` |
| `results/canonical/a2_results.csv` | 82 | 82 | `2613650bc7dcaa7a5bb5292c1953b9742657fbcb7967da3016b8c84c9f6ffa5f` |
| `results/canonical/b2_results.csv` | 113 | 113 | `0d4941709af83ba72820e4c463b96775d240096b44fca14bd4c5243e2dfa5f20` |
| `results/canonical/c1_results.csv` | 96 | 96 | `b88a37e239d150d297a4e97838d93888d5b6d3d0780df510e65b7f916b2ade1a` |
| `results/canonical/c2_results.csv` | 368 | 92 | `8c96a38500420af11427ed506bc92d4e1fc9d6b636bc34f0c4baf2d279fd9524` |
| `results/canonical/c3_results.csv` | 714 | 102 | `6cb2bec11abecb7cdf7549f14db84d58345e2bb5e9a0172f9435c80e5e2b7c5d` |

完整性核验：

- registry experiment IDs：955/955 unique。
- canonical `all_results.csv`：761/761 unique；与 registry COMPLETE ID set 完全相同；family/probe/model/layer/prefix/metric/value/raw path mismatch=0。
- exact duplicate canonical rows=0；duplicate experiment IDs=0。
- 若把 `model_variant` 纳入配置 identity，761 rows 全唯一。
- 若忽略 `model_variant`，存在 **21 个 scientific-config collision groups**（42 rows、21 excess）。主要是 VideoMAE/V-JEPA2 A1/A2 的 default vs 20260817 eval，以及 Wan B1/B2 的 historical vs gap-fill。它们数值可显著不同，必须按 campaign 分开，不能平均或任选较优值。
- `scripts/check_project_consistency.py` PASS；保留两个既有 VideoMAE p12 missing-manifest path warnings。

family 文件的展开口径：A1=265 rows、A2=82、B2=113、C1=96；C2=368 rows/92 IDs（每 ID: global + h1/h2/h4）、C3=714 rows/102 IDs（每 ID: 4 global metrics + h1/h2/h4）。B1 的 11 个 primary rows 只存在于 `all_results.csv`，没有单独 `b1_results.csv`。

## 2. 完整分析矩阵与覆盖

`results/analysis/global_restart/experiment_matrix.csv` 有 761 行，每行对应一个 canonical atomic primary experiment，补全了以下分析维度：task/family、model、model/objective family、layer 及 registered-grid position、streaming prefix、config-declared diffusion step、streaming/target/prediction output role、condition、metric direction、campaign cohort、collision flag 与 provenance。

需要注意：

- canonical registry 本身没有独立 timestep/feature-type/condition 列；矩阵中的 timestep 来自对应 config，task contract 来自 probe/data flow。
- 生成模型 canonical step 为 Wan/CogVideoX/Aether=`749`、MiniMax-H3/LingBot=`250`；非生成 backbone 记为 `NA_non_diffusion`。
- A/B input 是 `streaming`；C input 是 `streaming`、监督 target 是 `target_isolated`，probe output 才是 feature prediction。**target feature 不是 prediction feature。**
- 所有 canonical rows 均为 seed 42、val split。

### family-level 覆盖

| family | primary rows | models | layers | prefixes | primary metric(s) | variants |
| --- | ---: | ---: | ---: | --- | --- | --- |
| A1 | 265 | 10 | 18 | 8,12,16,24 | Auc_30, mean_loss_depth, mean_loss_identity | default_streaming, eval_20260817 |
| A2 | 82 | 9 | 13 | 8,12,16,24 | balanced_acc | default_streaming, eval_20260817 |
| B1 | 11 | 2 | 9 | 16,24 | mean_az_err_deg | eval_20260817, wan_cog_layer_sweep |
| B2 | 113 | 6 | 15 | 8,12,16,24 | mean_ang_err_deg | eval_20260817, wan_cog_layer_sweep |
| C1 | 96 | 10 | 17 | 8,12,16,24 | global_R@1 | c_retrain_with_extra, eval_20260817 |
| C2 | 92 | 10 | 16 | 8,12,16,24 | global_R@1 | c_retrain_with_extra, eval_20260817 |
| C3 | 102 | 10 | 17 | 8,12,16,24 | intervention_validity | c_retrain_with_extra, eval_20260817 |

### model × family primary-row 矩阵

| model | A1 | A2 | B1 | B2 | C1 | C2 | C3 | total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| aether | 12 | 4 | 0 | 0 | 4 | 4 | 4 | 28 |
| cogvideox | 30 | 4 | 7 | 26 | 4 | 4 | 4 | 79 |
| dino | 12 | 4 | 0 | 0 | 4 | 4 | 4 | 28 |
| f3r | 12 | 4 | 0 | 0 | 4 | 4 | 4 | 28 |
| lingbot | 26 | 0 | 0 | 9 | 10 | 10 | 10 | 65 |
| minimax_h3 | 29 | 15 | 0 | 20 | 20 | 20 | 20 | 124 |
| qwen2_5_vl | 12 | 4 | 0 | 0 | 4 | 4 | 4 | 28 |
| videomae | 56 | 19 | 0 | 15 | 16 | 16 | 16 | 138 |
| vjepa2 | 47 | 24 | 0 | 20 | 20 | 20 | 20 | 151 |
| wan | 29 | 4 | 4 | 23 | 10 | 6 | 16 | 92 |

### comparable core 与扩展 coverage

- 可严格横向比较的 default core：8 models × 4 prefixes × 7 primary task/metrics（A1 depth/camera/mask、A2、C1、C2、C3）= **224/224 rows**。
- B2 不在这个完整 grid 内。registry 中 32 个 default B2 rows 仍为 INCOMPLETE，因为旧总表只机器可读地封装了 top1；独立核验显示其中 31/32 个现存 `val_summary.json` 已包含 `mean_ang_err_deg`，F3R p24 summary 缺失。应先 staging/audit，再补 1 个 eval-only，而不是重训 32 个。
- Layer expansion 覆盖不对称：LingBot 主要只有 p16/p24；MiniMax A2 无 p24；Wan/Cog historical layer sweep 有大量 provisional rows 被正确排除；不能用“每模型 best cell”替代公平 matrix。

## 3. A1、A2、B2、C1、C2、C3 到底测什么

| family | 实际测量对象 | canonical primary metric | 当前覆盖边界 |
| --- | --- | --- | --- |
| A1 | streaming features 的 depth、camera pose 与 identity/mask 可读性 | depth loss ↓；AUC-30 ↑；identity loss ↓ | 10 models，但扩展 layer grid 不对称 |
| A2 | 两段 view 是否共享视域 | balanced accuracy ↑ | 9 models；LingBot 缺失，MiniMax 缺 p24 |
| B1 | legacy masked/query hidden-object bearing | mean azimuth error ↓ | 仅 Wan/CogVideoX 11 rows；supplemental，不作为主线 |
| B2 | object query + prefix patch tokens 对 hidden-object 方位/距离的读出 | mean angular error ↓ | canonical 仅 6 models；standard construct 非 strict-zero-hidden |
| C1 | context + 单步相机 action 预测 isolated target feature 的检索 | global R@1 ↑ | 10 models；raw metric 必须和 no-action/shuffle/copy 一起解释 |
| C2 | action sequence 的 waypoint target-feature rollout/path integration | global R@1 ↑；expanded file 另有 h1/h2/h4 pose error | 10 models；primary table 每 experiment 1 行，expanded file 每 experiment 4 行 |
| C3 | alternative action 对 counterfactual target prediction 的 intervention consistency | intervention validity ↑ | 10 models；expanded file 每 experiment 7 行 |


更严格的解释边界：A1/A2 是 frozen-feature readout，不等于显式 3D world model；B2 可能受 query/category/scene/visibility prior；C1/C2/C3 是 target-feature prediction/retrieval，不自动等于物理 dynamics。C3 的 intervention-validity 和 C1/C2 R@1 也不在同一量纲。

## 4. 历史计划搜索结果与完成度

已对 `planning`、`INTACT`、`diffusion step`、`shuffle`、`without condition`、`prediction feature`、`streaming prefix`、`layer sweep`、`analysis cube`、`probing_spatial`、`paper`、`canonical` 做 repository-wide 搜索，并查阅 canonical build/update scripts、历史 reports、paper-validity、probe-validity、SE(3) MVP 与 tstep tooling。

- `INTACT`：仓库 0 个 literal hit。
- `prediction feature` / `analysis cube`：0 个明确计划或产物 hit。
- `planning`：只有 Aether pipeline 内部 task mode 和无关的 capacity-planning 文本；没有 probing task/config/metric/manifest。
- diffusion/tstep：有完整 extraction/probe orchestration code 与当前 feature indexes，但还没有 tstep probe run directory 或 canonical metrics。
- shuffle：有 legacy A3、paper-validity legacy controls、selected p12 B2/C1 sample audit。
- without-condition：有 legacy eval intervention、Wan eval-null-action（明确不是 trained no-condition）和 non-canonical DINO SE(3) MVP。
- streaming prefix/layer/canonical/paper：历史与当前产物充分，但部分文档（例如早期 new-backend status）已被 20260817 canonical gap-fill 超越，不能再作为结果 source of truth。

`planned_vs_completed.csv` 共 16 项；状态分布 `{'completed': 1, 'completed_now': 1, 'completed_core_partial_extended': 1, 'partially_completed': 1, 'features_partial_no_probe': 1, 'partial_supplemental': 1, 'partial_noncanonical': 1, 'not_started_definition_missing': 1, 'not_started_features_partial': 1, 'not_started_construct_undefined': 1, 'not_started_project_level': 1, 'partially_completed_stale_scope': 1, 'partial_selected_complete': 1, 'partial_selected_audit': 1, 'not_started': 1, 'completed_with_caveats': 1}`。按需求分类分布 `{'existing_results': 6, 'existing_features_need_probe_eval': 3, 'construct_audit_then_eval_or_extract': 1, 'requires_feature_extraction': 1, 'mixed_feature_and_probe_work': 1, 'construct_validity_first': 3, 'existing_results_then_small_engineering': 1}`。

### 四类缺口

1. **已有数据即可完成：**当前 global plots、prefix robustness、campaign-separated layer curves、task correlation、default B2 的 31 个 raw angle rows staging audit、Wan case-study integration。
2. **已有 feature 但需 probe/eval：**Wan t499/t999 全 feature grid；MiniMax t500/t1000 streaming-only grid；DINO matched no-condition controls；F3R p24 B2 eval-only；frozen headline test/multi-seed。
3. **必须重新抽 feature：**prediction-feature A pilot；CogVideoX incomplete tstep cells；如果问题是“backbone 是否对 frame order 敏感”，则需 pre-backbone shuffled-frame features。
4. **先做 construct-validity audit：**INTACT-style target-conditioned planning/action output；prediction feature 的操作化；strict B2；任何把 C raw retrieval 解释成 dynamics 的扩展。

## 5. 当前最可信的全局发现

完整证据、反例与 confound 写在 `robust_findings.md`。摘要如下：

1. 表征呈能力分工：F3R 是 direct geometry specialist，DINO/Qwen 的 C raw retrieval 强，但两者不是同一能力轴。
2. 长 prefix 经常退化：56 个 core p24-vs-p8 配对只有 13 个改善，C2 是 0/8。
3. universal middle-layer hypothesis 不成立：99 个 campaign-internal complete layer groups 中 intermediate/last/first = 41/32/26。
4. 当前 ViGen layers/steps 没有通用空间优势；结论受 timestep coverage 限制。
5. C1 raw R@1 常被 persistence 解释；DINO/Qwen 尽管 raw 高，却在四个 prefix 都输给 copy。
6. B2 selected audit 更支持 query/history-content retrieval，不支持强 ordered-memory claim。
7. C1/C2 model prefix-AUC 排序高度相关（Spearman 0.952），可能共享 target-feature readability 因子。
8. diffusion-step 与 prediction-vs-context 目前均未被 canonical 数据回答。

所有这些结论都受 seed42/val-only 限制；当前不能声称 held-out、多 seed generalization。

## 6. Diffusion-step 与 analysis cube 的真实状态

当前 completion indexes 的 exact-manifest 重算结果：

- `wan_t499`: {'ready_full': 12}; streaming found/group range 2721–2723; target 681–681/681; target_indices 681/681.
- `wan_t999`: {'ready_full': 12}; streaming found/group range 2721–2723; target 681–681/681; target_indices 681/681.
- `cogvideox_t499`: {'wait_streaming': 12}; streaming found/group range 1866–1875; target 0–0/681; target_indices 0/681.
- `cogvideox_t999`: {'wait_streaming': 12}; streaming found/group range 1872–1881; target 0–0/681; target_indices 0/681.
- `minimax_h3_t500`: {'ready_streaming_only': 12}; streaming found/group range 2721–2723; target 113–113/681; target_indices 113/681.
- `minimax_h3_t1000`: {'ready_streaming_only': 12}; streaming found/group range 2721–2723; target 0–0/681; target_indices 113/681.

这说明最小可执行路径不是继续抽 Wan feature，而是直接用已有 Wan t499/t999 做 A1/A2 小 pilot，并与 canonical t749 比较；不应把 tstep 工作重新变成 Wan action_dynamics。MiniMax 可作为 streaming-task replication。CogVideoX 只有 pilot 显示 timestep effect 后才值得补 extraction。

完整 `layer × step × prefix` cube 目前不能分析：只有 feature cells，没有 probe metrics。先做 layer20、p8/p24 的 reduced pilot，过 gate 后再加入 early/middle/late layers。

## 7. 原计划中真正缺失的实验

- target-conditioned planning/action output：task definition、INTACT mapping、data/metric/baseline 全缺。
- prediction feature A-series：feature definition、extraction mode 与任一 result 全缺。
- fair trained without-condition/visual-only/action-only C controls：只有 isolated examples/legacy interventions，没有跨模型 canonical grid。
- current streaming encoder-level shuffled-frame baseline：legacy A3 不能替代。
- diffusion-step probes：features 部分或全部存在，但 probe/eval 全缺。
- held-out test + multi-seed frozen confirmation：全缺。
- strict-zero-hidden B2 与完整 simple-prior baselines：selected audit 已明确当前 standard B2 不足。
- current 10-model cross-model plot/demo pack：旧 core figures 有，当前 full canonical 版本缺。

## 8. 下一批最小优先任务

详见 `next_experiment_queue.csv`。P0 顺序是：

1. 用现有 761 rows 生成 current plot pack；0 training。
2. staging/audit 31 个 existing B2 angle summaries，仅对 F3R p24 做 eval-only；不重训整套。
3. Wan tstep 的非 action pilot：t499/t999、layer20、p8/p24、A1+A2，共 8 个新 probe training jobs；t749 复用 canonical。
4. 先完成 planning/INTACT task construct contract；0 training。

之后才是一个 non-Wan matched no-condition C pilot、MiniMax timestep replication、prediction-feature minimal extraction，以及 frozen headline test。

## 9. 不值得继续扩 sweep 的任务

- **Wan p8 action-validity：停止。**Phase-1 gate 已给出明确答案，不进入 capacity-matched Phase 2，不再扩 layers/prefixes。
- 不补齐 162 个 historical provisional Wan/Cog rows，只在某个 frozen headline cell 不可替代时恢复。
- 不做 legacy non-streaming A3 full sweep。
- 不做未过 construct gate 的 full planning 或 prediction-feature Cartesian sweep。
- 不做 raw C1 best-layer/best-prefix sweep；任何扩展先看 gain-over-copy 与 matched controls。
- 不做完整 tstep cube；先过 reduced pilot。
- B1 不作为主线继续扩，因为其 mask/visibility design 只适合 upper-bound/diagnostic。

## 10. Wan action-validity 应怎样嵌入全局故事

把它放在方法论/limitations 中，作为一个清晰的 **case study**：

> Raw probing performance 不等于目标构念。Wan p8 的 correct action 确实显著优于 shuffled/wrong action，说明存在 weak action signal；但 layer 10/15 没有击败 last-state copy，所以原始中间层峰值不能直接解释成 action-conditioned dynamics。

这个 case study 支持三条全局规范：先报告 shortcut-normalized gain；中间层 peak 不能单独支撑机制解释；construct audit 可以在不修改 canonical 的情况下改变结论强度。它不应变成项目标题、主要 model ranking 或新的 Wan sweep 主线。

## 产物

- `reports/probing_spatial_global_restart_20260827.md`
- `results/analysis/global_restart/experiment_matrix.csv`
- `results/analysis/global_restart/planned_vs_completed.csv`
- `results/analysis/global_restart/robust_findings.md`
- `results/analysis/global_restart/next_experiment_queue.csv`

生成后脚本再次核验所有 source-of-truth SHA256；任何变化都会 fail closed。
