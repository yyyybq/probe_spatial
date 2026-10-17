# Other analysis 实验与发布结果清单 — 2026-10-11

逐条清单：[metadata/release_inventory/analyses.csv](../metadata/release_inventory/analyses.csv)。这次只读取本地 protocol、训练/评估 receipt、结果表和 checkpoint 文件元数据；没有训练、评估、特征抽取、scheduler 查询或联网，也没有反序列化/重新散列 checkpoint。

**六个既有结果包可以按实际 val/dev 范围作为发布候选。V-JEPA2 D2 的三次训练与 development evaluation 也确已完成，缺的是三种子汇总及统一导出。** 独立盲测是更强结论需要的新工作，不是诚实发布现有 val/dev 结果的前置条件。

## 计数与证据口径

CSV 共 **630 条**：617 条 `entry_level=run`、7 个 package、6 个未完成 scope。`run` 包含 trained configuration、训练 stage bundle 和 evaluation-only cell，**不是 617 次新增训练**。其中 555 条有 `training_status=COMPLETE`，62 条为 `NOT_APPLICABLE`；555 也不能直接称为独立训练任务数，因为 D-series stage 内含多个 head，而已有 predictor 的干预评估不增加训练数。

617 条 run 的发布标记为 **550 YES_WITH_SCOPE、47 NO、20 UNKNOWN**；7 个 package 为 **6 YES_WITH_SCOPE、1 NO**。package 与其子 run 重叠，禁止相加。`NO` 包括等待汇总的已完成 D2，不能反推为训练失败；20 个 `UNKNOWN` 是 SE3 V-JEPA2 L11 源文件的当前读取权限限制，不能反推为 frozen package 无效。

- `training_status` 区分该条目是否包含本次分析的训练。C1/B2 validity 和 C3-from-C1 使用历史训练权重，标记 `NOT_APPLICABLE`，其上游 checkpoint 状态单列。
- `correctness_status=VERIFIED` 指可读 receipt、协议身份与已有审计支持声明的范围，不表示本次重新运行了数值实验或重算了所有 bootstrap。
- `observed_at` 是本次本地查验时间；`source_observed_at` 仅记录文件正文的 finished/frozen 时间。无时间字段写 `NOT_RECORDED`，不从目录名或文件 mtime 推断。
- `metric_rows` 是关联证据行数，可能是原始样本行或聚合行，**不是训练次数**。checkpoint 只做存在性/非空检查；`checkpoint_status` 明示已有 SHA 是 receipt 记录还是本次新验证。

## 可发布的 val/dev 范围与逐实验矩阵

| 结果包 | 清单实际展开单位 | 模型、层、prefix、训练 seed | 现状与边界 |
|---|---|---|---|
| Independent P8 actionability/algebra | 180 trained configurations：144 scorer、36 dynamics | DINO project-final L0、F3R returned-index L24、V-JEPA2 L23、Wan L20/scheduler-index749；P8；41/42/43 | 四模型 receipt 均 `COMPLETE_VALID`，每个配置有训练曲线和对应样本结果；12 个 full-dynamics ckpt 已留存。结果可发，pilot 接触过 validation |
| Continuous-action relational V2 | 100 variant/seed checkpoints | DINO L0、F3R L24、V-JEPA2 L23、**Wan L29/t749**；P8 + isolated h4；41–45 | 100 个 matrix/receipt SHA 身份一致、文件非空、每配置 256 val 记录；是 supervised first-action readout |
| SE3 Wan/V-JEPA2 final r3 | 60 layer/seed bundles 内的 240 trained heads，加 5 个 synthetic positive-control heads | Wan L0/5/10/15/20/25/29；V-JEPA2 L0/5/11/17/23；P12；42–46 | final freeze/statistical audit PASS，21 个 generated-file SHA 本次全部匹配；69 个 pilot scenes 上新 frame triples 的 development evidence |
| Probe-validity paper | 12 evaluation cells：8 C1 + 4 B2；**68 是 metric rows** | V-JEPA2/VideoMAE L23，Wan/CogVideoX L20；C1 P12/P16；B2 P12；历史 probe seed42 | 本分析不新增训练；P16 C1 四项 exact-checkpoint repair 有 PASS receipt；P12 为 accepted paper source mapping |
| D-series Wan D1/D2 | 9 stage bundles | Wan L20/t749，P8，h2/h4 joint，seed0/1/2；D1 spatial/pooled，D2 pooled | 9 个 receipt 均完成，receipt/metric SHA 与 aggregate provenance 匹配；138 per-seed/46 aggregate/18 paired 行 |
| D-series V-JEPA2 D1/D3 | 24 stages，其中 6 个 AB 是 evaluation-only | V-JEPA2 L23，P8，seed0/1/2；D1 isolated/native × spatial/pooled；D3 AB/ABCD × spatial/pooled | 24 个 receipt 均完成，receipt/metric SHA 匹配；276 per-seed/92 aggregate/16 paired 行 |

### Independent P8：不把 workflow ledger 当训练清单

[RUN_MATRIX.csv](../results/independent_actionability_algebra_p8_final_v1_20260913/RUN_MATRIX.csv) 的 44 行含 feature extraction shard、退役从未执行的分支、smoke、aggregation。清单改从四份 `raw_results/<model>/training_and_integrity.json` 展开：每模型 4 种 scorer（direct/action_only/current_action_no_goal/goal_action_no_current）× raw/centered/standardized × 3 seeds = 36 个配置；另有 full/action_only/visual_only dynamics × 3 seeds = 9 个配置。

每个配置均匹配到非空样本结果，检查了选定 epoch 和有限训练曲线。协议先在 scene-disjoint train-derived fit/dev 上选 epoch，再以固定配置在原训练集 refit；本表将 selection/refit 视为同一最终配置，不冒充两项实验。Scorer 和 non-full dynamics 没有各自归档权重，CSV 标为 `NOT_ARCHIVED_BY_PROTOCOL`；这不否定现成结果，但不能宣传为全部 180 个权重可下载的 checkpoint release。

[FINAL_INTEGRITY_AUDIT.md](../results/independent_actionability_algebra_p8_final_v1_20260913/paper/FINAL_INTEGRITY_AUDIT.md) 说明 validation 曾用于 pilot，因此只能保留 validation 表述。真实特征来自独立 exact store，不借用旧 C checkpoint 或 canonical-v2 feature store。

### Continuous action 与 SE3：冻结协议不同

[Continuous-action protocol](../results/independent_goal_conditioned_continuous_action_p8_relational_v2_20260923/CONTINUOUS_ACTION_PROTOCOL.md) 使用 fit-only normalization、dev epoch selection 和 fit+dev refit；五种 learned variants 为 full_relational、full_concat、current_only、goal_only、shuffled_goal_train。Inference shuffled-goal、motion prior 和 a1 reference 是评估条件，不增加 100 个训练配置。全包 32,768 val rows、140 paired-effect rows 也不是训练计数；validation 仍按 validation 发布。

[SE3 frozen package](../results/experimental/se3_wan_vjepa2_paper_final_20260923_r3/FREEZE_MANIFEST.json) 则是 128 train / 32 validation / 276 evaluation，A→B-only supervision。每 cell/seed 的 full、action_only、without_condition、commutative_additive 是四个训练 head；copy 和 matched_shuffle 不另算训练。P1 composition/order 分析重用这 240 个 head，只有 synthetic positive control 新增五个 head。它与上面的 P8 实验不共用冻结协议。

本次直接读到了 55/60 个 cell/seed 的 COMPLETE、training_history 和 per_sample 证据，共 220 个真实表示 head；加五个 positive control 是 225 个可直接复查条目。V-JEPA2 L11 的五个 seed（20 head）可 stat 到非空 ckpt，也有可读 per-seed aggregate 和 final statistical audit，但其源 receipt/metric 文件拒绝读取。因此这 20 个 run 为 `NOT_VERIFIED/UNKNOWN`，package 仍为 `VERIFIED/YES_WITH_SCOPE`。逐 run 的新核验及权重发布前需提供可访问的持久化 receipt/metric/权重位置，**无需据此重新训练**。

## D-series V-JEPA2 D2：已训练、已评估，尚未统一汇总

三条 [D2 run receipts 的根目录](../experiments/d_series_scannetpp_v1/diagnostic_runs/) 均在 `dseries_vjepa2_d2_seed{0,1,2}_20260917_103733_zoetrope_r1/d2_vjepa2_pooled_s{seed}/` 下。逐条读取结果：

| seed | receipt status | 实际 epochs / best epoch | receipt dev scene-macro MAE | checkpoint bytes | per-sample metric rows |
|---|---|---|---|---|---|
| 0 | complete_development_evaluation | 15 / 8 | 0.22894822860109634 | 18,031,714 | 3,864 |
| 1 | complete_development_evaluation | 26 / 19 | 0.22935045856064645 | 18,031,714 | 3,864 |
| 2 | complete_development_evaluation | 20 / 13 | 0.22778667004751987 | 18,031,714 | 3,864 |

每条均含 552 个唯一 sample、69 dev scenes、h2/h4，以及七项条件；mae/absrel/delta1 全部有限。七条件为 true_motion、identity_motion、matched_horizon_shuffled_motion、independently_trained_context_only、last_context、train_layout_prior、true_pooled_target_reference。receipt 中 feature-audit SHA 与本地文件匹配，feature audit 的 failures 为空，sample-manifest SHA 与每一行指标一致。

这三条应进入“训练与 development evaluation 已完成”列表。它们仍以 `ready_for_release=NO` 标记的是**正式汇总表尚缺**：补三种子聚合、paired contrasts 和统一 D-series export；不要求先重训。Wan D1/D2 与 V-JEPA2 D1/D3 的已完成子矩阵可先按范围发布。[D-series protocol](../experiments/d_series_scannetpp_v1/protocol.yaml) 明确 612 train / 69 dev / **0 test**，不能把 dev 更名为 test。来源 depth 是 sensor 还是 mesh-rendered 的语义及 upstream release identifier 未解决，应保留协议里的未解决标注，不升级数据来源声明。

## 未 ready 的其他条目

| 工作 | 实际实验/计划单位与本次证据 | 完成所需的必要工作 |
|---|---|---|
| C3-from-C1 V1 | 10 模型 × P8/12/16/24 = 40 evaluation cells；DINO L0/F3R L24/V-JEPA2 L23/VideoMAE L23/Qwen -1/Wan L20/CogVideoX L20/Aether L1 共 32 个源 ckpt 可 stat；MiniMax L30、LingBot L30 的 8 个缺失 | 为 32 项恢复 exact V1 inputs；另 8 项先恢复精确 ckpt；随后执行 frozen action-set evaluation。没有 scored result，不能称为零效应；canonical-v2 不可替代 V1 |
| B2 P16 四模型 extension | V-JEPA2/VideoMAE/Wan exact ckpt 缺失；CogVideoX 为 NO_CONFLICT_EXISTING_RESULT，仅 output fingerprint 级身份 | 恢复三个精确 ckpt，补四模型可比 perturbation 结果；Cog 单模型不能替代完整 extension。已接受 B2 P12 不受影响 |
| Active representation MVP | 5 个请求身份分别列 scope：c8_step300、v46_step200、v50_step150、Cambrian/Qwen pretrain baselines；v50 只有真实单 state 三 tap forward PASS；r8 是 25/48 state FAILED | 完成所有 exact-checkpoint 的匹配 48-state canary，再完成约 2,000-state probe；不能把 forward smoke 算训练成功 |
| SeeSE3 replication | DINO middle-layer 三 seed42/43/44 请求；preflight 仅 276 train / 137 val / 138 test pairs；当前 project L0 是 last-hidden-state | 提供 representative middle-layer 特征，达到至少 1,000 train / 200 evaluation pairs 后执行请求的训练；与已完成 observed-order SE3 分开 |

## 发布前必需整理与可选研究

**对已完成六包必需**：保留本表的 protocol、split、seed 与 scope，使用各自唯一结果表和现有 provenance；不要混入 retired workflow 行、C3 输入失败、B2 P16 缺失项、smoke 或旧 pipeline 的不兼容特征。若发布权重，列出真实保留的 checkpoint 清单；对 SE3 L11 先解决已有证据的可访问性。D-series 全集发布另需 D2 aggregation 和统一 export，现有 complete submatrix 不必等待。

**可选新工作**：独立 scene/blind test、额外 probe seeds、SeeSE3 replication、Active MVP、P16 extension，以及为未归档 scorer 补齐可下载权重。只有打算声称这些新范围时它们才成为必要工作。不能为了“release-ready”虚构 test split，也不能把这些范围扩展当成重跑已完成训练的理由。
