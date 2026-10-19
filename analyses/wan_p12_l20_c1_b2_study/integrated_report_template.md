# Wan P12/L20 C1/B2 Integrated Validity Analysis

Status: `COMPLETE_SYNTHESIS`

本报告由结构化结果自动生成，综合以下两组已经完成的实验，不引入新的模型、样本、层、prefix 或训练结果：

- Wan P12/L20 Action Dynamics Construct Validity
- C1/B2 Cross-Objective Sample-Level Phase 2

Wan P12/L20 construct-validity 审计是论文中 Wan action dynamics 的主结果；跨模型 C1/B2 实验用于解释该结果在不同候选池、任务和预训练目标下的相对位置。Wan P8 仅保留为 supplementary prefix-robustness evidence。

## 1. 综合结论

Wan P$prefix/L$layer 表征包含可复现、分布较广的动作敏感信号，但现有证据不足以把它解释为稳定的 action-conditioned dynamics representation。

> Wan 的 probe 输出会随动作改变，正确动作在 ordinary shuffle、scene-independent formal shuffle、same-scene hard negatives 和 exhaustive wrong actions 下具有一致优势；但该优势未可靠超过强 persistence baseline，也未在 matched-action hard control 中稳定成立。B2 同时表明 Wan 明显利用历史中与目标相关的可见内容，却没有稳定的 frame-order 或 correct-query dependence。因此，整体证据更符合 weak action sensitivity on top of history-content retrieval，而不是 structured action-conditioned world dynamics。

综合证据标签：

- `ACTION_SENSITIVITY = REPRODUCIBLE_BUT_WEAK`
- `ACTION_CONDITIONED_DYNAMICS = NOT_ESTABLISHED`
- `HISTORY_CONTENT_DEPENDENCE = SUPPORTED`
- `CORRECT_QUERY_DEPENDENCE = NOT_RELIABLY_SUPPORTED_FOR_WAN`
- `ORDERED_TEMPORAL_MEMORY = NOT_SUPPORTED`
- `WAN_C1_PAPER_ROLE = PRIMARY_WEAK_ACTION_SIGNAL`
- `WAN_B2_PAPER_ROLE = HISTORY_CONTENT_RETRIEVAL_EVIDENCE`

## 2. Wan P12/L20 关键结果对齐

### 2.1 C1 严格 construct-validity 审计

| 条件或对比 | 结果 | 95% scene-bootstrap CI | 判定 |
| --- | ---: | ---: | --- |
| Correct action R@1 | $correct_r1 | - | 主条件 |
| Formal shuffled action R@1 | $formal_shuffle_r1 | - | 100 个 exact-marginal、scene-independent derangements |
| Eval-null action R@1 | $null_r1 | - | zero-action control |
| Last-state copy R@1 | $last_copy_r1 | - | persistence control |
| Prefix-pooled copy R@1 | $prefix_copy_r1 | - | pooled-state control |
| Correct minus formal shuffle R@1 | $formal_gain | [$formal_ci_low, $formal_ci_high] | 通过，permutation p=$permutation_p |
| Correct minus eval-null R@1 | $null_gain | [$null_ci_low, $null_ci_high] | 通过 |
| Correct minus all-wrong cosine | $wrong_gain | [$wrong_ci_low, $wrong_ci_high] | 通过 |
| Correct minus last-state copy R@1 | $copy_gain | [$copy_ci_low, $copy_ci_high] | 未通过 |
| Action-class breadth | $positive_classes/$total_classes positive | - | 通过 |
| Scene breadth | $positive_scenes/$total_scenes positive | - | 通过 |

严格审计说明动作信号并非由单个 scene、少数动作方向或候选泄漏驱动。失败点很集中：正确动作没有显著击败 last-state persistence copy。因此最终 gate 是 `$phase1_gate`，最终解释为 `$interpretation`，而不是无动作信号。

### 2.2 与前一轮 C1 sample-level 结果的关系

| C1 控制 | Correct gain | 95% clustered CI | 解释 |
| --- | ---: | ---: | --- |
| Historical batch-roll shuffle | $legacy_shuffle_gain | 未作为最终 formal CI | 与新 formal shuffle 数值高度一致 |
| 新 formal scene-independent shuffle | $formal_gain | [$formal_ci_low, $formal_ci_high] | 将旧 shuffle 趋势升级为正式证据 |
| Matched-magnitude wrong action | $matched_gain | [$matched_ci_low, $matched_ci_high] | hard control 下不稳定 |
| Same-scene wrong action | $same_scene_gain | [$same_scene_ci_low, $same_scene_ci_high] | scene 内 hard-negative specificity 为正 |
| Last-state copy | $copy_gain | [$copy_ci_low, $copy_ci_high] | persistence gate 未通过 |

旧 batch-roll gain `$legacy_shuffle_gain` 与新 formal-shuffle gain `$formal_gain` 几乎相同，说明旧结果不是 batch-local mapping 偶然产生的。新审计进一步排除了 fixed point、same-scene donor 和 action-marginal 改变，并通过 100 个固定 seed 给出正式 permutation 检验。

matched-magnitude action 和 last-state copy 都是更难的控制，二者 CI 均跨过 0。这说明 Wan 能区分正确动作与一般错误动作，但还不能证明它学习了超过状态持久性和动作幅值相关性的精细 transition function。

## 3. B2 对 Wan 机制解释的约束

Wan B2 使用 $full_n 个 hidden-object 样本、$full_scenes 个 scene；same-scene query swap 使用其中 $query_swap_n 个样本、$query_swap_scenes 个 scene。角度误差增加为正表示 perturbation 使结果变差。

| B2 条件 | Mean error | 相对 full-history 增量 | 95% clustered CI | 判定 |
| --- | ---: | ---: | ---: | --- |
| Full history + correct query | $full_error deg | 0 | - | 主条件 |
| Current frame + correct query | $current_frame_error deg | $current_frame_delta deg | [$current_frame_ci_low, $current_frame_ci_high] | 历史信息贡献显著 |
| Remove target-visible frames | $remove_visible_error deg | $remove_visible_delta deg | [$remove_visible_ci_low, $remove_visible_ci_high] | 依赖历史目标内容 |
| Query feature only | $query_only_error deg | $query_only_delta deg | [$query_only_ci_low, $query_only_ci_high] | 仅 query 明显不足 |
| Query mask + full history | $query_mask_error deg | $query_mask_delta deg | [$query_mask_ci_low, $query_mask_ci_high] | correct-query dependence 不稳定 |
| Same-scene query swap | $query_swap_error deg | $query_swap_delta deg | [$query_swap_ci_low, $query_swap_ci_high] | correct-query dependence 不稳定 |
| Frame-order shuffle | $shuffle_error deg | $shuffle_delta deg | [$shuffle_ci_low, $shuffle_ci_high] | 无可靠顺序依赖 |
| Reversed history | $reverse_error deg | $reverse_delta deg | [$reverse_ci_low, $reverse_ci_high] | 无可靠方向依赖 |
| Global constant prior | $global_prior_error deg | $global_prior_delta deg | [$global_prior_ci_low, $global_prior_ci_high] | full model 未可靠胜过 prior |

B2 排除了一个过度简单的解释：Wan 并非在所有任务上只复制最后一帧，因为 current-frame-only 相比 full history 显著恶化。真正稳定的 B2 信号来自历史内容，特别是过去曾包含目标的帧。

但 B2 也没有支持更强的时序机制：shuffle 和 reverse 的 CI 均跨过 0；mask 和 same-scene query swap 也没有给出可靠的 correct-query dependence。因而 Wan 更像是在聚合或检索历史中的目标相关内容，而不是执行顺序敏感、query-addressed 的 belief update。

## 4. C1 与 B2 的联合机制判定

| 机制假设 | C1 证据 | B2 证据 | 联合判定 |
| --- | --- | --- | --- |
| 动作输入确实影响 probe | Correct 显著优于 formal shuffle 和 eval-null | B2 不直接测试动作输入 | `SUPPORTED` |
| 学到了精细 action-conditioned transition | Matched-action 与 copy hard gates 不稳定 | 无对应正证据 | `NOT_ESTABLISHED` |
| 仅复制最后状态 | C1 copy 很强且未被可靠击败 | Current-frame-only 显著变差 | `TOO_SIMPLE_ACROSS_TASKS` |
| 利用历史目标内容 | C1 高 state persistence | 移除 target-visible frames 显著变差 | `SUPPORTED` |
| 对正确 query 做寻址 | C1 不测试该机制 | Mask/swap 均不显著 | `NOT_RELIABLY_SUPPORTED` |
| 编码有序的时间更新 | C1 不直接判定顺序 | Shuffle/reverse 均不显著 | `NOT_SUPPORTED` |

联合结果因此不是“Wan 完全没有动力学信息”，也不是“Wan 已形成世界模型”。最符合现有数据的中间结论是：Wan 的表征含有动作相关变化，但 probe 性能仍高度受状态持久性与历史内容可检索性解释。

### 4.1 Representation Geometry / Cosine Illusion Bridge

历史 P12 common-set 分析复用了同一 276 个样本和 Wan P12/L20 checkpoint；它是事后机制解释，不改变严格 construct-validity gate。Wan 的 last-state-to-target raw cosine 为 `$wan_copy_raw_cosine`，即 temporal displacement 仅为 `$wan_temporal_displacement`；full prediction 的 raw cosine 为 `$wan_full_raw_cosine`。因此，Wan 的 MSE/cosine 目标确实处在高时间平滑区间。

但这不是一个可由全局 feature mean 完全解释的假象：target-mean centering 后，Wan 的 copy cosine 仍为 `$wan_copy_centered_cosine`，full prediction cosine 为 `$wan_full_centered_cosine`。这与严格审计中的正但不精确的 full-minus-copy R@1 共同支持“强 temporal smoothness 加上弱的增量 action signal”，而不支持“没有任何动力学”或“已建立 non-copy dynamics”这两种极端解释。

此前记忆中的 DINO “最高”应区分指标：DINO 的 centered-copy cosine 为 `$dino_copy_centered_cosine`，而其 P12 copy R@1 为 `$dino_copy_r1`；Wan 才是 raw temporal cosine 更高的模型。完整表、方法和不可用量（random/same-scene pairs、effective rank、whitening、action-matched separation）见 `analyses/probe_validity/REPRESENTATION_GEOMETRY_BRIDGE.md`。该旧包装结果明确将 whitening 留为 `NaN`，不应由 aggregate cosine 伪造 covariance 或 whitening 结论。

## 5. 对跨目标 C1/B2 结论的修正

### 可以保留

1. `OBJECTIVE_DEPENDENT_SIGNAL` 可以作为跨模型描述性结论保留。Wan 的 global C1 R@1 为四个 core models 中最高，但 same-scene gain 低于 V-JEPA2 和 VideoMAE，继续支持 absolute predictability 与 action specificity 的 ranking dissociation。
2. Wan 与 CogVideoX 在 C1 same-scene action gain 上方向一致，generation-family 的 C1 sign agreement 可以保留为有限的 family-level observation。
3. B2 的 `REINTERPRET_AS_HISTORY_CONTENT_RETRIEVAL` 得到 Wan 单模型证据支持。

### 必须降级或避免

1. 不应把 Wan 的最高 C1 R@1 描述为最强的 action-conditioned dynamics。它部分反映目标特征的可预测性和 state persistence。
2. 不应由 Wan/CogVideoX 的 C1 同号结果推出 generation objectives 普遍产生统一的动态记忆机制。两者在 B2 query/history dependence 上方向分裂。
3. 不应声称 Wan 具有 ordered temporal memory、query-addressed belief state 或经过 construct validation 的 world model。
4. 不应将 B2 full-history 与 global prior 的接近解释为历史完全无用。paired perturbations 已明确显示历史目标内容有贡献。

## 6. Paper-Ready Interpretation

在预先固定的 Wan Prefix=$prefix、layer=$layer 设置上，正确动作相较于 100 个保持动作边际分布且跨场景的正式 shuffle，使未来特征检索 R@1 提高 $formal_gain，scene-bootstrap 95% CI 为 [$formal_ci_low, $formal_ci_high]。该优势覆盖全部 $total_classes 个动作 strata 和 $positive_scenes/$total_scenes 个场景，且相较全部错误动作的 cosine margin 为正。然而，正确动作相较 last-state copy 的 R@1 优势仅为 $copy_gain，95% CI 为 [$copy_ci_low, $copy_ci_high]，未通过 persistence gate。互补的 B2 干预显示，Wan 依赖历史中目标可见内容，但对帧顺序和正确 query 的依赖不稳定。综合而言，Wan 表征包含可复现但较弱的动作敏感信号，更符合历史内容检索叠加有限动作调制，而不足以证明结构化的动作条件动力学表征。

At the pre-specified Wan Prefix-$prefix/layer-$layer setting, correct actions improved future-feature retrieval over 100 exact-marginal, scene-independent action shuffles by $formal_gain R@1 (scene-bootstrap 95% CI [$formal_ci_low, $formal_ci_high]). The effect was broad across all $total_classes action strata and $positive_scenes of $total_scenes scenes, and the cosine margin against exhaustive wrong actions was positive. However, the R@1 advantage over last-state copy was only $copy_gain (95% CI [$copy_ci_low, $copy_ci_high]) and failed the persistence gate. Complementary B2 interventions showed dependence on historically target-visible content but no reliable frame-order or correct-query dependence. Together, these results support a reproducible but weak action-sensitive signal layered on history-content retrieval, rather than a validated structured action-conditioned dynamics representation.

## 7. 范围与限制

- C1 和 B2 使用不同任务、指标和样本集合，联合结论属于机制 triangulation，不是可直接合并效应量的 meta-analysis。
- C1 action strata 是从连续 9-D relative pose 确定性派生的诊断类别，不是原生动作标签。
- 严格 C1 Phase 1 因 copy gate 失败而按协议停止，未执行 capacity-matched Phase 2。
- B2 的 global-prior 比较标记为 `TENTATIVE_SMALL_SUBSET`，应以 paired perturbation 结果作为主要机制证据。
- 跨模型 objective-family 结论是固定模型集合上的描述性结果，不应外推到所有模型。

## 8. Evidence Sources

- `reports/wan_p12_l20_action_dynamics_construct_validity.md`
- `results/analysis/wan_p12_l20_action_validity/summary.json`
- `analyses/probe_validity/PROBE_VALIDITY_REPORT.md`
- `analyses/probe_validity/c1_validity.csv`
- `analyses/probe_validity/b2_validity.csv`
- `analyses/probe_validity/c1_sample_diagnostics.csv.gz`
- `analyses/probe_validity/b2_sample_diagnostics.csv.gz`
- `analyses/probe_validity/REPRESENTATION_GEOMETRY_BRIDGE.md`
- `results/paper_validity_20260821T200224Z_2075f3e/c_baselines_summary.csv`
