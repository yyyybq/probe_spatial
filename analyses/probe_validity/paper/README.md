# Probe validity / capability decomposition 论文材料

## 为什么做

本节检验 probe 高分究竟来自目标能力，还是 copy、表征连续性、query 构造或无序历史内容等替代解释。它是整篇论文中的 validity/decomposition 小节，不是独立 benchmark，也不把固定 predictor 的输入扰动解释为重新训练后的能力上限。

## 数据与模型

正文只比较默认层上的 V-JEPA2（L23）、VideoMAE（L23）、Wan（L20）和 CogVideoX（L20）。C1 使用已核验的 P12 结果与四项 `P16_VERIFIED_REPAIR`；B2 只使用完成 sanity 检查的 P12 paper-strengthening 结果。所有结果属于 development validation；名义 master test 与历史 development universe 不独立，但这不等于已经证明实际 legacy C train/val 存在监督样本污染。

## 设计

- C1 主诊断固定 predictor、context 与 target，比较 correct action 和 `MATCHED_TOTAL_POSE_DISPLACEMENT` K=8、candidate seed 42 的 wrong action。该池是跨样本的物理位姿变化匹配池，不是只有四个唯一候选的 same-scene pool。
- B2 以真实球面角误差为指标，报告 query mask/same-scene swap、current-frame/remove-visible 和 content-only shuffle/reverse 相对各自有效样本 FULL 的配对差值。GT object mask 参与历史 appearance query 的构造，但不会直接输入 spatial readout。
- `paper_results.csv` 是本节唯一取数入口；原始 P12 表、旧 P16 sensitivity rows 和项目 canonical 均不由本材料替代。

## 可写入论文的三条 insights

1. Global future-feature predictability 与 physical hard-pool action sensitivity 的模型排序不同；高 copy 分数也可以与固定 predictor 对 action 的使用同时存在。
2. V-JEPA2、VideoMAE、CogVideoX 的 P12/P16 hard-K8 action gain 为正，Wan 在两个 prefix 的 scene-clustered CI 均包含零；这是 cross-prefix-setting robustness，不是 history length 的因果效应，也不是跨模型显著性检验。
3. B2 对正确 query 和历史内容的依赖具有模型差异，但在当前协议下没有检测到可靠的 content-slot order effect。该结论不表示顺序必然不重要，也不表示 backbone 不含时间信息。

## 限制与排除项

单个 probe-training seed；candidate sampling seeds 不是训练 seeds；没有独立 frozen confirmation；P12/P16 target/query 不严格配对；模型架构和训练数据未完全控制；C1 controls、完整 K-grid 与 B2 priors 只能辅助使用且 provenance 不完整。V-JEPA2、VideoMAE、Wan 的 B2/P16 perturbation 因指定 checkpoint 缺失而排除；CogVideoX B2/P16 不能单独拼成四模型复现。真实 alternative futures、B2-O、DINO 和新的 geometry-bridge 结论均不纳入本节；F3R 只保留项目中既有的 smoke/control 身份。

## 如何使用

- 将 `probe_validity_section.tex` 插入论文 experiments/results 中合适位置；它包含约定的英文正文和两张主表。
- 所有表格数字应从 `paper_results.csv` 反查。修改数字时先更新该 CSV 的准确来源映射，再同步 LaTeX；不要直接从历史报告抄写或使用原始 `c1_validity.csv` 中身份未确认的 P16 行。
- 本目录只承担 section-level 写作，不修改 registry、canonical、ledger 或 paper-ready。
