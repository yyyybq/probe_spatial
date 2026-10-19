# 5.3 / 5.4 / 5.5 整合与配图建议

> **布局已更新：** 下方保留首次整合与配图设计的记录。当前正文已按[实验图布局重设计](PAPER_FIGURE_LAYOUT_20260923.md)压缩为两组并排图，方法示意和完整扫描移到附录；图片使用 `figures/exp_figures/compact_layout/` 中的半栏版本。连续动作图已对齐当前 relational V2，旧 V1 图不再引用。

已直接修改 `iclr27.tex`，并完成优先级最高的三张图。图中数值直接读取已有结果，没有训练或重跑模型。TeX 的三个新图片引用均使用相对路径。

## 已生成并加入正文的图片

| 位置 | PDF（TeX 使用） | PNG 预览 |
|---|---|---|
| 5.3 干预示意图 | [fig07_probe_interventions.pdf](figures/exp_figures/fig07_probe_interventions.pdf) | [PNG](figures/exp_figures/fig07_probe_interventions.png) |
| 5.4 共享冻结读出与误差图 | [fig08_predictive_interface.pdf](figures/exp_figures/fig08_predictive_interface.pdf) | [PNG](figures/exp_figures/fig08_predictive_interface.png) |
| 5.5 当前状态与目标贡献图 | [fig09_goal_conditioning.pdf](figures/exp_figures/fig09_goal_conditioning.pdf) | [PNG](figures/exp_figures/fig09_goal_conditioning.png) |

将三个 PDF 放到主 TeX 相邻的 `figures/exp_figures/` 即可；编译不需要 Python 或实验结果文件。每张图另提供同名 SVG，供后续编辑。5.4/5.5 的原始数值表已移到附录，正文使用图并保留对附录表的引用。

绘图源代码：`scripts/build_paper_intervention_figures.py`。从项目根目录运行 `python scripts/build_paper_intervention_figures.py` 可重建全部文件（需要 matplotlib 和 numpy）。`figures/exp_figures/fig07_09_plot_data.json` 记录实际绘图数据、相对源路径及 SHA256。5.3 的 history 面板采用明确标注的概念 token 图，不冒充真实 RGB 轨迹。

## 正文安排

- **5.3 — What do the probes actually use?** 按 action、history、transformation order 三个问题组织；配一张三面板干预示意图、一张两面板控制表和一个 SE(3) 公式。SE(3) 代表层的详细比较及正控制解释放入附录。
- **5.4 的 prediction-interface 段落。** 根据最新 D-series 结果，采用 D1 目标特异性、D2 运动条件效应、D3 原生预测器的冻结读出迁移三步论证。正文的组合图明确区分 real、native prediction、copy、retrained reader 与 layout prior，五行精确数值表移至附录。
- **5.5 — From representation to action.** 保留候选动作选择作为铺垫，接上连续动作输出。保留候选选择表，用双面板效应图展示当前状态与目标的不同贡献；连续动作的绝对数值表移至附录。
- 同步调整第 3 节的四层能力表述、第 4.1 节的 split 限定和第 5 节导语，使它们与新分析一致。

结论保留三层区分：有 action signal 不等于已学会空间动力学；使用历史内容不等于按顺序更新空间状态；候选动作选择依赖目标，不代表连续动作输出也能可靠依赖目标。SE(3) 的弱顺序信号还受到单步训练的 probe 泛化能力限制，不能单独归因于 backbone。

## 配图设计说明（前三项已实现）

### 优先级 1：5.3 的 intervention 示意图

**目的：让读者看懂“究竟改了什么、什么保持不变”。** 建议放在 5.3 开头，用一张横向三面板矢量图，约四分之一页；正文已有表格，不必在此图重复全部数字。

1. **Action 面板：** 同一个当前状态和未来 target，四条输入路径分别为 correct、shuffled、zero action 与 copy-last。固定的 context/target 用相同灰色，只有被替换的 action 用橙色。copy-last 直接连到评估端，明确它不经过动作条件预测。
2. **History 面板：** 已使用五个示意 token slots，标出哪些内容来自目标可见时刻；用保留/屏蔽矩阵展示 full、current-only、remove-visible、no-visual-tokens，并用 token 编号变化展示 shuffle/reverse。历史 appearance query 在各行保持固定。此图明确对应已提取 feature tokens 上的干预，五个 slots 只是示意，不是实验的真实 prefix 长度，也不是重新跑 backbone 的 RGB 删除实验。
3. **Order 面板：** 用相同的起始状态画三条路径：`T1 → T2`、`T2 T1` 的一次组合更新、`T2 → T1`。前两条共同对应真实 endpoint C，交换顺序的输出仍与 C 比较。可加一个小型平面“旋转＋平移”示意帮助理解非交换性，但必须标为概念示意，不能把交换顺序后的相机视图画成已经采集的真实反事实样本。

建议图注中心句：*Matched interventions distinguish information use from decodability.* 用 TikZ/矢量绘图和真实轨迹缩略图即可；不需要生成式插画。若版面紧张，只保留 history 与 order 两个面板，因为 action controls 最容易用文字理解。

### 优先级 2：5.4 的共享冻结读出示意图＋定量结果

**建议配图，重点解释“同一个 reader 能否直接读懂预测状态”。** 这比再增加 Wan/V-JEPA2 的总体排名图更有价值。建议做成一张横向组合图，替换本轮新增的 D3 小表，避免重复占页：

- **左侧约 40%：实验结构。** 上路是真实 future state，下路是 context 经 V-JEPA2 native predictor 得到的 predicted future state。两路明确接入同一个 `g_real`，标注“train on real, then freeze”；第三条灰色支路输入 copied context。另用虚线框画出 predicted state → independently retrained `g_pred`，说明它是相同容量的诊断对照，不是冻结迁移的一部分。
- **右侧约 60%：两个 horizon 的点图。** h2、h4 分面且使用相同 MAE 横轴；四行分别为 real/original、predicted/frozen、predicted/retrained、copy/frozen。每个 horizon 各画一条 train-layout prior 参考线。点及短误差线表示三训练种子的 mean±SD；如需展示迁移差值的显著性，应另外标注 B−A 的 paired scene CI，不能把 SD 解释为 CI。
- **图上应能一眼读出的关系：** real 最好；predicted/frozen 优于 copy、但略差于 layout prior；predicted/retrained 超过 prior、仍未追平 real。不要把这种关系画成“信息全部消失”或“重训已完全恢复”。

建议图注中心句：*Native prediction retains recoverable spatial structure but does not preserve the original readout interface.* 数据使用 V-JEPA2 三种子报告的 `D3 / native_target / spatial` 分支，不与 D1 的 `isolated_target` 数字混用。Wan D1 使用 spatial readout，D2 使用 pooled branch；它们是前置控制证据，不与 D3 拼成一条横向模型排行榜。

**深度图案例可选，优先放附录。** 若已有逐样本深度输出，可对同一场景并列展示真实目标 RGB/GT、real-reader 输出、predicted/frozen 输出、predicted/retrained 输出和 copy 输出，使用统一深度范围；展示一个典型场景和一个迁移差距较大的场景，并说明挑选规则。当前材料主要支持定量比较，不应凭 aggregate 数字绘制看似真实的预测深度图，也不必为正文配图重新运行模型。

### 优先级 3：5.5 的目标依赖效应图

**目的：直观呈现“当前状态有用，但加入目标没有增益”。** 建议两列横向点区间图，每列四行对应四个模型，均画零效应参考线：

- 左列：FULL − GOAL-ONLY 的 positive-progress 差值，单位为百分点，四个区间均在零线右侧。
- 右列：FULL − CURRENT-ONLY 的 goal reduction 差值，四个区间均在零线左侧。

两列使用各自的横轴单位，不能共用刻度。点表示配对效应，线段表示 scene-bootstrap 95% CI；训练种子 SD 与 CI 不混用。数据直接来自 `continuous_action_paired_effects.csv`。

这张图比再画一组 FULL 绝对成功率柱状图更能承载本节结论。若采用，建议**替换**正文的连续动作表，把方向误差等完整数值放到附录，避免重复占页。不要把候选选择的约 76–80% 与连续输出的约 49–55% 画成同一纵轴上的能力下降：二者样本、任务和指标不同。

### 可选：SE(3) 多层结果放附录

现有 SE(3) 结果包已包含 `figures/figure_multilayer.*`、代表层比较及几何正控制图。可以在附录采用“endpoint gain 与 order margin 随层变化”的双面板图：前者显示预测收益，后者保留零线与 CI。正文不建议再加整套多层曲线，以免冲淡控制实验的主线。若展示 oracle 与 learned GRU，要与 latent 指标分开标注单位，不能把米与 normalized latent margin 放在同一尺度。

**当前版面安排：** 已加入上述前三张图，5.4/5.5 的原始数值表移至附录。若后续必须进一步压缩，可优先保留 5.3 和 5.4 的示意与结果图。

## 数字来源与核对

- C1 正式 p12/p16 action-gain、copy 和 B2 current/remove-visible/order：`analyses/probe_validity/paper/paper_results.csv`。不使用报告中已 superseded 的 P16 sensitivity 数字。
- 用户要求的“删除所有 visual history”未收入上述精简论文表。本次明确补用 `analyses/probe_validity/b2_validity.csv` 中 `phase2_b2_{model}_l{layer}_p12`、`QUERY_FEATURE_ONLY` / `CORRECT_QUERY+NO_HISTORY` 的已有固定 predictor 结果；不是新训练的 query-only baseline。四模型均为 165 个样本。
- Wan correct/shuffled/zero/copy 使用 `c1_validity.csv` 的 `phase2_c1_wan_l20_p12`、`HISTORICAL_GLOBAL` 条件。原稿 shuffled 值 9.95% 与该记录的 `batch_roll` R@1 = 0.0978261 不一致，已改为 **9.78%**；其余三个值分别为 15.22%、5.43%、11.96%。
- SE(3)：用户指定的 `se3_wan_vjepa2_paper_final_20260923_r3` 结果包。436 是整个协议的 tuple 总数，正文明确以 276 个 evaluation tuples 报告评估范围。
- 连续动作：用户指定的 `CONTINUOUS_ACTION_RESULTS.md`，并核对其 protocol、main CSV 与 paired-effects CSV。与候选动作选择分别交代样本数和 split。
- D-series：用户指定的 `experiments/d_series_scannetpp_v1/paper_outputs/dseries_development_20260923_r1/{paper_section.tex,analysis_report.md}`，并回查 `diagnostics/dseries_wan_d1_d2_three_seed_20260923_zoetrope_r2` 与 `diagnostics/dseries_vjepa2_d1_d3_three_seed_20260917_zoetrope_r1` 的报告、`across_seed_metrics.csv` 和配对差值表。D3 新表保留四位小数，避免真实 future 条件的种子 SD 被舍入成零。Wan D2 的六个 MAE 对照区间均排除零，但不能把此结论扩展到所有指标（例如 h2 context-only 对照的 Delta1 CI 包含零）。冻结 D3 相对 prior 的劣势仅作均值描述；没有据此补写未报告的显著性检验。

已核对图中数值与误差线、检查 PNG 预览和 LaTeX 引用；三个新 PDF 都存在且使用相对路径，5.3/5.4/5.5 编号保持不变。当前版本的标签与交叉引用均可解析，原有 `tab:spatial-task-definitions` 引用也已在当前原稿中补齐。当前环境没有可用的 LaTeX 编译器，尚未验证整篇论文的最终分页和浮动体位置。
