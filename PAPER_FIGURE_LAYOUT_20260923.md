# 实验部分图的布局重设计

正文原有 10 个独立整栏图，调整为 **2 个组合图、4 个半栏面板**。每个面板占 `0.48\linewidth`，用 `subfigure` 并排；正文的大表继续承担完整数值比较。所有新图片引用均为相对路径。

[查看组合布局预览](figures/exp_figures/compact_layout/layout_preview.png)（这是面板排列预览，不是整篇论文的编译分页）。

## 正文保留什么

| 位置 | 左半栏 | 右半栏 | 用途 |
|---|---|---|---|
| 5.2 | 最佳观测层：四个 backbone × 六项读出；数字为 block index，颜色为归一化深度 | 八模型 × 七指标的 p24−p8 效应 | 同时说明能力随网络层和历史长度变化 |
| 5.4–5.5 共用 | V-JEPA2 的 real / predicted / retrained / copy 读出误差，合并 h2/h4 | Goal-conditioned action prediction 的 continuous-output setting 三组配对效应，统一用 positive-progress 百分点 | 分别说明空间接口的部分保留，以及目标响应缺少可靠效用 |

半栏图按约 2.7 英寸原生宽度重新绘制，移除图内的大标题及重复说明，保留字号、误差线和必要图例。没有把原来的宽幅多面板图直接压缩到半栏。

层图中只有一个测点的单元格使用浅色和星号，明确不称其为“最佳层”。历史长度图保留七指标、55 个有效端点对及唯一缺失项，未挑选有利的模型子集。

## 原有十张图的去向

| 原图 | 调整 | 理由 |
|---|---|---|
| Capability atlas | 不再插入论文，保留原文件 | 与主结果表重复，排名热图没有新增实验信息 |
| Geometry–dynamics scatter | 不再插入论文，保留原文件 | 再次聚合主表数值；正文两句即可说明不同能力有不同领先模型 |
| 全层曲线 | 移至附录 | 适合核对全扫描，正文只需摘要 |
| 峰值层 heatmap | 重绘成正文左半栏 | 删除单输出模型的空白带，单测点明确标记 |
| Prefix delta heatmap | 重绘成正文右半栏 | 保留八模型、七指标，以紧凑矩阵替代大面积色块 |
| Layer × prefix 的 instance identity / out-of-view object localization 联合图 | 保留最新原图并移至附录 | 是交互细节，不需要与边际摘要并列占正文；保留 VideoMAE/V-JEPA2 两模型、两任务的对称展示 |
| Action/history/order 示意图 | 移至附录 | 解释干预设计，正文已有公式和结果表 |
| Timestep 全扫描图 | 移至附录 | 正文的小表和结论已足够，全网格用于核对 |
| D3 共享 reader＋结果组合图 | 正文只保留半栏定量图 | reader 冻结/重训流程已由正文和图注说明 |
| continuous-output setting 双面板图 | 重绘成半栏三对照 forest plot | 相同指标共享横轴，加入同一 checkpoint 的 goal shuffle 对照 |

## 数据与排版细节

- continuous-output setting 图对齐当前论文的 **relational V2**：五个训练 seeds、5,000 次 scene-bootstrap、Wan L29/t749。之前的 V1 三种子图不再由 TeX 引用。
- 动作图的三条对照为 FULL−GOAL-ONLY、FULL−CURRENT-ONLY、correct−inference-shuffled goal，全部采用 positive progress 的百分点差值；goal reduction 的完整数值继续放附录表。
- D3 图仍是 native-target spatial 分支，误差线为三训练种子的 SD；不与动作图的 95% CI 混淆。
- 正文浮动体使用 `[tbp]`，允许页顶、页底或浮动页安排；没有逐小节硬分页或负间距压缩。仅在参考文献前用一次 `\FloatBarrier`，避免实验图漂移到参考文献中。导言区新增常用包 `placeins`。
- 图注、正文与附录的引用同步调整。原始结果数据及原图片文件不删除。

复现入口：`python scripts/build_paper_compact_layout.py`（需要 matplotlib、numpy）。新文件在 `figures/exp_figures/compact_layout/`，包含 PDF、PNG、SVG、布局预览及 `plot_data.json`；后者记录图中实际数值与原始 CSV 的 SHA256。

当前环境缺少 LaTeX 编译器和会议样式等完整依赖，所以这里验证的是图像可读性、TeX 结构、相对路径、引用和数据一致性；最终页数与浮动体落页仍需在完整论文环境中编译确认。
