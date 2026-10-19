# 预测表征是否仍保留可读的空间结构？

- **问题：** 视频模型不仅要“看见”空间，还要在预测未来 latent 后，仍让同一个空间 readout 读出目标视图布局。
- **主实验：** D1 用真实 target feature 训练读出器；D2 比较真实相机运动与 identity / 打乱运动 / context-only；D3 将训练在真实未来 V-JEPA2 feature 上的读出器冻结后直接用于预测 feature。
- **D1（Wan）：** spatial target readout 的 MAE 为 h2 `0.155`、h4 `0.152`，明显优于 layout prior（`0.241/0.238`）、wrong-target 与 last-context 对照。
- **D2（Wan）：** 真实 motion 在 h2/h4 都优于 identity、matched-horizon shuffled motion 和同容量 context-only；增益小但三 seed、场景配对 CI 一致。
- **D3（V-JEPA2，核心结果）：** 真实 future feature `0.182/0.183`；冻结读出器用于 predicted feature 为 `0.246/0.243`；copy last state 为 `0.289/0.297`。
- **关键 insight：** V-JEPA2 的预测 latent 比简单复制保留更多可读空间信息，但与真实 encoder latent 存在 readout-interface shift；同容量重训可恢复至 `0.223/0.227`。
- **支撑检查：** prior、wrong target、copy、identity/shuffled motion、context-only 共同排除了纯布局先验、时间复制和无运动条件解释。
- **边界：** 全部为 development-only；没有独立 test split；相机位姿是条件变量，不等于对真实动作的因果干预。
