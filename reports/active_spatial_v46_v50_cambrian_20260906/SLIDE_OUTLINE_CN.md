# 组会展示提纲（8页）

## 1. 问题与实验矩阵

- 科学问题：不同 credit window、KL、格式约束和 backbone 是否改善 embodied spatial navigation？
- 强调只分析 D0 修复后的 clean runs；V49 因 renderer failure 无有效训练。
- 展示实验定义表，不先报结论。

## 2. RL 是否真的有效？

- 放 `figures/02_offline_navigation.png`。
- 讲法：step 50 仍在 critic warmup，之后 ID 提升约 5-8pp、OOD 提升约 7-14pp。
- 结论：有效，但曲线非单调，必须做 checkpoint selection。

## 3. Qwen 版本对比

- V47 强 KL 最弱；V48-W1 没有明显输给 V46-W3；V50 OOD 略好但差距很小。
- 不把单 seed、<1pp 差异解释成显著提升。

## 4. 泛化发生在哪里？

- 放 `figures/03_ood_profile_step700.png`。
- scene/instance 提升强，category 持续弱；OOD aggregate 受 suite 难度影响。

## 5. 训练为什么不稳定？

- 放 `figures/04_training_stability.png`。
- entropy 与 invalid action 后期上升；critic EV 弱；V46 step 250 后曾明显回撤。

## 6. Cambrian-S 的结论

- 放 `figures/01_validation_learning_curves.png` 右图。
- C8 step 250 是候选窗口，final 退化；C4 格式错误长期更高。
- 外部评测仍缺失，不做跨 backbone 排名。

## 7. 最关键 insight

- 放 `figures/06_static_qa_transfer.png`。
- Navigation 增长明显，EASI-8 几乎不变：RL 更像学会交互策略，而非提升通用空间表征。

## 8. 下一步决策

- 补齐 Cambrian-C8 step 100/250/1000 的统一离线 ID/OOD + EASI-8。
- 对 V46/V48/V50 最有希望的 checkpoint 做至少 3 seeds 或 paired bootstrap。
- 引入 task-balanced objective/selection，重点解决 category、occlusion、centering。
- 训练侧加入 entropy/invalid-action early-stop gate，并长期保留中间 checkpoint。
