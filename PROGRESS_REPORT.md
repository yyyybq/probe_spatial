# 研究进度汇报

**时间**：2026年5月  
**数据集**：InsScene-15K（8389训练 / 953验证样本）  
**两个视频基础模型（VFM）**：Wan (C=1536) 和 V-JEPA2 (C=1024)

---

> **2026-06-19 protocol change:** B1/B2 no longer receive camera pose or an
> explicit current-frame role condition. C1 now requires exact target/action
> alignment and global retrieval evaluation. Older B1/B2/C1 numbers below are
> historical diagnostics and require rerunning before publication.

## 2026-06-14 方法更新：支持 layer-wise probing

当前默认 probe 层已确认并保持不变：

| Backend | Default feature file | 含义 |
|---|---|---|
| Wan2.1 | `feature_t749_layer20.sft` | diffusion transformer block 20 at timestep 749 |
| CogVideoX | `feature_t749_layer20.sft` | diffusion transformer block 20 at timestep 749 |
| V-JEPA2 ViT-L | `feature_layer23.sft` | last encoder block, 0-based layer 23 |
| Qwen2.5-VL / BAGEL | `feature_layer-1.sft` | 当前 MLLM default visual-token / last-layer cache |

新增能力：
- `vidfm3d/utils/feature_layers.py` 集中维护默认层、last layer、文件名和通道数规则。
- `features/run_inscene15k.py` 支持 `--output-layers default|last|all|...` 和 `--all-layers`，可一次缓存多层 Wan / CogVideoX / V-JEPA2 feature。
- `features/run_inscene15k_mllm.py` 支持同样的层参数；Qwen2.5-VL 的 `-1` 保持当前 visual-merger 默认，非负层号通过 hooks 抓 vision tower block 输出。
- `scripts/run_layer_sweep.sh` 可复用现有 experiment YAML 扫不同 layer；`scripts/summarize_layer_sweep.py` 汇总 best layer score、default layer score、last layer score 和 layer-wise CSV。
- `vidfm3d/eval_diag.py` 的 summary 写入 `feature_layer / feat_postfix / job_name`，便于跨层比较。

---

## 一、研究目标

我们想知道：**视频基础模型是否理解三维场景结构？**

具体方法：用一个轻量级的"探针"网络，冻结 VFM 的特征，训练探针来完成几个和三维场景理解相关的任务。如果探针能学好，说明 VFM 特征里已经包含了这些三维信息。

---

## 二、探针实验概览

### A 组：基础三维感知（A1，之前已完成，结果见下方单独小节）

A1 是沿用 VidFM3D 项目的原始基线探针，直接从 VFM 特征预测**深度图、相机位姿、场景实例分割**，属于像素对齐任务。

### A/B/C 组新探针（本次汇报重点）

#### C1 — 动作预测（Action Dynamics）
**问题**：给定当前视频片段和相机运动指令，预测下一帧的 VFM 特征向量是什么？  
**评估**：R@1（在一个 batch 内，预测的特征向量能否找回正确目标，随机基线=6.25%）

| 实验 | R@1 | cos相似度 | 平均排名 |
|------|-----|---------|---------|
| Wan v1（真实特征） | **26.7%** | 99.3% | 3.83 / 16 |
| Wan ctrl（扰乱特征） | 5.8% ≈ 随机 | 97.5% | 7.48 / 16 |
| V-JEPA2 v1（真实特征） | **22.8%** | 88.3% | 4.44 / 16 |

**重要发现：cos相似度会误导人。** Wan ctrl 的 cos 相似度 97.5% 和 Wan v1 的 99.3% 看起来差不多，但 R@1 才暴露真相：ctrl（5.8%）约等于随机猜，说明扰乱特征完全没有信号。而 Wan v1（26.7%）和 V-JEPA2（22.8%）明显高于随机，说明两个 VFM 都确实编码了运动动态信息。

---

#### A2 — 视角一致性（View Consistency，新）
**问题**：给定同一场景的两段视频，预测它们的拍摄视角是否有重叠？  
**评估**：overlap_acc（分类准确率）和 overlap_mae（重叠比例误差）

| 实验 | overlap_acc | overlap_mae |
|------|------------|-------------|
| Wan v1（真实特征） | 85.6% | 0.143 |
| Wan ctrl（扰乱特征） | 78.7% | 0.180 |
| V-JEPA2 v1（真实特征） | 85.2% | 0.142 |

**发现了数据集设计问题：** 测试集中，84.3% 的样本是"无重叠"（负样本），只有 15.7% 是"有重叠"（正样本）。这意味着**全部预测"无重叠"的最简单基线就能达到 84.3% 准确率**。  

Wan v1（85.6%）和 V-JEPA2（85.2%）只比最简单基线高 1 个百分点左右，这说明：
1. 准确率这个指标在此任务上过于乐观，意义不大；  
2. Wan ctrl（78.7%）低于最简单基线（84.3%），说明扰乱特征甚至会引入干扰。  
3. 建议后续用 AUC 或分别报告正负样本准确率来更好地评估这个任务。

---

#### A3 — 异常序列检测（Abnormal，新）
**问题**：给定一段视频，判断帧的时间顺序是否被打乱？  
**评估**：`pair_acc`（严格双向判别：同一样本上 `p(shuffled)>0.5` 且 `p(normal)≤0.5` 同时满足才算对；随机基线=25%）；`mean_delta` = `p(shuffled) − p(normal)`，越大越好。

| 实验 | pair_acc | mean_delta | 备注 |
|------|---------|-----------|------|
| Wan v1（真实特征） | **86.3%** | 0.811 | dtype-fix 重训，50 epochs，`y7ll3fpw/epoch=49` |
| Wan ctrl（扰乱特征） | **16.5%** | 0.002 | dtype+seed 双修复，50 epochs，`d518meez/epoch=49` |
| V-JEPA2 v1（真实特征） | 97.5% | — | 旧结果（dtype-leak 前），**dtype-fix 重训中**（GPU 7，PID 1040875） |

**Bug 调查历程（ctrl 经历了两轮根因排查）**：

#### Bug 1（已修复）：dtype 不一致 → 探针成了 dtype 探测器
- `vfm_feat` 在 [inscene15k_dataset.py L734](vidfm3d/data/components/inscene15k_dataset.py#L734) 处加载后**未做 dtype 转换 → fp16**；scramble 分支的 `torch.randn` 继承 fp16。
- `vfm_feat_shuffled` 在 `_load_shuffled_feat`（[L619](vidfm3d/data/components/inscene15k_dataset.py#L619)）显式 `.float()` → fp32；scramble 分支继承 fp32。
- fp16 的离散化指纹与 fp32 不同（fp16 尾数只有 10 bit），2 层 Transformer 完全能学到这种指纹。
- **验证**：swap 两条流 dtype，预测完全翻转；matched fp32 后 ctrl 训练出 99.1%（而非 25%），说明还有第二个 bug。
- **修复**：L734 追加 `.float()`。

#### Bug 2（已修复）：randn seed 仅依赖 idx → 跨 split 噪声复用，探针记忆训练噪声
- scramble_feat 的噪声 seed 原为 `hash((int(idx), 0/1))`，其中 `idx` 是该 split 内的位置编号（val 0–952 ⊂ train 0–8388）。
- 结果：val 中 idx=0..952 的噪声模式与 train 中 idx=0..952 的噪声模式**完全相同**。
- 探针（7.1M 参数，952 个 val 样本）在训练时已见过所有 val 噪声模式，直接记忆了 `noise_pattern → normal/shuffled` 的映射，与时序内容无关。
- **修复**（[inscene15k_dataset.py L768/L899](vidfm3d/data/components/inscene15k_dataset.py#L768)）：seed 改为 `hashlib.md5(scene_dir + ":n/s")` → 每个场景唯一、跨 split 不重叠。

**结论**：A3 ctrl pair_acc=16.5%（mean_delta≈0.002），接近随机基线（25%）且 delta ≈ 0，证实两个 bug 均已修复，ctrl probe 无法从纯随机噪声中学到任何信号。Wan v1（86.3%）结果完全可信——Wan 特征确实编码了帧时序信息。A3 V-JEPA2 v1 需重训（同样受两个 bug 影响）。

---

### B 组：隐藏物体位置预测（Ego Belief，B1）

**问题设定**：  
视频中某个物体之前在相机视野里，随后相机移走了。现在，给定这段运动轨迹，预测该物体**现在相对于相机是在哪个方向**（方位角 azimuth、仰角 elevation）和大致距离（对数距离误差）？

**这个任务分了四组实验，用来做消融分析（ablation study）：**

| 实验 | 输入信息 | 方位角误差 | 仰角误差 | 对数距误差 |
|------|---------|----------|---------|-----------|
| Wan v1（完整版）| VFM特征（有GT掩码） + 相机位姿 | **18.1°** | **11.4°** | **0.249** |
| Wan ctrl（对照组）| 扰乱后的特征 + 相机位姿 | 23.4° | 15.1° | 0.297 |
| Wan nomask（无掩码）| VFM特征（均匀池化，无GT掩码） + 相机位姿 | 23.7° | 15.1° | 0.289 |
| Wan poseonly（仅位姿）| 只有相机位姿（无视觉特征） | 24.3° | 15.7° | 0.301 |
| V-JEPA2 v1（完整版）| VFM特征（有GT掩码） + 相机位姿 | **17.6°** | **11.7°** | **0.257** |

**核心发现**：

1. **VFM 特征确实有用**：Wan v1（18.1°）vs. ctrl（23.4°）vs. poseonly（24.3°）——加入真实 VFM 特征使方位角误差下降了约 22%。

2. **没有 GT 掩码，VFM 特征就没有用（nomask ≈ ctrl ≈ poseonly）**：  
   无掩码时（23.7°）和对照组（23.4°）、仅位姿（24.3°）几乎一样差。说明如果不告诉模型"哪部分特征对应目标物体"，VFM 特征无法帮助推理物体位置。

3. **这是 B1 的设计缺陷**：GT 掩码在实际场景中是无法获得的，属于"信息泄露"。这就是为什么我们要设计 B2。

4. **V-JEPA2 略好于 Wan（B1）**：V-JEPA2（17.6°）比 Wan（18.1°）表现稍好，说明 V-JEPA2 特征可能更擅长编码三维空间位置信息。

---

## 三、A1 基线结果（VidFM3D 原始探针：深度 / 相机位姿 / 场景分割）

**任务说明**：直接从冻结的 VFM 特征中通过像素对齐的 DPT 解码器预测三个任务：
- **深度**：每帧逐像素深度预测（loss_depth，越低越好）
- **相机位姿**：估计相邻帧之间的相机变换（Auc_30：旋转误差<30° 的 AUC；Rac_15：旋转误差<15° 的比例；Tac_15：平移角误差<15° 的比例，越高越好）
- **场景实例分割（Identity）**：预测像素属于哪个物体实例（loss_identity，越低越好）

| 指标 | Wan v1 | V-JEPA2 v1 |
|------|--------|------------|
| depth loss（↓） | **0.334** | 0.322 |
| identity loss（↓） | 5.550 | **4.628** |
| Auc_30（↑） | 1.76% | **2.15%** |
| Rac_15（↑） | 7.49% | **7.97%** |
| Tac_15（↑） | 7.22% | **7.43%** |

**按场景来源分层**：

| 来源 | 模型 | depth | camera loss | identity | Auc_30 |
|------|------|-------|-------------|----------|--------|
| Infinigen（合成，149 samples） | Wan | 0.280 | 0.512 | 2.822 | 2.11% |
| Infinigen（合成，149 samples） | V-JEPA2 | 0.271 | 0.512 | 2.768 | 2.85% |
| ScanNet++（真实，804 samples） | Wan | 0.344 | 0.407 | 6.056 | 1.69% |
| ScanNet++（真实，804 samples） | V-JEPA2 | 0.331 | 0.398 | 4.973 | 2.03% |

**主要发现**：
- V-JEPA2 在 depth、identity 和所有相机精度指标上均略优于 Wan；
- 合成数据（Infinigen）相机估计比真实数据（ScanNet++）更准（camera loss 0.51 vs 0.41）但 identity 更容易（2.8 vs 6.1）；
- Auc_30 整体偏低（<3%），说明从冻结特征直接预测相机位姿是高难度任务。

---

## 四、过程中发现的问题和解决方案

### 问题 1：cos 相似度指标误导性强（C1 动作预测任务）
**现象**：Wan ctrl（扰乱特征）的 cos 相似度高达 97.5%，和真实 Wan v1（99.3%）只差 1.8%，单看这个指标感觉不明显。  
**原因**：模型学会了预测一个"平均特征"，它和所有目标的 cos 相似度都很高（类似于预测中心点）。  
**解决方案**：改用 R@1（在 batch 内做检索，排名第一才算正确），这才暴露出 ctrl ≈ 随机。

### 问题 2：A2 视角一致性的标签不平衡
**现象**：测试集 84% 是负样本，准确率指标被通胀了。  
**状态**：目前已发现此问题，需后续用更好的评估指标（如 AUC-ROC）重新评估。

### 问题 3：B1 的 GT 掩码"信息泄露"
**现象**：B1 完整版（v1）表现很好，但消融发现，一旦去掉 GT 掩码（nomask），效果立即跌回 baseline。  
**含义**：B1 其实是"带作弊"版，GT 掩码告诉模型精确的物体位置，使问题变得容易。  
**最新定义**：B2 不向 head 提供空间掩码或相机位姿，但用过去帧的
GT mask 构造 1D object query，以明确指定要预测哪个 object。GT mask
因此是 object-condition 构造工具，不应再描述为完全不使用 GT mask。

---

## 五、B2 探针：纯外观特征 → 空间位置（无信息泄露）

**B2 探针设计**：  
- 输入：整段视频的 VFM 特征（4帧） + 由过去 GT object mask 构造的 1D 外观 query；head 不接收空间 mask 或相机位姿。
- 输出：预测物体相对于最后一帧相机的方向（方位角 × 仰角的分类问题，16×8 = 128 个方向格子）+ 对数距离回归  
- 这才是真正的"仅凭外观特征推断空间位置"的探针，无任何泄露  

**B2 v1 训练完成**（50 epochs，Wan，节点 106 GPU 4–5）。最终验证集（n=951）指标：  

| 指标 | overall | infinigen (n=148) | scannetpp (n=803) | 随机基线 |
|------|---------|-------------------|-------------------|---------|
| 角度误差（°，↓） | **47.1** | 46.2 | 47.2 | ≈90° |
| 方位角误差（°，↓） | 41.2 | 49.2 | 39.7 | ≈90° |
| 仰角误差（°，↓） | 29.3 | 26.6 | 29.8 | ≈45° |
| Top-1（↑） | 9.5% | 12.2% | 9.0% | 0.78% |
| Top-3（↑） | 23.7% | 27.0% | 23.0% | 2.34% |
| log-distance err（↓） | 0.518 | 0.510 | 0.520 | — |

**解读**：  
1. Top-1/Top-3 比随机基线高 10–12 倍，说明 VFM 特征确实编码了一定的空间记忆信息，但绝对水平仍弱（角度误差 47°，相当于把物体定位到半个视野内）。  
2. 与 B1 v1（带 GT 掩码作弊：az 18.1° / el 11.4°）相比，B2 误差大约是 B1 的 2.5×；这正是 GT 掩码信息泄露的代价。  
3. B2 v1 与 B1 nomask（23.7° / 15.1°）差距也很明显，说明 query-token 形式重新引入了一些有用信号，但仍远不足以匹敌带掩码版本。  
4. infinigen vs scannetpp：infinigen 的 Top-1 略高（12% vs 9%），但 azimuth 误差更大（49° vs 40°）——可能与场景结构、可见 hint 数量分布差异有关。  
5. **结论**：现有 Wan VFM 在「仅凭外观签名做空间记忆」任务上能力有限，主要难点在 azimuth 估计；元素级精度（Top-1）虽然高于随机但绝对值仍低，是后续改进的主要方向。  

**B2 ctrl 对比消融**（scrambled VFM，50 epochs，节点 105 GPU 6 eval，n=952）：

| 指标 | **B2 v1** | **B2 ctrl** | v1 − ctrl | 随机基线 |
|------|-----------|-------------|-----------|---------|
| 角度误差（°，↓） | **47.1** | 59.3 | −12.2 | ≈90° |
| 方位角误差（°，↓） | **41.2** | 52.9 | −11.7 | ≈90° |
| 仰角误差（°，↓） | **29.3** | 35.2 | −5.9 | ≈45° |
| Top-1（↑） | **9.5%** | 4.0% | +5.5 pp（2.4×） | 0.78% |
| Top-3（↑） | **23.7%** | 11.9% | +11.8 pp（2.0×） | 2.34% |
| log-distance err（↓） | **0.518** | 0.623 | −0.105 | — |
| val/loss（↓） | **3.756** | 4.549 | −0.79 | — |

**对比消融的关键解读**：

1. **VFM 表征确实贡献了空间记忆信号**。v1 与 ctrl 的差距非常清晰：角度 −12°、Top-1 提升 2.4×、Top-3 提升 2×、val/loss 降 0.79。这说明 v1 的提升不是任务本身的偏差（比如 query token 含有物体类别提示、或隐藏物体位置分布偏背后象限），而是真实 VFM 表征贡献的。

2. **ctrl 仍带一点信号**。ctrl Top-1 = 4.0% ≈ 5× 随机基线（0.78%），说明部分提升来自任务/数据集本身的偏差（同一样本内 query 与 patch 共享 scramble 种子，仍可学到「样本特异的虚假匹配」；隐藏物体方位分布也不均匀）。但这部分偏差只能解释 ctrl-vs-random 的差，不能解释 v1-vs-ctrl 的差，因此 v1 的「真实信号」是 robust 的。

3. **infinigen vs scannetpp 的错位在 ctrl 上也保留**：infinigen Top-1（6.1%）> scannetpp（3.6%），印证了第 5 节解读 #4 中提到的「场景方位分布偏差」假设——这是任务本身的统计偏差，与 VFM 是否真实无关。

4. **距离信号有限但存在**：v1 logd_err = 0.518 vs ctrl = 0.623，差距 0.10 nat（约 11% 距离偏差减少），说明深度信号确实存在但作用比方向信号弱。

---

## 六、数值汇总表

### C1 动作预测（越高越好，随机基线=6.25%）
| 模型 | R@1 |
|------|-----|
| Wan v1 | 26.7% |
| Wan ctrl | 5.8% |
| V-JEPA2 v1 | 22.8% |

### A2/A3 新探针结果

#### A3 异常检测（pair_acc，随机基线=25%）

| 模型 | pair_acc | mean_delta |
|------|---------|----------|
| Wan v1（dtype+seed 双修复） | **86.3%** | 0.811 |
| Wan ctrl（dtype+seed 双修复） | **16.5%** | 0.002 |
| V-JEPA2 v1 | 97.5% | — | 旧结果（dtype-fix 前），重训中 |

> ctrl pair_acc=16.5%≈随机基线（25%），mean_delta≈0，与 v1 差距（86.3% vs 16.5%）完全由真实时序信号贡献。先后发现并修复 dtype-leak（Bug 1）和 seed-leak（Bug 2），共经历 3 次重训。V-JEPA2 v1 97.5% 来自早期日志 val_acc，尚未用 pair_acc 复评，且受两个 bug 影响，需重训。

#### A2 视角一致性（注意：负样本占84.3%，简单基线=84.3%）
| 模型 | overlap_acc | overlap_mae |
|------|------------|-------------|
| Wan v1 | 85.6% | 0.143 |
| Wan ctrl | 78.7% | 0.180 |
| V-JEPA2 v1 | 85.2% | 0.142 |

### C2/C3 新增探针 5 epoch 结果（1000 train steps）

> 这是 C2/C3 第一版完整 5 epoch 实验：四个实验均从 `epoch=0-step=200` 自动续训到 `epoch=4-step=1000`，并在 val split（n=953）上重新评估。它仍不是 50 epoch 长训；主要瓶颈是大 VFM/target feature 的 I/O 和 CPU 张量处理。

#### C2 Path Integration（越低越好，检索指标越高越好）
| 模型 | final_pose_error | drift_rate | mean_step_pose_error | global_R@1 | global_R@5 |
|------|------------------|------------|----------------------|------------|------------|
| Wan v1 | **2.508** | **0.563** | 1.615 | 0.14% | 0.63% |
| V-JEPA2 v1 | 2.514 | 0.568 | **1.591** | **0.17%** | **0.73%** |

**初步解读**：5 epoch 后两个模型的 pose-error 已明显低于 epoch0，说明 Path Integration probe 确实开始学习轨迹积分；Wan 与 V-JEPA2 的 final_pose_error 基本打平，V-JEPA2 的 mean_step_pose_error 略好。检索 R@1/R@5 仍很低，说明 target feature 检索结构尚未稳定形成，后续需要更长训练或改进特征压缩/采样策略。

#### C3 Counterfactual（越高越好）
| 模型 | n_valid_interventions | counterfactual_consistency | intervention_validity | intervention_margin | global_R@1 |
|------|-----------------------|----------------------------|-----------------------|---------------------|------------|
| Wan v1 | 1527 | **0.9890** | 41.7% | **-0.00016** | 0.20% |
| V-JEPA2 v1 | 1509 | 0.8659 | **41.9%** | -0.00108 | **0.53%** |

**初步解读**：Counterfactual 的 cosine consistency 已从 epoch0 的近 0 提升到很高，说明 probe 学会了生成与目标特征空间相容的 counterfactual 表征；Wan 的 consistency 更高，V-JEPA2 的 retrieval R@1 更高。intervention_validity 仍约 42%，intervention_margin 仍接近 0，说明模型还没有稳定学到“干预后目标应比原目标更匹配”的判别性结构，当前更像学到了特征空间平滑/平均化，而非强 counterfactual 推理。

### A1 基线探针（来自 VidFM3D，100 epochs）

| 指标 | Wan v1 | V-JEPA2 v1 | 说明 |
|------|--------|------------|------|
| depth loss（↓） | 0.334 | **0.322** | 深度预测误差 |
| identity loss（↓） | 5.550 | **4.628** | 实例分割误差 |
| Auc_30（↑） | 1.76% | **2.15%** | 旋转误差<30° AUC |
| Rac_15（↑） | 7.49% | **7.97%** | 旋转误差<15° 比例 |
| Tac_15（↑） | 7.22% | **7.43%** | 平移角误差<15% 比例 |

### B1 隐藏物体位置预测（误差越小越好）
| 实验 | 方位角误差 | 仰角误差 |
|------|----------|---------|
| Wan v1（完整） | 18.1° | 11.4° |
| Wan ctrl | 23.4° | 15.1° |
| Wan nomask | 23.7° | 15.1° |
| Wan poseonly | 24.3° | 15.7° |
| V-JEPA2 v1 | 17.6° | 11.7° |

### B2 隐藏物体位置预测 v2（无 GT 掩码、无相机位姿；query-token 形式）
| 实验 | 角度误差 | 方位角误差 | 仰角误差 | Top-1 | Top-3 | logd_err |
|------|---------|-----------|---------|------|------|---------|
| Wan v1（overall） | **47.1°** | **41.2°** | **29.3°** | **9.5%** | **23.7%** | **0.518** |
| ↳ infinigen | 46.2° | 49.2° | 26.6° | 12.2% | 27.0% | 0.510 |
| ↳ scannetpp | 47.2° | 39.7° | 29.8° | 9.0% | 23.0% | 0.520 |
| Wan ctrl（overall） | 59.3° | 52.9° | 35.2° | 4.0% | 11.9% | 0.623 |
| ↳ infinigen | 56.1° | 51.8° | 35.0° | 6.1% | 16.2% | 0.626 |
| ↳ scannetpp | 59.9° | 53.1° | 35.2° | 3.6% | 11.1% | 0.622 |
| 随机基线 | ≈90° | ≈90° | ≈45° | 0.78% | 2.34% | — |
