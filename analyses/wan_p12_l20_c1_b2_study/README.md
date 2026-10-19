# Wan P12/L20 C1/B2 Validity Study

这个目录是两套已完成 validity analysis 的统一入口。它不复制或改写冻结的实验实现，而是把问题定义、canonical code、运行顺序、输入输出和综合报告组织在一起。

## 一句话理解

| 分析 | 它在问什么 | 主要控制 | 能支持什么结论 |
| --- | --- | --- | --- |
| Cross-objective C1 | 正确动作是否帮助预测未来特征，且这种效应是否随预训练目标变化？ | zero、shuffle、matched action、same-scene candidates、direction flips | 跨模型的 action sensitivity 和 objective-dependent differences |
| Cross-objective B2 | hidden-object readout 使用了历史、正确 query 或帧顺序中的哪一种信息？ | query mask/swap、current frame、remove-visible、shuffle、reverse、priors | history-content、query-addressing 和 ordered-memory 机制 |
| Wan P12 construct validity | Wan P12/L20 的 C1 正结果是否能通过更严格的构念效度门？ | 100 个 formal shuffles、全部 275 个 wrong actions、last-state/prefix copy、breadth gates | Wan 单模型的论文主结论 |
| Integrated synthesis | C1 与 B2 联合起来最支持哪一种机制？ | 比较两套已冻结结果，不重新拟合 | weak action sensitivity vs. structured dynamics 的最终解释 |
| Representation geometry bridge | Wan 的 copy gate 是否受 feature temporal smoothness 影响？ | 历史 P12 raw/centered cosine 与 retrieval summary，不重放模型 | 将 copy-gate 失败解释为高平滑性加上尚未可靠的增量 action signal |

## 数据流

```text
fixed checkpoints + P12 features + manifests
                |
        +-------+--------------------+
        |                            |
        v                            v
cross-objective C1/B2          Wan P12 strict C1
replay -> sample dumps         prepare -> GPU interventions
       -> summarize                    -> summarize
        |                            |
        v                            v
c1_validity.csv /              summary.json /
b2_validity.csv                construct-validity report
        |                            |
        +-------------+--------------+
                      v
             integrated report
```

## 从这里开始

```bash
PY=/mnt/umm/users/yinbaiqiao/.conda/envs/probe-spatial/bin/python

# 只读：查看研究结构
$PY -m analyses.wan_p12_l20_c1_b2_study describe

# 只读：检查代码、结果文件和关键证据是否完整
$PY -m analyses.wan_p12_l20_c1_b2_study check

# 只打印将要运行的命令
$PY -m analyses.wan_p12_l20_c1_b2_study --dry-run c1b2-replay --model wan

# 从现有 CSV/JSON 重建综合报告
$PY -m analyses.wan_p12_l20_c1_b2_study integrate
```

现有正式结果目录非空，prepare 和 GPU runner 都有防覆盖保护。不要在当前正式输出上直接重跑；需要复现实验时应使用新的隔离输出目录或完整复制工作区。

## 完整运行顺序

### A. Cross-objective C1/B2

```bash
# 每个模型同时 replay C1 和 B2；固定模型集合为
# vjepa2, videomae, wan, cogvideox, f3r
$PY -m analyses.wan_p12_l20_c1_b2_study c1b2-replay --model all --task all

# 读取 PASS sample dumps，生成 CSV、图和 PROBE_VALIDITY_REPORT.md
$PY -m analyses.wan_p12_l20_c1_b2_study c1b2-summarize
```

Replay 只读取固定 checkpoint 和 feature tensors，不训练 probe，也不提取 backbone features。F3R 是 geometry-specialist pipeline control，不参与 core objective-family verdict。

### B. Wan P12/L20 strict construct validity

```bash
# CPU：冻结输入 hash、样本清单、动作 strata 和 100 个 permutation mappings
$PY -m analyses.wan_p12_l20_c1_b2_study p12-prepare

# GPU：复现 frozen evaluator，并计算 null、copy、100 shuffles 和 276x275 wrong actions
$PY -m analyses.wan_p12_l20_c1_b2_study p12-run

# CPU/GPU node 均可：scene bootstrap、permutation test、breadth gates、表格和图
$PY -m analyses.wan_p12_l20_c1_b2_study \
  p12-summarize --job-id <execution-id> --tests-passed
```

`p12-run` 当前委托给 legacy 文件名 `scripts/run_wan_p12_l20_action_validity_sco.sh`。它实际是通用单 GPU wrapper，可以在直接 SSH 的 H800 节点运行；`sco` 仅是保留的历史命名。

### C. Integrated synthesis

```bash
$PY -m analyses.wan_p12_l20_c1_b2_study integrate
```

该阶段只读取 P12 `summary.json`、`condition_metrics.csv` 和 C1/B2 validity CSV，通过模板重建综合报告，不执行新的统计拟合或模型推理。

## Canonical Code Map

### Cross-objective C1/B2

- `analyses/probe_validity/run_probe_validity.py`
  - `replay_c1()`：固定 C1 checkpoint，构造 action interventions 和 candidate pools。
  - `replay_b2()`：固定 B2 checkpoint，构造 history/query/order perturbations。
  - `run_phase2_replay()`：按模型执行 C1/B2 replay 并写 sample dumps。
  - `_phase2_c1_rows()`：按 candidate pool 和 motion/drift strata 聚合 C1。
  - `_phase2_b2_rows()`：对 B2 paired angular-error deltas 做聚合。
  - `run_phase2_summarize()`：生成最终 CSV、图和跨模型报告。

这个文件较长，因为它同时包含多模型固定 contract、样本级 replay、候选池定义、统计聚合和绘图。统一 CLI 不改变这些逻辑。

### Wan P12/L20 construct validity

- `scripts/prepare_wan_p12_l20_action_validity.py`
  - 校验 manifest/checkpoint/evaluator hashes。
  - 固定 276 samples、69 scenes 和 12 个派生 action strata。
  - 生成 100 个 exact-marginal、scene-independent derangements。
- `scripts/run_wan_p12_l20_action_validity.py`
  - 严格复现 frozen prediction 和 retrieval ranks。
  - 计算 correct、eval-null、last-state copy、prefix copy。
  - 运行 27,600 formal-shuffle pairs 和 75,900 exhaustive wrong-action pairs。
- `scripts/summarize_wan_p12_l20_action_validity.py`
  - 执行 10,000 次 scene bootstrap 和 permutation test。
  - 检查 copy、action-class breadth、scene breadth 和 concentration gates。
  - 生成 tables、figures、summary.json 和论文主报告。
- `vidfm3d/utils/action_validity.py`
  - 两阶段共用的 retrieval、action descriptor、derangement 和 bootstrap 基础函数。

### Integrated report

- `build_integrated_report.py`：从结构化结果中选择 Wan P12/L20 evidence。
- `integrated_report_template.md`：将统计代码与论文叙述分开。
- `study_manifest.json`：机器可读的研究问题、入口、输出和 protected files 清单。
- `analyses/probe_validity/build_representation_geometry_bridge.py`：只读重建历史 P12 的 raw/centered temporal cosine、retrieval proxy 与边界说明；它不重训或重放模型。

## 控制条件为什么不同

不同 C1 control 回答不同强度的问题：

1. `eval-null` 检查 probe 是否使用 action 输入。
2. 普通或 formal shuffle 检查正确 action 是否优于一般错误 action。
3. `matched-magnitude` 检查效应是否只来自动作幅值。
4. `same-scene` 检查在视觉内容相近的 hard negatives 中是否仍有动作 specificity。
5. `last-state copy` 检查 probe 是否真正超过状态持久性。
6. exhaustive wrong actions 和 breadth gates 检查效应是否只来自少数 donor、类别或场景。

因此，correct 优于 shuffle 并不自动等于通过 construct validity。Wan 的正式结果是 shuffle、all-wrong 和 breadth gates 通过，但 copy gate 失败，最终解释为 `WEAK_ACTION_SIGNAL`。

B2 的逻辑与 C1 互补：current-frame、remove-visible 和 query-only 测试历史内容贡献；query mask/swap 测试 query addressing；shuffle/reverse 测试 ordered temporal memory。

## 当前论文结论

> Wan P12/L20 包含可复现但较弱的动作敏感信号。B2 显示其依赖历史中目标可见内容，但没有可靠的帧顺序或正确 query 依赖。联合证据更符合 history-content retrieval 上叠加有限动作调制，而不足以证明结构化的 action-conditioned dynamics representation。

正式报告：

- `reports/wan_p12_l20_action_dynamics_construct_validity.md`
- `analyses/probe_validity/PROBE_VALIDITY_REPORT.md`
- `reports/wan_p12_l20_c1_b2_integrated_analysis.md`
