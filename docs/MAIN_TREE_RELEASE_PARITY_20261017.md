# 以主目录为发布代码：补回 owner overlay 与标签路径，并用 canary 核对与冻结包一致 — 2026-10-17

分支 `release-main-overlay-20261017`（基于 `release-prep-20261017` 的 `118a832`），只在本地，未 push。
配套文档：[代码版本对齐](CODE_ALIGNMENT_20261017.md)、[正式评估方案](EVALUATION_PLAN_20261017.md)。

## 结论

- **主目录现在可以作为发布代码。** 本分支 `cb87839` 原样补回 owner overlay，并补回冻结包的全部标签与取样路径，
  同时保留主目录后来的修复与新功能。
- **样本级一致（本次实际执行通过）。** 52 个评估项覆盖 9 个模型 × 6 种探针和两个训练包。冻结包与主目录的
  配置、完整样本列表，以及 1,560 个抽样样本的全部共有字段，逐位相同。
- **预测级一致（本次实际执行通过）。** 同样 52 项、同一批 checkpoint：25 项完全相同；18 项（C1、C2）预测与
  全部共有指标逐位相同；9 项（A1 mask）的差异在 GPU 重跑噪声内，与冻结包自己跑两次的差异同量级。
  没有一项判为不同。
- **发现一个数据问题。** VideoMAE 前缀 24 的 8 个 C1/C2 run 需要的 2,724 个旧目标特征文件已不在 AOSS 上，
  本地也没有，这 8 个 run 现在不能评估也不能重训。其余 668 项正式评估已通过验收。
- **还有一个发布缺口。** 74 个测试文件里有 48 个依赖从未提交的文件，只含已提交内容的树跑不通自己的测试。
- 所有 commit 只在本地，未 push，未合并进 `release-prep-20261017`。

## 一、补回了什么

选择“补回 owner overlay”，没有选“证明 owner catalog 等价”。原因：两者格式不同
（overlay 是 SHA 绑定的 SQLite 两表，catalog 是 JSON），而且服务器上没有任何 catalog 文件可供对照。

| 内容 | 来源 | 做法 |
|---|---|---|
| `vidfm3d/utils/probe_owner_overlay.py` | `code/main_v2_probe_v2_r7_single_v3`（与 Aether 包逐字节相同） | 原样复制 |
| `vidfm3d/utils/aether_temporal.py` | `code/main_v2_probe_v2_aetherfix_r1` | 原样复制 |
| 数据加载器 `inscene15k_dataset.py` | 主目录、r7 包、Aether 包三方合并 | 见下 |
| `probe_ext_module.py` 的 `transfer_batch_to_device` | r7 包 | 复制；另保留 `scene_path`，让非有限输入报错时能写出场景；字段不全的批次原样传输（冻结包会报错） |

数据加载器的合并方法：以 `2f19ecb` 为共同祖先，`git merge-file` 合并主目录与 r7 包（21 处冲突逐处处理），
再用同样方法合入 r7→Aether 包的差异。合并结果与 Aether 包加载器相比，只多出主目录的这些改动：

- owner catalog（`INSCENE_OWNER_CATALOG`）保留；与 overlay 同时设置时直接报错，二者只能绑定一个；
- 划分名校验（只接受 train/val/test/all）与几处注释；
- C 探针快速路径：在冻结包的输出之外，再输出主目录评估要用的字段（相机位姿、时间间隔、
  锚点的独立特征及其有效标记、样本编号）。

冻结包独有、现已补回的路径：owner overlay 的流式与目标特征解析、标签缓存（`INSCENE_LABEL_CACHE_ROOT`）、
场景元数据缓存（`INSCENE_METADATA_CACHE_MAX_SCENES`）、按探针只加载所需数据的快速路径（`INSCENE_PROBE_TASK`）、
A2 的 overlap 专用取样、B 探针的精简路径，以及 Aether 的时间对齐修复。

**唯一有意改变的实现**：C 探针的标签缓存。冻结包为 C1/C2/C3 各缓存一种动作标签；合并版改为缓存
参考帧下的相机位姿（C3 另加锚点重叠率），所有动作标签与评估位姿都由它算出。这样缓存命中时也能给出评估字段，
命中与重算结果逐位相同（有单元测试）。缓存任务名改为 `c_probe_geometry`，不会误读冻结包留下的旧缓存。

## 二、测试

本次实际执行通过。新增 7 个测试（`tests/test_owner_overlay_label_cache.py`）：

- overlay 精确解析流式与目标 owner，载荷大小、头部与全文哈希逐项校验；改动载荷或数据库哈希都会报错；
- overlay 与 owner catalog 同时设置时报错；
- overlay 下加载流式前缀特征、目标特征与锚点独立特征（锚点缺失时记为无效，不报错）；
- C1、C2 的标签缓存命中与重算，全部输出逐位相同；
- 传输过滤只把探针用到的张量交给 GPU，字段不全的批次原样传输。

另改了两个已有测试：Aether 时间对齐测试改为导入主目录模块（原先读取被 gitignore 的 `runtime/`），
C2 快速路径测试补上新属性。

全套测试在两个只含已提交内容的 worktree 上对照运行（登录节点，`--continue-on-collection-errors`，
未提交的辅助代码通过 `PYTHONPATH` 指向主 checkout）：

| 树 | 通过 | 失败 | 收集错误 |
|---|---:|---:|---:|
| 基线 `118a832` | 414 | 44 | 6 |
| 本分支 `cb87839` | 421 | 44 | 6 |

两边的失败与收集错误逐条相同，多出的 7 个通过就是新增测试。这 50 项失败与改动无关：已提交的测试引用了
从未提交的文件。74 个测试文件中有 48 个引用 `analyses/`、`scripts/`、`runtime/` 或 `artifacts/` 下
未提交的内容；主 checkout 里全部通过，是因为那些文件就放在那里。这是发布前要补的缺口，见“待你决定”。

## 三、样本级一致性（数据加载器）

本次实际执行通过（8 个 SCO 作业，2026-10-17 12:07–12:49 UTC）。

**范围**：52 个评估项，每个模型 × 探针取一个（9 个模型，A1 depth、A1 mask、A2、B2、C1、C2；
Wan 没有 A2，Aether 没有 B2）。47 项的训练包是 `r7_single_v3`，5 项是 `aetherfix_r1`。

**做法**：每项在两个独立进程里构建训练集与验证集，一个用冻结训练包（先按 `source.sha256` 校验），
一个用主目录（先按冻结的 306 个代码文件 SHA-256 清单校验）。两边用同一份训练环境（owner overlay、
manifest、fp16 设置）和各自新建的标签缓存。验证集取 16 个、训练集取 8 个分散的样本（训练集固定随机种子）。

| 比较内容 | 结果 |
|---|---|
| Hydra 组合后的 data 与 model 配置 | 52/52 相同 |
| 完整有序样本列表 | 52/52 相同（验证集共 19,320 个、训练集共 127,140 个样本身份） |
| 逐样本、逐字段（浮点按原始位比较） | 1,560 个样本、15,510 次字段比较全部相同；加载错误 0 |
| 只有冻结包有的字段 | 无 |

只有主目录有的字段都在 C 探针上：C1 多出锚点独立特征及其有效标记、两个审计编号和整段前缀特征；
C2 另多出相机位姿、时间间隔与反事实字段。这些是主目录评估要用的字段，冻结包的训练不读它们。

结果：`artifacts/main_v2_probe_v2_816_20260917/loader_parity_main_vs_pack_20261017/<key>/PARITY.json`。

## 四、预测级 canary（同一 checkpoint，主目录评估 vs 冻结包评估）

本次实际执行通过（26 个单卡作业，同样 52 项，12:15–12:49 UTC，0 失败）。主目录 `cb87839` 用 v2 评估器
评估冻结包评估过的同一批 checkpoint（SHA-256 逐项相同），再与冻结包评估（`eval_v2_canonical_20261017_r2`）
逐样本比较全部记录字段与 summary 指标。

| 结论 | 项数 | 探针 | 说明 |
|---|---:|---|---|
| 完全相同 | 25 | A1 depth、A2、B2 | 所有逐样本记录与全部指标逐位相同 |
| 预测相同 | 18 | C1、C2 | 预测、目标、主指标与全部共有指标逐位相同；差异只在下面两处 |
| 在重跑噪声内 | 9 | A1 mask | 逐样本 identity loss 最大差 2.3e-4，平均 loss 最大差 3.3e-6 |

C1、C2 的两处差异：

- 冻结包的 C 快速路径不输出审计编号，评估器给所有记录填了同一个占位值（C1 为 1，C2 为 -1）；
  主目录给出真实的窗口号与末帧号。样本编号两边相同。
- 主目录多给出锚点独立特征，所以 C1 多出“复制锚点”对照指标（`copy_anchor_isolated_*`，共 56 个键）。

A1 mask 的差异不是代码造成的：identity loss 用 CUDA `scatter_add_` 汇总类原型，加法顺序不固定。
同一个 V-JEPA2 checkpoint 用冻结包跑两次（r1 canary 与 r2 正式），逐样本最大差 2.1e-5；主目录对冻结包是 3.2e-5，
量级相同。A1 depth 不经过这一步，9 项全部逐位相同。

| 模型 | A1 mask 逐样本最大差 | 平均 loss 差 |
|---|---:|---:|
| Aether | 2.3e-4 | 1.4e-6 |
| CogVideoX | 1.8e-4 | 1.1e-7 |
| Wan | 1.2e-4 | 3.3e-6 |
| LingBot | 9.3e-5 | 1.5e-6 |
| V-JEPA2 | 3.2e-5 | 2.2e-7 |
| Qwen2.5-VL | 2.6e-5 | 6.4e-8 |
| Fast3R | 2.3e-5 | 7.7e-8 |
| DINO | 1.8e-5 | 9.7e-8 |
| VideoMAE | 1.5e-5 | 2.2e-7 |

比较规则写在 `scripts/eval_main_tree_canary_20261017.py compare` 里：只有冻结包未输出的字段、只能由主目录新字段
算出的指标，以及绝对差不超过 1e-3 的浮点差，才不算“不同”；其余任何差异都判为 DIFFERENT。
结果：`artifacts/main_v2_probe_v2_816_20260917/eval_v2_main_tree_canary_20261017/COMPARISON.json`。
评估作业按 `CAMPAIGN.json` 绑定 `cb87839` 的 306 个代码文件；比较脚本随后在下一个 commit 里加入上述分类规则，
所以若要重跑评估作业，需要先用 `freeze --force` 重新绑定。

## 五、数据可用性（正式评估中发现）

本次实际执行通过的只读检查（`artifacts/release_audit_20261017/aoss_payload_probe/`）。

- r2 正式评估有 8 项失败，全是 VideoMAE 前缀 24 的 C1 与 C2（层 5、11、17、23）。它们的 overlay 把部分目标特征
  指向旧目录 `aoss://baiqiao/InsScene-15K/FEAT_TARGET/videomae/scannetpp/<场景>/feature_layer<L>.sft`。
  这些对象 9 月训练时存在，overlay 记录了远端全文 SHA 核对通过。
- 676 个评估项的 overlay 能解析到 357,164 个载荷对象。其中旧目录只有这一处：2,724 个对象（681 个场景 × 4 层）。
  探测结果：**2,724 个全部不在 AOSS 上，本地镜像也没有**。按场景划分，训练集 2,448 个、验证集 276 个。
  所以这 8 个 run 现在既不能评估，也不能重训。
- 规范命名空间随机抽查 3,000 个对象，63 个不在 AOSS 上（VideoMAE 60、Wan 3），但 63 个都在本地镜像
  `/mnt/umm/users/yinbaiqiao/InsScene-15K/` 里。overlay 先查本地镜像，这些对象可用。
- 没有找到删除或迁移这些对象的记录。未查：其余 354,164 个对象（只抽查了 3,000 个）、节点本地缓存里是否还有副本。

## 六、复现命令（服务器，`/mnt/umm/users/yinbaiqiao/probe_spatial`）

主目录代码在 worktree `.worktrees/release-main-overlay`（分支 `release-main-overlay-20261017`）。
评估与一致性检查的状态写在主 checkout 的 `artifacts/` 下。

```bash
PY=/mnt/umm/users/yinbaiqiao/.conda/envs/probe-spatial/bin/python
W=.worktrees/release-main-overlay
C=artifacts/main_v2_probe_v2_816_20260917/eval_v2_main_tree_canary_20261017
# 把 canary 绑定到一个 commit（登录节点；生成 306 个代码文件的 SHA-256 清单）
$PY -B $W/scripts/eval_main_tree_canary_20261017.py freeze --keys-file $C/keys.txt
# 样本级一致性：8 个作业，每个作业逐项构建冻结包与主目录的数据集并逐位比较
$PY -B $W/scripts/check_loader_parity_main_vs_pack_20261017.py submit --items $(cat $C/keys.txt) --shards 8 \
    --out artifacts/main_v2_probe_v2_816_20260917/loader_parity_main_vs_pack_20261017 --campaign-json $C/CAMPAIGN.json --submit
# 预测级 canary：26 个单卡作业，每个 2 项
$PY -B $W/scripts/eval_main_tree_canary_20261017.py submit --per-job 2 --submit
# 与冻结包评估逐样本比较，写 $C/COMPARISON.json
$PY -B $W/scripts/eval_main_tree_canary_20261017.py compare
```

## 待你决定

1. **合并分支**：(a) 推荐：把 `release-main-overlay-20261017` 在本地合并进 `release-prep-20261017`，不 push；
   (b) 先保持独立分支。
2. **缺失的 VideoMAE 目标特征**：(a) 推荐：先只读查找副本（其他 AOSS 命名空间、节点本地缓存）；找不到再用冻结的
   提特征代码重提，按 overlay 记录的 SHA 核对，字节不同则这 8 个 run 不进发布结果；(b) 直接把这 8 个 run 排除。
3. **测试依赖未提交文件**：(a) 推荐：发布分支只带发布测试，并补交它们需要的模块；内部战役测试留在内部；
   (b) 把 `analyses/` 等目录全部提交。
4. **仍未决定（上一轮）**：timestep 结果的评估方式，推荐为 `opt_v3` 与 `pack_v4_1` 建评估包。
