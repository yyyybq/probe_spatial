# Canonical V2 特征位置复核（2026-09-22 UTC）

结论：`GAP_DELTA_V1` 确实承载十模型主线的补提取结果。此前将它排除、只查无后缀 `CANONICAL_V2` 的结论错误。Wan/Aether 最新 streaming 输出路径正确；两者的所需 streaming 文件已经能在 AOSS 或本地找到，但上传状态不同。Aether target 另有失败分片，不能称所有模式都完成。

## 复核范围与方法

- 使用当前 AOSS `[fj]` 配置，对 `baiqiao/InsScene-15K/` 下六个 Canonical V2 根进行真实、完整分页列举；共 28 个模型分支、965,148 个对象。
- 按目录、模型、特征文件名计数，记录对象大小和修改时间；没有下载特征张量正文。
- 将 Wan/Aether 的 87,072 个正式 streaming 请求逐项与 AOSS 清单比对；缺失对象再查本地同路径及 manifest。
- 抽取四个 streaming 索引样本比较 canonical manifest，indices/window_indices 均匹配；这不是全量标注验证。
- 查阅最新任务提交、提取日志、上传日志、完成回执；只读查询两个 SCO 任务状态。
- Aether 正在持续上传，因此计数是约 07:18–07:25 UTC 的过程快照，非同一时刻原子快照。

## 三组目录的含义

共同前缀：`fj:s3://baiqiao/InsScene-15K/`。

| streaming 根（target 有对应同名根） | 当前实际内容 | 对主线的含义 |
|---|---|---|
| `FEAT_STREAMING_CANONICAL_V2` | Wan：12 场景；t499、layer15/20/25 各 192 文件。target 仅 3 场景 | 8 月 30 日早期 V2 留存；抽样含 canonical 协议但无 seeded 噪声字段，不能代表当前主线全集 |
| `FEAT_STREAMING_CANONICAL_V2_SEEDED_V1` | Wan、CogVideoX、MiniMax-H3，三者各 681 场景。各两个 timestep × 三层，streaming 每组 10,884 文件、target 每组 681 文件 | 确定性噪声版本的提取输出；当前物理内容主要是三模型 timestep campaign。它也被公共 protocol/manifest 写为默认根，但这不能覆盖后续 gap 任务的显式输出根 |
| `FEAT_STREAMING_CANONICAL_V2_GAP_DELTA_V1` | Aether、CogVideoX、DINO、F3R、LingBot、MiniMax-H3、Qwen2.5-VL、VideoMAE、V-JEPA2、Wan | 主线 canonical gap/re-extraction 输出。9 月 21 日 Wan/Aether 正式补提取明确写到这里 |

三个目录不能仅按名称理解为互斥实验等级。Canonical 协议定义样本/帧对齐，命名空间还区分提取批次及增量输出。同属 V2 不表示 timestep、layer 和样本内容可以任意替代。

## GAP streaming 实物清单

下面仅是 GAP 根内对象数，不是合并历史可复用来源后的就绪结论。每一完整 layer/config 的 canonical streaming 应有 10,884 个对象。

| 模型 | 文件组 | 每组对象数 |
|---|---|---:|
| Wan | layer0/10 | 各 9,942 |
| Wan | layer5/15/25/29 | 各 9,941 |
| Wan | layer20 | 10,818 |
| Aether | layer1 | 8,269，仍在上传 |
| CogVideoX | layer20 | 6,410 |
| CogVideoX | 其余八层 | 各 4,108 |
| DINO / F3R / Qwen2.5-VL | 各自单层 | 各 6,410 |
| LingBot / MiniMax-H3 / VideoMAE | 各五层 | 各 4,108 |
| V-JEPA2 | layer23；其余四层 | 4,109；各 4,108 |

完整文件名、日期和对应 target 数量见 [inventory.tsv](inventory.tsv)；原始逐对象清单为同目录下 `.jsonl.gz` 文件。

## Wan：streaming 提取完成，上传失败

- 最新任务：`pt-7fbth6la`，2026-09-21 10:00 UTC 提交，13:30:26 UTC 以 `FAILED` 结束。
- 本轮 16 个提取进程退出码全部为 0；上传退出码为 1。
- 正式要求 76,188 个 streaming 特征，AOSS 实有 70,466 个；其余 **5,722 个全部在本地同名 GAP 路径，且均有配套 manifest**。文件存在性意义上的 remote/local 联合覆盖为 76,188/76,188。
- 上传日志末尾反复报 `before snapshot final atomic temp files remain; safely deferring`，最后 `final sync failed 8 times; giving up`。本轮 done-index 为 0 条，没有 remote completion receipt。
- 本地 Wan 子树发现 5 个旧 `.tmp-*` 文件，修改日期为 9 月 9–16 日。它们早于本轮提取。实际 `atomic_temps.txt` 对当前账号不可读，因此未声称这 5 个全部都是本轮阻断集合。
- AOSS 此配置最新特征修改时间仍为 9 月 17 日，也支持 9 月 21 日这一轮未完成发布。
- target 的最新 8 个分片回执均为 PASS，AOSS 各正式层有覆盖 681 场景的 delta 对象；这是补提取任务完成证据，仍需按实际 target row/index 读取，不能只用场景数代替全量行验证。

本地 root：`/mnt/umm/users/yinbaiqiao/InsScene-15K/FEAT_STREAMING_CANONICAL_V2_GAP_DELTA_V1/wan/scannetpp/`。

关键回执：`/mnt/umm/users/yinbaiqiao/InsScene-15K/FEAT_STREAMING_CANONICAL_V2_GAP_DELTA_V1/.windowfix_v2_gap_logs/wan/wa-hole-002-wan-749-streaming_prefix/20260921T100007.598562Z-p24/receipt.json`。

## Aether：streaming 提取完成，上传仍进行；target 尚有失败

- 最新 streaming 任务：`pt-vyz569ze`，查询状态 RUNNING；16 个提取日志都已结束且报告 Failed: 0，当前在上传。
- AOSS 扫描快照有 8,269/10,884 个正式 streaming 文件。缺少的 2,615 个中，2,609 个在本地且带 manifest；另外 6 个在查询期间刚上传并从本地清理，再次 HEAD 均存在于正确的 GAP AOSS 路径。
- 所需 streaming 文件均已找到，不支持“提取遗漏或写到另一个 namespace”的猜测；AOSS 尚未全量可读。
- target AOSS 只有 170 个 layer1 delta tensor（另有索引/sidecar 残留使目录计数为 172）。最新 8 个 target 分片回执中 2 PASS、6 FAILED；失败分片提取进程被终止（-15），上传退出 1。
- 抽查失败 target p02 的上传日志为清理 `.tstep_sync_staging_0033_97/...` 时 `Directory not empty`，随后 runner 因上传失败终止提取。此错误仅已核实 p02，未将它推广为其余五个失败分片的根因。
- 因此可以确认 Aether streaming 已提取齐，但不能确认 Aether target 已提取齐。

## 应该如何定位最新主线

正式 streaming 路径：

```text
fj:s3://baiqiao/InsScene-15K/FEAT_STREAMING_CANONICAL_V2_GAP_DELTA_V1/
  {wan|aether}/scannetpp/{scene}/window_{w}/prefix_{tail}/feature_t749_layer{layer}.sft
```

场景索引：同场景目录的 `prefix_index.npy`。prefix 8 对应 tail 7，依此类推。

正式 target 增量路径：

```text
fj:s3://baiqiao/InsScene-15K/FEAT_TARGET_CANONICAL_V2_GAP_DELTA_V1/
  {model}/scannetpp/{scene}/.canonical_v2_target_delta/{worklist_hash}/
    feature_t{t}_layer{layer}.delta.sft
    target_indices_layer{layer}.npy
```

target 不是简单的场景根 `feature.sft + target_indices.npy`。同一层可有多个 worklist 的分片，例如 Wan layer20 有 686 个 delta 文件，不能将 686 当作 686 个场景，也不能任选一个分片代表整场景。

脚本证据：

- [正式 Wan/Aether 请求路径](../../scripts/build_probe_wan_aether_gate_requests.py:31)
- [最新提取任务的 namespace 和本地 root](../../scripts/build_probe_wan_aether_zoetrope_extract.py:119)
- [9 月 21 日 8 卡提交记录](../../artifacts/wan_aether_streaming_official_r1/extract_zoetrope_r2/submission_receipt.json)
- [上传目标保持相同 namespace](../../scripts/run_tstep_sync_loop.sh:927)
- [公共 protocol 默认 SEEDED 根](../../protocols/inscene_streaming_canonical_v2.json:26)
- [gap 工作列表的正式 namespace](../../features/canonical_gap_worklist.py:30)

## 其他模型“历史已齐”与当前找不到的差别

旧主线记录确实设计为“已验证的历史特征 + GAP 补提取”联合使用。历史 streaming owner 表引用 `FEAT_STREAMING`、`FEAT_STREAMING_BY_PREFIX`；target owner 表引用 `FEAT_TARGET` 加 GAP delta。只数 GAP 根里的 4,108/6,410 就宣布整个模型没提取齐，也是错误的。

但当前 AOSS `InsScene-15K/` 顶层未列出上述历史根，而且按每个模型/namespace 抽查的历史对象 HEAD 均为 404。历史表中一个 VideoMAE GAP 引用也为 404，而 V-JEPA2 GAP 和抽查的 target GAP 对象存在。因此不能直接拿旧表的 READY 状态当作当前 AOSS 事实。现在仅能证明被抽查旧路径失效，不能断言文件已删除、移走，或仍可从别处恢复。

`INVALID_ARCHIVE/` 本次只列出 `FEAT_STREAMING_CANONICAL_V2_SEEDLESS_V2_20260830/`，没有证明其他历史根移到了这里。

抽查证据见 [recorded_reference_current_checks.json](recorded_reference_current_checks.json)。这也是为什么本次不把历史“已提取”直接等同于“现在所有 probing 输入均可从 AOSS 读取”。

## 可复核附件与下一步

- [AOSS 根清单](roots.json)、[逐配置计数](inventory.tsv)、[结构化清单](inventory.json)
- [Wan/Aether 逐请求对账](wan_aether_reconciliation.json)、对应 `*_missing_remote.jsonl.gz`
- [manifest 与索引样本、Aether 动态上传复查](manifest_samples.json)
- [target 最新分片回执摘要](target_latest_receipts.json)

后续应优先恢复 Wan 的上传、等待 Aether streaming 上传收尾，再处理 Aether target 失败分片；不需要因目录计数不足就重跑已存在的 streaming 特征。此轮仅诊断和写入本地审计文件，没有修改提取/上传脚本、移动数据或提交新任务。
