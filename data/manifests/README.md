# 数据清单

[catalog.json](catalog.json) 是可公开的资源索引，记录当前本地清单的用途、样本数、字节数和 SHA-256。它不包含字幕、标签正文或媒体路径，训练程序不会把它当成 JSONL 输入。

## 本地文件

| 文件 | 用途 |
| --- | --- |
| `mercaptionplus.train.jsonl` | MER-Caption+ 的 31,327 条训练样本 |
| `mercaptionplus.train.receipt.json` | 训练清单和原始标注文件的哈希 |
| `benchmark/<数据集>.jsonl` | 单个测试集的官方样本、字幕、标签和媒体路径 |
| `benchmark/<数据集>.receipt.json` | 对应测试清单及来源文件的哈希 |
| `sft/mercaptionplus-target-plan.json` | 固定的 15,664 条解释／15,663 条标签任务分配 |
| `sft/mercaptionplus-length-bounds.json` | 原生输入的保守长度界，用于选取长输入预检样本 |
| `sft/native-media-metadata.json` | 原生输入所需的媒体长度等元数据 |

JSONL 公共字段为 `id`、`dataset`、`split`、`text`、`video`、`audio`；训练样本另含 `description`、`openset`，测试样本另含 `label`。字段含义、路径解析和完整性校验以 [mer_omni/data.py](../../mer_omni/data.py) 为准。

当前 LoRA 使用的 `runs/peer_lora_epoch2_run02/manifest.jsonl` 由训练源清单和固定任务分配生成，两轮重复每个 ID 的同一目标。恢复旧运行时应恢复原文件。对新的运行，`mer_omni.sft.native_inputs.plan_for` 可按给定种子和有序训练清单生成任务计划；原始样本顺序也是复现条件。

资源索引中的哈希对应已完成实验的原始文件；重新生成清单后，路径或序列化方式变化都会改变文件哈希。不要为通过校验而修改历史收据。
