# 数据目录

保存 MER-Caption+ 训练数据、MER-UniBench 九个测试集及其本地清单。仓库发布本说明和[清单资源索引](manifests/catalog.json)；音视频、人脸数组、完整字幕／标签清单和下载包保存在本地。

## 本地布局

```text
data/
├── datasets/                     # MER-Caption+ 与九个测试集的原始发布目录
│   ├── mer2025-dataset/           # MER-Caption+ 所在发布包
│   └── *-process/                # MER-UniBench 各数据集
├── shared/                       # 官方 AffectGPT 使用的同名数据目录或软链接
├── manifests/
│   ├── catalog.json              # 已发布的资源索引，不是训练输入
│   ├── mercaptionplus.train.jsonl
│   ├── mercaptionplus.train.receipt.json
│   ├── benchmark/                # 每集 JSONL 和对应 receipt.json
│   └── sft/                      # 固定任务分配、长度界和媒体元数据
└── downloads/                    # 本地下载缓存
```

## 准备数据

1. MER-Caption+ 来源为 [MERChallenge/MER2025](https://huggingface.co/datasets/MERChallenge/MER2025)。本项目使用其 31,327 个唯一训练视频，依据 `track2_train_mercaptionplus.csv`、`track3_train_mercaptionplus.csv` 和 `subtitle_chieng.csv` 对齐标签、解释和字幕；媒体放在对应的 `video/` 与 `audio/` 下。
2. MER-UniBench 数据来源和发布目录见仓库内固定版本的[官方说明](../baselines/affectgpt_official_face/upstream/README.md#-mer-unibench)。保留各数据集的文件命名和测试划分；九集名称、目录和样本数见[资源索引](manifests/catalog.json)。
3. Qwen 使用完整视频、独立音频和字幕。官方 AffectGPT face 还需要发布包中的 `openface_face/` 人脸数组；`data/shared/` 的同名数据目录应与 Qwen 清单指向同一批样本。
4. 清单生成逻辑在 [mer_omni/data.py](../mer_omni/data.py)：`build_train_rows`、`build_benchmark_rows` 和 `save_manifest`。保存清单时应同时保存来源文件的收据，并检查媒体文件存在、ID 唯一及样本覆盖完整。详细清单约定见[manifests/README.md](manifests/README.md)。

现有运行绑定原始清单、任务分配及来源哈希。恢复历史运行需要同时恢复这些本地文件；在新机器重新生成路径不同的清单时，应登记新的运行身份，不能覆盖已完成实验的记录。

MER-UniBench 全量共 12,799 条；OV-MERD+ 的 532 条中有 99 条与本轮训练集重叠，沿用项目已记录的全量与子集报告规则。
