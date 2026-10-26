# 目录指南

更新于 **2026-10-26**。活动工作围绕 SFT 基线及下一轮 Atsuko 实验；RL 尚未建立运行目录。

## 活动代码

| 路径 | 职责 |
| --- | --- |
| `mer_omni/peer_lora_sft.py` | 当前 Qwen LoRA 的上游训练实现，保留来源记录 |
| `mer_omni/sft/omni.py` | 公共模型加载、回答掩码及设备转换 |
| `mer_omni/sft/native_inputs.py` | 原生输入、固定目标计划、累积更新与状态恢复 |
| `mer_omni/sft/atsuko.py` | 可复用 Atsuko 输入、优化器、校准和 checkpoint 组件 |
| `mer_omni/native_atsuko*.py`、`layers/` | Atsuko 模型与频带实现 |
| `mer_omni/data.py`、`media.py`、`protocol.py`、`evaluation.py` | 共享数据、媒体、协议与指标 |
| `config/` | 当前 LoRA 训练／评测／选模配置，以及官方 face 配置，共四份 |
| `tools/` | 当前 LoRA 和 face 入口、权重下载及工作区检查，共十个文件 |
| `baselines/affectgpt_official_face/` | 固定版本的官方 face 源码和来源记录 |
| `tests/fixtures/` | 独立测试夹具；旧配置和旧 CLI 仅用于回归，不是运行入口 |

测试已脱离 A/B3 的运行目录。活动训练和评测代码不从历史归档导入任何模块。

## 数据和产物

| 路径 | 内容 |
| --- | --- |
| `data/manifests/sft/` | 固定目标分配、保守长度界和原生媒体元数据，原样从历史审计提取 |
| `data/manifests/benchmark/` | 九个官方测试集清单 |
| `models/` | 基座、提取模型和官方 face 权重，保留原位置 |
| `runs/peer_lora_epoch2_run02/` | 当前 LoRA 结果、三份 checkpoint、完整训练日志与原始源码快照 |
| `runs/affectgpt_official_face_screen20_run01/` | 官方 face 筛选与九集结果 |
| `.cache/sft/` | 当前训练和全量评测的 JPEG 帧缓存，已从运行目录分离 |
| `reports/peer_lora_epoch2_run02/` | 冻结评测计划、筛选计划和选择结果 |
| `reports/peer_lora_data_audit_20261025/` | 数据集核验记录 |
| `reports/sft_transition_20261026/` | 本次整理计划、移动记录、逐文件核验和测试日志 |
| `docs/results/` | 可发布的结果、来源及归档索引 |

`runs/peer_lora_epoch2_run02/checkpoint_receipts/`保留所有 checkpoint 的原始收据；不必保留 33 个活动权重目录来检查筛选记录。重新推理某个已归档候选前，需要按索引恢复其完整权重。

## 历史归档

`.local-archive/sft-transition-20261026/` 包含：

- `source_before/`：整理前完整代码、配置、测试和文档快照。
- `content/`：按原目录结构保存的历史运行、报告、日志、旧入口及校准产物。
- `previous/`：此前的 GitHub 清理归档。

移动采用同文件系统重命名，保留原文件内容。没有删除实验权重、预测、训练记录或审计证据。A/B3 权重原样保留在 `content/runs/`；历史配置与原始代码按索引恢复后才可复测，不能把旧 B3 校准直接套到新 Atsuko 实验。

来源记录见[上游代码来源](results/peer-source-provenance.json)、[数据来源](results/sft-data-provenance.json)、[当前布局迁移](results/sft-layout-migration-20261026.json)和[归档映射](results/archive-index-20261026.json)。历史收据不重写；活动源码使用实际哈希，并仅接受迁移记录明确列出的历史变化。
