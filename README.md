# Xiaoshaoxinxing

多模态情绪识别项目，研究路线为 **Qwen SFT → Atsuko＋SFT → RL**。当前处于 SFT 阶段，先建立稳定的 Qwen LoRA 基线，再检验 Atsuko 模块的增益，最后开展 RL。

更新于 **2026-10-26**。当前 Qwen LoRA 与官方 AffectGPT face 均已完成 MER-UniBench 九集全量评测。实验 A/B/B3 已退役为历史对照，整理前现存的源码、保留权重和原始结果保存在本地归档。

## 当前结果

单位为百分比。九项平均为四个情绪识别 HR、四个情感极性 weighted F1 和 OV-MERD+ 五情绪轮 Fₛ 的等权平均。

| 数据集 | Qwen LoRA SFT | AffectGPT face |
| --- | ---: | ---: |
| MER2023 | 80.43 | 79.23 |
| MER2024 | 79.15 | 79.04 |
| MELD | 58.44 | 55.94 |
| IEMOCAPFour | 63.97 | 57.62 |
| CMUMOSI | 83.94 | 78.64 |
| CMUMOSEI | 82.74 | 79.30 |
| SIMS | 84.79 | 86.14 |
| SIMSv2 | 84.97 | 85.51 |
| OVMERDPlus | 65.46 | 60.55 |
| **九项平均** | **75.99** | **73.55** |

两组均评测全部 **12,799** 条样本。LoRA 在 16 个收敛区间 checkpoint 中选中 **step34000**；face 在 7 个官方 checkpoint 中选中 **epoch50**。两者都使用同一批每集 20 条、共 180 条测试样本筛选，所以这不是独立验证集选模结果。

LoRA 使用贪心生成，face 保留官方采样生成。OV-MERD+ 全量 532 条含 99 条已知训练重叠；保留官方全量分数，并记录 LoRA 的重叠／非重叠子集结果。不能将上述分差归因于单一模块或训练超参数。

完整精度、checkpoint 排名及来源哈希见[当前结果索引](docs/results/experiments-20261026.json)。

## 当前 SFT 基线

- 基座：Qwen2.5-Omni-3B Thinker；训练集：MER-Caption+ 的 31,327 个唯一视频。
- 固定目标：15,664 条解释、15,663 条情绪标签；每条样本的目标在两轮中保持一致。
- 两轮、62,654 次更新，batch1、无梯度累积；attention q/k/v/o LoRA，r8、alpha16、dropout0.05。
- 学习率 1e-4，AdamW weight decay0.01，无 warmup／scheduler；两张 JPEG 帧、独立音频和字幕。
- 活动目录保留 step31327（第一轮结束）、step34000（本轮选中）、step62654（最终状态）。其他 30 个 checkpoint 已完整归档。

训练实现位于[peer_lora_sft.py](mer_omni/peer_lora_sft.py)，来源与固定上游版本见[代码来源记录](docs/results/peer-source-provenance.json)。`peer_*` 文件名对应这一轮已完成的配方复现。

## 后续 Atsuko 工作

[mer_omni/sft](mer_omni/sft) 包含从旧实验拆出的公共加载、原生输入、训练状态和 Atsuko 校准／优化器组件。Atsuko 模型实现继续保留在 `mer_omni/native_atsuko*.py`、`mer_omni/layers/`。

**尚未运行“当前两帧 LoRA 配方＋Atsuko”的新实验。**旧 B3 使用不同输入及训练配方，其结果只能作为历史参考。下一轮需要先确定两帧输入与 Atsuko 的接口，完成前缀、梯度和训练集校准检查，再登记新的配置和输出目录。RL 尚未启动。

## 检查当前工作区

```bash
# 只读核验三个保留 checkpoint、九集预测、缓存评分及官方 face 身份；不启动 GPU 推理
.venv/bin/python -B tools/check_sft_workspace.py

# CPU 回归测试
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  .venv/bin/python -B -m unittest discover -s tests -v
```

当前基线已经训练和评测完成。新实验使用新配置、审计及输出身份；不要把已有完成目录当成新任务的输出位置。数据、模型、缓存、运行产物和历史归档均不上传 GitHub。

## 导航

| 入口 | 内容 |
| --- | --- |
| [实验状态](docs/experiment-status.md) | SFT 主线、已完成结果和下一步 |
| [运行协议](docs/protocol.md) | 环境、配置、检查与运行边界 |
| [目录指南](docs/directory-guide.md) | 活动代码、数据产物、缓存与历史归档 |
| [历史 A 记录](docs/history/native-omni-experiment-a.md) | A 的退役说明与恢复位置 |
| [归档索引](docs/results/archive-index-20261026.json) | 旧、新路径映射及核验记录 |

本次整理完整保留实验文件；A、B3 最终权重、其他 LoRA checkpoint 和历史预测均可在本地归档找到。
