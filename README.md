# Xiaoshaoxinxing

多模态情绪识别研究代码：**Atsuko 频带分解与路由 + Qwen2.5-Omni-3B**，使用 MER-Caption+ 训练、MER-UniBench 评测。当前版本供代码审查，重点排查模型表现明显低于官方 AffectGPT baseline 的原因。

## 从哪里看

| 文件 | 审查内容 |
| --- | --- |
| [config/mer_omni_atsuko_padding.json](config/mer_omni_atsuko_padding.json) | 当前训练配置：分别定长补零、仅 CE、完整一轮 |
| [mer_omni/encoders/omni.py](mer_omni/encoders/omni.py) | 冻结的文本 embedding、音视频编码、视频空间平均、语言主干 LoRA |
| [mer_omni/model.py](mer_omni/model.py) | 序列补零、频带与模态路由、末位置读取、融合 token、答案监督与生成 |
| [mer_omni/layers/](mer_omni/layers/) | 频谱分解、router、Transformer 的实际实现 |
| [mer_omni/calibration.py](mer_omni/calibration.py) | 训练集等能量频带校准及指纹检查 |
| [mer_omni/data.py](mer_omni/data.py)、[mer_omni/media.py](mer_omni/media.py) | 数据清单、hybrid 任务采样、媒体预处理 |
| [mer_omni/training.py](mer_omni/training.py)、[mer_omni/objectives.py](mer_omni/objectives.py) | CE、梯度累积、尾批归一化、采样覆盖与断点恢复 |
| [mer_omni/evaluation.py](mer_omni/evaluation.py)、[mer_omni/protocol.py](mer_omni/protocol.py) | 标签抽取、指标、九集协议及重叠检查 |
| [mer_omni/reference/](mer_omni/reference/) | 上传的 Atsuko 原文件、AffectGPT 评分函数、情绪轮及来源记录 |
| [tests/](tests/) | 合成输入、微型 Omni、原版核心数值对照等离线测试 |

计算路径：原始音视频与字幕 → 冻结 Omni 特征 → 各模态补零 → 卷积投影 → 三频带分解与路由 → Transformer 末位置 → 模态路由 → 一个融合 token → Thinker + LoRA → 自然语言回答。

## 当前实验

- 训练数据保持原发布形态，共 **31,327** 条，随机不重复遍历一轮；microbatch=1、梯度累积=3，共 **10,443** 个优化器步，最后一步只有 1 条。
- 文本/视频/音频分别补到 **96 / 50 / 704** 个位置；卷积后长度为 **92 / 46 / 700**。训练超长报错，推理可显式选择右截断；不做词级时间对齐。
- 频带在完整训练集上校准，边界分别为 `[29,61]`、`[3,8]`、`[32,252]`。
- `mcm.loss_mode=ce_only`，实际 `loss=nll`；仅监督回答、`###` 和 EOS。六条辅助语言分支不执行，专用参数冻结。保留辅助模块构造及旧模式代码，以维持初始化随机数顺序、校准与检查点兼容；旧配置仅作为 [测试夹具](tests/fixtures/legacy_full_loss.json) 保留。
- 描述/标签 hybrid 任务各以 50% 概率采样。warmup=5,000 步，warmup_lr=1e-6，lr=min_lr=1e-5。

2026-10-13 完成的固定最终检查点结果如下；没有按测试分数选检查点。

| 模型 | MER2023 Hit Rate | MER2024 Hit Rate | MOSEI weighted F1 / Accuracy |
| --- | ---: | ---: | ---: |
| 当前模型，一轮，仅 CE | 42.96% | 43.03% | 52.74% / 56.69% |
| 官方 AffectGPT frame 发布权重 | 79.15% | 75.06% | 未评测 |

当前模型三集共 6,239 条完成生成和抽取，MOSEI 在 3,634 条非中性真值样本上计分。上述评分已与 AffectGPT 原函数核对，差异为 0。baseline 为 frame 版本，与论文 face 版本区分，也不是等主干、等训练预算消融。

## 希望重点排查

1. **信息压缩与读取**：各模态仅取 Transformer 末位置，再压缩为一个融合 token，能否保留足够情绪信息？旧插值实验的音频探针在末位置读取时明显弱于时间均值；这只是旧实验的候选解释，尚未证明对当前模型适用。
2. **补零后的序列处理**：补零参与频谱分解，前端 Transformer 没有另外传入有效长度 mask。请结合原版实现审查这一适配是否合理，尤其是最多 4 个视频时间位置补到 50 的情况。
3. **输入依赖**：旧实验曾出现正常、打乱、置零输入的回答或分数接近。当前完整一轮结果有所提升，但仅凭测试指标不能证明该问题已经解决。
4. **语言接口与优化**：文本特征是冻结 token embedding；原始字幕不直接进入生成提示。请检查融合 token 的尺度、提示形式、LoRA 梯度，以及长描述在按答案 token 归一化时对标签任务权重的影响。

原版数值对照验证的是相同张量、权重和边界下的核心融合计算；Omni 编码、补零长度、生成式预测头和 CE 属于本项目适配。Transformer/router 等依赖未获上传原文件独立认证，不能声称整个模型等价于原作者完整实验。详细约定见 [docs/protocol.md](docs/protocol.md)。

## 检查与运行

支持 Linux / Python 3.10，使用 PyTorch 2.4.0+cu121、Transformers 4.52.4，依赖见 [requirements/train.txt](requirements/train.txt)。系统需安装 `ffmpeg` / `ffprobe`。标签抽取使用 [独立环境](requirements/evaluation.txt)，其 Transformers 版本与训练环境不同。

```bash
git clone https://github.com/Xiaoshaoxinxing/Xiaoshaoxinxing.git
cd Xiaoshaoxinxing
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --extra-index-url https://download.pytorch.org/whl/cu121 -r requirements/train.txt

# 在已安装依赖的环境中运行，不需要真实数据或 3B 权重
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  python -m unittest discover -s tests -v

# 统一入口默认使用当前补零、仅 CE 配置
python tools/mer_omni.py --help
python tools/mer_omni.py preflight
```

完整运行需自行准备配置中的本地模型、原始数据、清单与校准结果；GitHub 版本不包含这些资源。数据目录约定和准备、训练、评测命令见 [运行协议](docs/protocol.md)。本机已完成本轮训练，`train.sh` 会拒绝覆盖已有运行目录。

配置里的资源路径默认相对本仓库根目录解析，也支持自行设置绝对路径和 `~`；不需要原服务器目录。新生成的清单和标注收据保存相对路径，可随数据目录一同搬迁。`train.sh` 优先使用本仓库 `.venv/bin/python`，没有时使用 PATH 中的 Python，也可通过 `MER_OMNI_PYTHON=/path/to/python bash train.sh` 显式指定环境。旧服务器生成的绝对路径清单、评测缓存不会自动迁移，边界见[路径与迁移](docs/protocol.md#路径与迁移)。

数据、权重、检查点、日志、缓存及清理前归档均由 `.gitignore` 排除。主目录只保留审查与运行相关源码、配置、测试、依赖及说明；本地历史文件在 `.local-archive/github-cleanup-20261013/`，不应加入版本控制。
