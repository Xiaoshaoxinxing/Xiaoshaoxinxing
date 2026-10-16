# Xiaoshaoxinxing

多模态情绪识别研究代码：**Atsuko 频带分解与路由 + Qwen2.5-Omni-3B**，使用 MER-Caption+ 训练、MER-UniBench 评测。

更新至 **2026-10-16**：当前版本在融合 token 外保留原始字幕，并增加独立音频、视频 token。完整一轮训练和 MER2023 评测已完成，Hit Rate 为 **61.54%**；MER2024 暂缓。字幕对照为 61.01%，新增音视频入口提高 0.52 个百分点，尚不能视为经过多随机种子验证的稳定提升。

最新结果、检查点指纹和评测边界见 [实验进展](docs/experiment-status.md)；待排查原因见 [问题记录](AffectGPT对照差距与待排查问题.md)。

## 从哪里看

| 文件 | 审查内容 |
| --- | --- |
| [config/mer_omni_av_tokens_ablation.json](config/mer_omni_av_tokens_ablation.json) | 当前训练配置：字幕、融合 token、独立音频与视频 token |
| [config/mer_omni_av_tokens_evaluation.json](config/mer_omni_av_tokens_evaluation.json) | 当前正式评测的生成与标签抽取配置 |
| [config/mer_omni_subtitle_ablation.json](config/mer_omni_subtitle_ablation.json) | 仅增加字幕直达的对照 |
| [config/mer_omni_atsuko_padding.json](config/mer_omni_atsuko_padding.json) | 最初的单融合 token 基线 |
| [mer_omni/encoders/omni.py](mer_omni/encoders/omni.py) | 冻结的文本 embedding、音视频编码、视频空间平均、语言主干 LoRA |
| [mer_omni/model.py](mer_omni/model.py) | 序列补零、频带与模态路由、末位置读取、融合 token、答案监督与生成 |
| [mer_omni/layers/](mer_omni/layers/) | 频谱分解、router、Transformer 的实际实现 |
| [mer_omni/calibration.py](mer_omni/calibration.py) | 训练集等能量频带校准及指纹检查 |
| [mer_omni/data.py](mer_omni/data.py)、[mer_omni/media.py](mer_omni/media.py) | 数据清单、hybrid 任务采样、媒体预处理 |
| [mer_omni/training.py](mer_omni/training.py)、[mer_omni/objectives.py](mer_omni/objectives.py) | CE、梯度累积、尾批归一化、采样覆盖与断点恢复 |
| [mer_omni/evaluation.py](mer_omni/evaluation.py)、[mer_omni/protocol.py](mer_omni/protocol.py) | 标签抽取、指标、九集协议及重叠检查 |
| [mer_omni/reference/](mer_omni/reference/) | 上传的 Atsuko 原文件、AffectGPT 评分函数、情绪轮及来源记录 |
| [tests/](tests/) | 合成输入、微型 Omni、原版核心数值对照等离线测试 |

共同前端：原始音视频与字幕 → 冻结 Omni 特征 → 各模态补零 → 卷积投影 → 三频带分解与路由 → Transformer 末位置 → 模态路由 → 一个融合 token。

当前 Thinker + LoRA 输入包含融合 token、独立音频 token、独立视频 token 和原始字幕。独立音视频 token 来自同一前端的单模态末位置表示，经各自的 `Linear(256, 2048)` 投影；它们绕过最后的模态加权，**仍经过补零、频谱和注意力处理**。

## 当前实验

- 训练数据保持原发布形态，共 **31,327** 条，随机不重复遍历一轮；microbatch=1、梯度累积=3，共 **10,443** 个优化器步，最后一步只有 1 条。
- 文本/视频/音频分别补到 **96 / 50 / 704** 个位置；卷积后长度为 **92 / 46 / 700**。训练超长报错，推理可显式选择右截断；不做词级时间对齐。
- 频带在完整训练集上校准，边界分别为 `[29,61]`、`[3,8]`、`[32,252]`。
- `mcm.loss_mode=ce_only`，实际 `loss=nll`；仅监督回答、`###` 和 EOS。六条辅助语言分支不执行，专用参数冻结。保留辅助模块构造及旧模式代码，以维持初始化随机数顺序、校准与检查点兼容；旧配置仅作为 [测试夹具](tests/fixtures/legacy_full_loss.json) 保留。
- 描述/标签 hybrid 任务各以 50% 概率采样。warmup=5,000 步，warmup_lr=1e-6，lr=min_lr=1e-5。

三个实验分别从相同 seed=42 的预训练底座及对应初始化开始，均完整训练一轮；字幕和音视频实验未从旧实验的已训练检查点继续微调。使用固定最终 step 10443 评测，没有按测试分数选检查点。

| 模型 | MER2023 Hit Rate | MER2024 Hit Rate | MOSEI weighted F1 / Accuracy |
| --- | ---: | ---: | ---: |
| 原单融合 token 基线 | 42.96% | 43.03% | 52.74% / 56.69% |
| 融合 token + 原始字幕 | 61.01% | 56.62% | 未评测 |
| 当前：融合 token + 原始字幕 + 独立音视频 token | **61.54%** | 暂缓，未完成 | 未评测 |
| 官方 AffectGPT frame 发布权重 | 79.15% | 75.06% | 未评测 |

原单融合 token 基线三集共 6,239 条完成生成和抽取，MOSEI 在 3,634 条非中性真值样本上计分；其评分已与 AffectGPT 原函数核对，差异为 0。字幕实验完成 MER2023/2024；当前独立音视频实验只完成 MER2023 的 411 条。MER2024 在生成 213/1,169 条后按用户要求停止，没有正式分数。九集均分尚不可报告。

官方 baseline 为 frame 版本，与论文 face 版本区分；它与本项目不是等主干、等训练预算消融。机器可读结果及来源文件哈希见 [结果快照](docs/results/experiments-20261016.json)。

## 希望重点排查

1. **信息压缩与读取**：独立音视频入口仍使用前端末位置向量，尚未验证原生音视频直接汇聚是否更合适。MER2023 的 0.52 个百分点提升不能排除前端瓶颈。
2. **补零后的序列处理**：补零参与频谱分解，前端 Transformer 没有另外传入有效长度 mask。请结合原版实现审查这一适配是否合理，尤其是最多 4 个视频时间位置补到 50 的情况。
3. **输入依赖**：10 月 15 日对原单融合 token 基线的诊断发现字幕影响较弱，音视频确实影响输出。字幕直达重训后 MER2023/2024 提高 18.05/13.58 个百分点；旧权重的诊断数值不能直接当成当前权重的实测结果。
4. **语言接口与优化**：前端文字分支仍是冻结 token embedding；当前同时保留完整字幕直达。长描述在按答案 token 归一化时占据较大损失系数，任务权重和长 warmup 的影响仍待独立验证。

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

# 统一入口默认仍为原单融合 token 基线；复现当前版本必须显式选择配置
python tools/mer_omni.py --help
python tools/mer_omni.py --config config/mer_omni_av_tokens_ablation.json preflight
```

完整运行需自行准备配置中的本地模型、原始数据、清单与校准结果；GitHub 版本不包含这些资源。数据目录约定和当前版本训练、仅 MER2023 评测命令见 [运行协议](docs/protocol.md)。`train.sh` 保留原基线配置，并拒绝覆盖已有运行目录。

配置里的资源路径默认相对本仓库根目录解析，也支持自行设置绝对路径和 `~`；不需要原服务器目录。新生成的清单和标注收据保存相对路径，可随数据目录一同搬迁。`train.sh` 优先使用本仓库 `.venv/bin/python`，没有时使用 PATH 中的 Python，也可通过 `MER_OMNI_PYTHON=/path/to/python bash train.sh` 显式指定环境。旧服务器生成的绝对路径清单、评测缓存不会自动迁移，边界见[路径与迁移](docs/protocol.md#路径与迁移)。

数据、权重、检查点、原始日志、缓存及本地归档由 `.gitignore` 排除。公开版本保留模型代码、三组配置、测试、依赖、问题记录和不含原始媒体/逐条预测的结果摘要。

`tools/run_av_tokens_ablation.py` 是本次服务器实验的调度记录，依赖未公开的 `reports/` 预注册收据和验证脚本，默认计划还包含 MER2024；不作为克隆后的运行入口。请使用运行协议中的 `tools/mer_omni.py` 分阶段命令，按需选择评测集。
