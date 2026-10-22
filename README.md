# Xiaoshaoxinxing

多模态情绪识别研究代码：**Atsuko 频带分解与路由 + Qwen2.5-Omni-3B**，使用 MER-Caption+ 训练、MER-UniBench 评测。

更新至 **2026-10-22**：已完成序列插值、优化日程、16帧采样、3轮预算及LoRA容量对照。本项目这组AV-token实验中，16帧、rank16、3epoch的MER2023 Hit Rate为 **64.56%**；相同设置1epoch为 **64.11%**，mscore更高，且已完成MER2024。LoRA64虽降低训练loss，MER2023降至 **60.68%**。

## 结果与配置

| 实验 | 训练配置 | MER2023 Hit Rate / mscore | MER2024 Hit Rate / mscore |
| --- | --- | ---: | ---: |
| 原独立AV token，8帧 | [ablation](config/mer_omni_av_tokens_ablation.json) | 61.54% / 44.13% | 未完成 |
| omni_resample | [resample](config/mer_omni_av_tokens_resample.json) | 61.48% / 43.51% | 未评测 |
| 分组学习率、短warmup、梯度裁剪 | [lrschedule](config/mer_omni_av_tokens_lrschedule.json) | 62.55% / 44.08% | 未评测 |
| **16帧，rank16，1epoch** | [frames16](config/mer_omni_av_tokens_frames16.json) | **64.11% / 45.52%** | **61.11% / 44.59%** |
| **16帧，rank16，3epoch** | [frames16_epoch3](config/mer_omni_av_tokens_frames16_epoch3.json) | **64.56% / 43.92%** | 未评测 |
| 16帧，rank64，1epoch | [frames16_lora64](config/mer_omni_av_tokens_frames16_lora64.json) | 60.68% / 42.41% | 未评测 |

各配置有对应评测配置，原ablation对应 `mer_omni_av_tokens_evaluation.json`，其余为同名 `_evaluation.json`。每个实验从预训练基座重新初始化，同一实验中断才从其自身检查点恢复。评测使用预定最终step：1epoch为10443，3epoch为31329；没有按测试分数挑中间检查点。

这些是单seed、单次生成/抽取结果，不能据此证明稳定收益或唯一退化原因。MER2023为411条，MER2024为1,169条；每个情绪轮有自己的适用样本数。Hit Rate/mscore不等同于accuracy/F1，尚无当前模型的九集均分。

更早的单融合token、字幕直达、原生均值和四段音频实验见[实验记录](docs/experiment-status.md)及[10月17日结果快照](docs/results/experiments-20261017.json)。官方AffectGPT frame发布权重MER2023/2024为79.15%/75.06%，其主干、编码器与训练预算不同，不是等条件结构消融。

## 当前保留设置

两组保留方案都使用 **zero_pad**（文本/视频/音频96/50/704）、最多16帧、LoRA r16/alpha32、完整字幕及前端末位独立音视频token、仅主路径CE。不是原生均值旁路，也不是resample模式。

- seed42，batch_size1，gradient_accumulation3；每轮31,327条各一次，尾步只处理剩余1条，不跨轮补样本。
- warmup500，LoRA lr=min_lr=2e-5、warmup_lr=1e-6，前端基础lr=1e-4；所有组共享归一化日程因子，warmup之后恒定。
- weight_decay0.05，grad_clip0.6，hybrid描述/标签目标，max_length1024，num_workers0。
- 1epoch和3epoch共用frames16训练集校准，频带边界text/video/audio为 `[29,61]/[1,3]/[32,252]`。改变帧数或LoRA的omni设置需对应的新校准；仅改训练预算不需要。

本地保留两组最终检查点；其余实验权重已清理，配置、日志及评分证据保留。**本仓库不分发模型权重、原始数据、训练日志或逐样本预测**；公开摘要与哈希见[10月22日结果快照](docs/results/experiments-20261022.json)。

## 检查与运行

Linux / Python3.10，系统安装ffmpeg/ffprobe；训练与标签抽取使用独立依赖环境。

```bash
git clone https://github.com/Xiaoshaoxinxing/Xiaoshaoxinxing.git
cd Xiaoshaoxinxing
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --extra-index-url https://download.pytorch.org/whl/cu121 -r requirements/train.txt
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  python -m unittest discover -s tests -v
python tools/mer_omni.py --help
python tools/mer_omni.py --config config/mer_omni_av_tokens_frames16.json preflight
```

此次发布副本通过98项离线测试，并完成8项优化配方CPU验证（含分组恢复和旧配置兼容）。离线测试使用合成输入与微型Omni组件，不需要实际3B权重。真实训练/评测需自行准备配置指定的本地模型、数据、清单及校准。资源路径相对仓库根目录解析；详见[运行协议](docs/protocol.md)。`train.sh`和统一CLI的默认配置仍是早期padding基线；运行当前方案必须显式传 `--config`，并使用新的输出目录。

## 文件入口

| 文件 | 内容 |
| --- | --- |
| [docs/experiment-status.md](docs/experiment-status.md) | 实验结果、训练恢复、检查点指纹与保留状态 |
| [docs/protocol.md](docs/protocol.md) | 资源、校准、训练、恢复、三阶段评测与迁移 |
| [问题记录](AffectGPT对照差距与待排查问题.md) | 证据支持的结论、尚未确定的解释 |
| [mer_omni/model.py](mer_omni/model.py) | 前端、提示、独立音视频入口和生成 |
| [mer_omni/training.py](mer_omni/training.py) | 分组优化、日程、梯度裁剪与恢复 |
| [tools/audit_native_sequence_lengths.py](tools/audit_native_sequence_lengths.py) | 全量训练清单的文本、音频估计和视频长度审计 |
| [tools/profile_router_weights.py](tools/profile_router_weights.py) | 前256条训练样本的只读模态/频带路由诊断 |
| [mer_omni/reference](mer_omni/reference/) | Atsuko原文件、AffectGPT评分函数与来源许可证 |

`tools/run_*_ablation.py`为历史服务器调度记录，依赖未发布的本机reports，不作为新克隆的统一入口。复现使用 `tools/mer_omni.py`，不要直接重启依赖旧收据的脚本。
