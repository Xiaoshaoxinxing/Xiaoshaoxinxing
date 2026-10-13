# 当前模型与运行协议

本文件说明当前补零、仅 CE 版本。训练配置为 `config/mer_omni_atsuko_padding.json`；评测配置为 `config/mer_omni_evaluation.json`，其中生成与标签抽取设置取自最终 step 10443 的正式评测记录。

## 模型边界

Omni-3B 的视觉编码器、音频编码器、文本 embedding 和原始语言权重冻结；语言主干使用 LoRA（r=16、alpha=32、dropout=0.05）。字幕经过 token embedding，音频保留原生时序，视频仅在每个时间位置内平均空间 patch。

三模态独立右补零至 96/50/704，不插值、不做词级对齐。kernel=5 的卷积投影到 256 维后，序列长度为 92/46/700。谱基和三频带重建沿用 Atsuko，等能量边界在全部 MER-Caption+ 训练输入上统计并冻结；模型不从测试批次重新估计边界。

频带路由后，三模态分别经过 4 层、8 头 Transformer，读取末时间位置。`router_input=pre_attn` 时，模态路由决策使用注意力前序列均值，融合对象仍是注意力后的末位置。融合向量经映射成为一个 Thinker 输入 token；提示中不另外拼接原始字幕或未路由的音视频特征。

训练只执行主语言路径 CE；提示与 padding 不受监督，目标包含回答、`###`、EOS。描述与开放标签 hybrid 采样概率各 50%。CE 模式冻结并跳过六条辅助分支和蒸馏映射，保留它们的构造以保持初始化顺序。`tests/fixtures/legacy_full_loss.json` 仅供旧模式及原版数值回归测试，不能当作当前训练设置。

上传的 Atsuko 四个原文件及哈希位于 `mer_omni/reference/atsuko_original/`。实际运行使用 `mer_omni/layers/`，不导入旧 Atsuko 项目。原版对照不覆盖作者未提供的数据预处理流程；Omni 特征、具体补零长度及生成式任务均是适配选择。AffectGPT 评分参考函数与情绪轮保留在 `mer_omni/reference/`，附带来源与原许可证。

## 本地资源

GitHub 仅提供代码。按配置准备下列目录，或在自定义配置中改路径：

```text
models/Qwen2.5-Omni-3B/             # 包含配置、tokenizer 和完整权重
models/Qwen2.5-7B-Instruct/         # 仅标签抽取阶段需要
data/shared/mer2025-dataset/       # 原 MER-Caption+ 发布文件
data/shared/<benchmark-directory>/ # 原 MER-UniBench 发布目录
data/manifests/                    # 本机生成的清单和标注收据
data/calibration/                  # 全训练集校准结果
```

Omni 固定版本为配置内的 `revision`。加载时只读本地权重，不自动下载；架构检查不等同于版本哈希核验。`preflight` 的 `model_weights_verified=false` 也不表示执行过完整性检查。

原始训练文件包括 `track2_train_mercaptionplus.csv`、`track3_train_mercaptionplus.csv`、`subtitle_chieng.csv` 及配套音视频，具体路径由 `mer_omni/data.py` 构造。本机 `data/shared` 是指向 `data/datasets` 的链接，训练数据和评测模型还复用 AffectGPT 目录；这些链接不随代码上传。

```bash
# 仅在本地尚无清单时执行，已有文件拒绝覆盖
python tools/mer_omni.py prepare --dataset train
python tools/mer_omni.py prepare --dataset MER2023
python tools/mer_omni.py prepare --dataset MER2024
python tools/mer_omni.py prepare --dataset CMUMOSEI
python tools/mer_omni.py preflight

# 在新环境中、资源完整且尚未校准/训练时执行
python tools/mer_omni.py calibrate --device cuda:0
python tools/mer_omni.py train --device cuda:0 --output runs/new_run
```

完整一轮覆盖 31,327 条且各一次，microbatch=1、累积=3，共 10,443 步。最后一步按实际样本和答案 token 数归一化，不跨轮补样本，超长文本报错。每步记录样本 ID；完成时保存覆盖收据。模型检查点包括适配器、频带、优化器和随机状态，加载仍需要相同的基础 Omni 权重。

恢复训练使用相同配置加 `--resume <checkpoint>`；`--max-steps N` 指绝对停止步数。迁移服务器后，旧检查点和清单可能仍绑定原绝对路径与清单哈希，不能只复制 `.pt` 文件就恢复。当前训练器为单设备、`num_workers=0`，不支持 DDP。

## 评测

| 标识 | 原始样本数 | 主指标 |
| --- | ---: | --- |
| MER2023 | 411 | 五情绪轮 level1 平均 Hit Rate |
| MER2024 | 1,169 | 同上 |
| IEMOCAPFour | 1,241 | 同上 |
| MELD | 2,610 | 同上 |
| CMUMOSI | 686 | 去除真值 0 后二元 weighted F1 |
| CMUMOSEI | 4,659 | 同上 |
| SIMS | 457 | 同上 |
| SIMSv2 | 1,034 | 同上 |
| OVMERDPlus | 532 | 五情绪轮 level1 平均集合 F1 |

目录、标注和字幕列遵循 `mer_omni/protocol.py`：MER2024 使用中文，OV-MERD+ 使用 `sentence`，其余使用英文字幕。保留原发布划分；OV-MERD+ 与当前训练集有 99 条已核实重叠，配置显式列出允许的 ID，其他重叠报错。相关结果不能表述为训练与测试完全无交集。

评测先生成描述，再用独立 Qwen2.5-7B-Instruct 抽取情绪词/极性，最后应用原评分规则。最终检查点不通过测试分数选择。当前只完成 README 中列出的集合，不能报告九集均分。

```bash
python tools/mer_omni.py --config config/mer_omni_evaluation.json infer \
  --dataset MER2023 --checkpoint runs/mer_omni_padding_epoch01/checkpoint_step_0010443.pt \
  --sequence-overflow truncate_right --output runs/eval/MER2023

# 在 requirements/evaluation.txt 对应的独立 Python 环境执行
python tools/mer_omni.py --config config/mer_omni_evaluation.json extract \
  --dataset MER2023 --predictions runs/eval/MER2023/predictions.jsonl \
  --gpu-memory-utilization 0.50 --max-model-len 4096 --max-num-seqs 8 \
  --output runs/eval/MER2023/extraction.json

python tools/mer_omni.py --config config/mer_omni_evaluation.json score \
  --dataset MER2023 --predictions runs/eval/MER2023/predictions.jsonl \
  --extraction runs/eval/MER2023/extraction.json --output runs/eval/MER2023/score.json
```

`truncate_right` 只在推理时覆盖超长策略，保留原生特征前缀，长度和训练校准不变；逐样本记录受影响模态，退出后恢复模型设置。MER2023 有 1 条、MOSEI 有 29 条超长；MER2024 无超长输入。训练默认仍为 `error`。

正式生成使用 seed=42、temperature=1、top_p=0.9、top_k=20、max_new_tokens=1200、max_length=2000、EOS `[151645,151643]`，并识别 `###`。抽取使用 vLLM 0.6.2 / Transformers 4.49.0、engine seed=0、sampling seed=None、temperature=0.7、top_p=0.8、repetition_penalty=1.05、max_tokens=512、batch=8。需要本机单进程 NCCL rendezvous 时可加 `--single-process-store`。

情感指标只过滤真值 0；预测 neutral/无效值转为 0，再按 `>0` 二分。OV 集合先对样本求 precision/recall 均值，再取调和平均，最后平均五个 wheel。缺失、重复预测和过期抽取缓存报错；九集聚合要求同一检查点与评测设置。

## 测试范围

`python -m unittest discover -s tests -v` 使用合成媒体、微型真实 Transformers Omni 和上传的原版快照，覆盖 CE 掩码/梯度、辅助路径跳过、尾批归一化、完整采样、断点恢复、补零/截断、频带校准及评分规则。不需要实际 3B 权重或数据集。

若服务器保留旧 Atsuko 目录，测试还比较未上传依赖的函数体；其他机器只对随库提供的参考文件做验证。通过这些测试不能证明模型识别能力或整个实验与原作者完全等价。
