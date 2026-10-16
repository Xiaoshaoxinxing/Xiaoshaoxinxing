# 当前模型与运行协议

本文件说明截至 2026-10-16 的补零、仅 CE 版本。当前训练配置为 `config/mer_omni_av_tokens_ablation.json`；评测配置为 `config/mer_omni_av_tokens_evaluation.json`。当前模型使用原始字幕、融合 token 及独立音视频 token。

原基线使用 `mer_omni_atsuko_padding.json` / `mer_omni_evaluation.json`；字幕对照使用 `mer_omni_subtitle_ablation.json` / `mer_omni_subtitle_evaluation.json`。统一 CLI 和 `train.sh` 的默认配置仍为原基线，当前版本必须显式传入 `--config`。三个版本的检查点和提示词设置不可混用。

## 模型边界

Omni-3B 的视觉编码器、音频编码器、文本 embedding 和原始语言权重冻结；语言主干使用 LoRA（r=16、alpha=32、dropout=0.05）。字幕经过 token embedding，音频保留原生时序，视频仅在每个时间位置内平均空间 patch。

三模态独立右补零至 96/50/704，不插值、不做词级对齐。kernel=5 的卷积投影到 256 维后，序列长度为 92/46/700。谱基和三频带重建沿用 Atsuko，等能量边界在全部 MER-Caption+ 训练输入上统计并冻结；模型不从测试批次重新估计边界。

频带路由后，三模态分别经过 4 层、8 头 Transformer，读取末时间位置。`router_input=pre_attn` 时，模态路由决策使用注意力前序列均值，融合对象仍是注意力后的末位置。融合向量经映射成为一个 Thinker 输入 token。

`prompt.include_subtitle=true` 时，原始字幕另行进入 Thinker 提示词，前端文字分支保持。当前还启用 `prompt.include_av_tokens=true`：从 `frontend.modalities` 取音频索引 2 和视频索引 1，分别经可训练 `Linear(256, 2048)` 得到一个独立 token。它们仍经过完整频谱前端，仅绕过最后的模态加权。原基线不启用这两个开关。

新增投影在原有模块之后初始化，并保存/恢复 PyTorch RNG，使共有参数初始值与原有实验一致。完整训练时各实验从底座和 seed=42 初始化开始。推理会核验检查点的字幕和音视频入口设置，拒绝跨版本混用。

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

原始训练文件包括 `track2_train_mercaptionplus.csv`、`track3_train_mercaptionplus.csv`、`subtitle_chieng.csv` 及配套音视频，具体路径由 `mer_omni/data.py` 构造。`data/shared` 可以直接是本地数据目录，也可以是你自行建立的符号链接。代码不要求原服务器上的 AffectGPT 目录或链接。

```bash
# 仅在本地尚无清单时执行，已有文件拒绝覆盖
python tools/mer_omni.py prepare --dataset train
python tools/mer_omni.py prepare --dataset MER2023
python tools/mer_omni.py --config config/mer_omni_av_tokens_ablation.json preflight

# 在新环境中、资源完整且尚未校准/训练时执行
python tools/mer_omni.py --config config/mer_omni_av_tokens_ablation.json calibrate --device cuda:0
python tools/mer_omni.py --config config/mer_omni_av_tokens_ablation.json train \
  --device cuda:0 --output runs/new_av_run --log-every 10 --checkpoint-every 1000
```

完整一轮覆盖 31,327 条且各一次，microbatch=1、累积=3，共 10,443 步。最后一步按实际样本和答案 token 数归一化，不跨轮补样本，超长文本报错。每步记录样本 ID；完成时保存覆盖收据。模型检查点包括适配器、频带、优化器和随机状态，加载仍需要相同的基础 Omni 权重。

本次三个实验的谱前端设置及共有投影初始化一致，因此复用原训练集校准；新环境需先校准。更改序列长度、卷积或前端配置后不能直接复用原校准结果。周期检查点在独立文件中保存，不中断训练循环。

恢复训练使用相同实验设置加 `--resume <checkpoint>`；`--max-steps N` 指绝对停止步数。恢复允许数据根目录、训练/评测清单位置、校准文件位置和 evaluator 本地位置变化；模型设置、训练设置、随机状态和训练清单哈希仍严格核对。当前训练器为单设备、`num_workers=0`，不支持 DDP。

## 路径与迁移

- 配置中的 `data_root`、`train_manifest`、`benchmark_manifest_dir`、`calibration_file`、`omni.source`、`evaluator.model` 使用仓库根目录作为相对路径基准；配置资源路径支持 `~`。运行时转换成当前机器的绝对路径是正常操作，不代表依赖原服务器。
- `--config`、`--output`、`--resume` 等显式命令行路径按当前工作目录解释；默认配置由脚本自身位置定位。长度审计工具的 `--manifest`、`--tokenizer`、`--output` 按仓库根目录解析；错误统计工具可用 `--manifest-dir` 指定清单目录。
- `prepare` 新生成的音视频路径及标注收据来源路径相对清单所在目录保存。整体搬迁时保持数据与清单的相对布局即可，清单字节及哈希不变；原始标注内容变化仍会报错。数据集通过符号链接访问时，保存配置中的链接路径；新机器上需要自行建立有效链接。
- 推理使用当前配置指定的训练清单，而不是盲目打开检查点里记录的旧路径，并验证该清单与检查点绑定的 SHA256 相同。已在合成输入上验证搬目录后继续训练与未中断训练一致。
- 迁移时应保持 `omni.source` 的相对配置值不变，将同一固定版本权重放到新仓库的对应位置。它仍参与架构比较和校准指纹；仅修改它的配置字符串也可能触发校准/检查点一致性检查，不会静默接受不同模型。
- 旧格式绝对路径清单和收据继续兼容，但不会被自动改写。旧清单若需要重建，原始字节哈希会改变，不能直接冒充旧检查点的训练清单；需要单独处理旧资源迁移。标签抽取缓存仍严格比较 evaluator 设置，移动其模型位置后应重新抽取。此类旧产物未包含在 GitHub 仓库中。
- `train.sh` 可使用仓库内 `.venv`、PATH 中的 Python 或显式 `MER_OMNI_PYTHON`，例如激活 Conda 后执行 `MER_OMNI_PYTHON=python bash train.sh`。脚本从其他目录启动时也会先切换到自身所在的仓库目录。

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
python tools/mer_omni.py --config config/mer_omni_av_tokens_evaluation.json infer \
  --dataset MER2023 --checkpoint runs/new_av_run/checkpoint_step_0010443.pt \
  --sequence-overflow truncate_right --output runs/new_av_eval/MER2023

# 在 requirements/evaluation.txt 对应的独立 Python 环境执行
python tools/mer_omni.py --config config/mer_omni_av_tokens_evaluation.json extract \
  --dataset MER2023 --predictions runs/new_av_eval/MER2023/predictions.jsonl \
  --gpu-memory-utilization 0.36 --max-model-len 4096 --max-num-seqs 8 \
  --output runs/new_av_eval/MER2023/extraction.json

python tools/mer_omni.py --config config/mer_omni_av_tokens_evaluation.json score \
  --dataset MER2023 --predictions runs/new_av_eval/MER2023/predictions.jsonl \
  --extraction runs/new_av_eval/MER2023/extraction.json --output runs/new_av_eval/MER2023/score.json
```

上面的 `python` 在生成时属于训练环境，在抽取和评分时属于独立评测环境。当前记录只完成 MER2023；MER2024 已暂停，213 条部分生成不构成正式成绩。上述命令不会自动启动 MER2024。

`truncate_right` 只在推理时覆盖超长策略，保留原生特征前缀，长度和训练校准不变；逐样本记录受影响模态，退出后恢复模型设置。MER2023 有 1 条、MOSEI 有 29 条超长；MER2024 无超长输入。训练默认仍为 `error`。

正式生成使用 seed=42、temperature=1、top_p=0.9、top_k=20、max_new_tokens=1200、max_length=2000、EOS `[151645,151643]`，并识别 `###`。抽取使用 vLLM 0.6.2 / Transformers 4.49.0、engine seed=0、sampling seed=None、temperature=0.7、top_p=0.8、repetition_penalty=1.05、max_tokens=512、batch=8。需要本机单进程 NCCL rendezvous 时可加 `--single-process-store`。

情感指标只过滤真值 0；预测 neutral/无效值转为 0，再按 `>0` 二分。OV 集合先对样本求 precision/recall 均值，再取调和平均，最后平均五个 wheel。缺失、重复预测和过期抽取缓存报错；九集聚合要求同一检查点与评测设置。

## 测试范围

`python -m unittest discover -s tests -v` 使用合成媒体、微型真实 Transformers Omni 和上传的原版快照，覆盖 CE 掩码/梯度、辅助路径跳过、尾批归一化、完整采样、断点恢复、补零/截断、频带校准及评分规则。不需要实际 3B 权重或数据集。

默认测试只依赖随库提供的参考文件，并核验其哈希。若自行准备了额外原版源码，可通过 `ATSUKO_REFERENCE_ROOT`、`AFFECTGPT_REFERENCE_ROOT` 环境变量显式启用额外函数体对照；不再探测原服务器目录。通过这些测试不能证明模型识别能力或整个实验与原作者完全等价。
