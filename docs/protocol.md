# 当前模型与运行协议

本文件说明截至2026-10-22的实现。当前保留设置为rank16、16帧、zero_pad、优化配方修正的一轮/三轮版本：`config/mer_omni_av_tokens_frames16.json`与`config/mer_omni_av_tokens_frames16_epoch3.json`，评测使用同名 `_evaluation.json`。历史均值/四段音频与resample/LoRA64对照继续保留配置；各自结果和限制见[实验进展](experiment-status.md)。

原基线使用 `mer_omni_atsuko_padding.json` / `mer_omni_evaluation.json`；字幕对照使用 `mer_omni_subtitle_ablation.json` / `mer_omni_subtitle_evaluation.json`；前端末位独立音视频对照使用 `mer_omni_av_tokens_ablation.json` / `mer_omni_av_tokens_evaluation.json`。统一 CLI 和 `train.sh` 的默认配置仍为原基线，复现各版本必须显式传入 `--config`。不同版本的检查点和提示设置不可混用。

## 当前16帧方案：优化与运行

前端组按可训练参数名 `frontend.`、`fusion_in.`、`audio_in.`、`video_in.` 前缀判定，其余可训练参数归LoRA；各组再按原decay/no_decay规则区分，`ndim<2`或名字含bias/ln/bn时不衰减。设置lr_frontend后，组学习率=各组base_lr×由lr/min_lr/warmup_lr计算的共同日程因子，warmup_lr也按同一比例作用于前端。当前min_lr=lr，warmup后因子恒为1。grad_clip存在时，在backward与非有限梯度检查后、optimizer.step前对所有可训练参数裁剪。两个新键可选且必须有限正数，缺省时兼容旧配置。

当前一轮设置：seed42、batch1、累积3、warmup500、lr=min_lr2e-5、warmup_lr1e-6、lr_frontend1e-4、grad_clip0.6、weight_decay0.05。3epoch仅预算改变；rank64仅容量和对应校准变化，r64/alpha128保持alpha/r=2。训练日志每步记录LoRA `lr`和 `lr_frontend`。

以下为**新环境复现示例**，不是重启已完成本地运行。已有且签名正确的校准直接复用；1epoch和3epoch共用frames16校准，无需重复执行calibrate。

```bash
# 配置中的模型、训练清单和媒体准备好后
python tools/mer_omni.py --config config/mer_omni_av_tokens_frames16.json preflight
# 仅在对应校准尚不存在时执行
python tools/mer_omni.py --config config/mer_omni_av_tokens_frames16.json calibrate --device cuda:0
python tools/mer_omni.py --config config/mer_omni_av_tokens_frames16.json train \
  --device cuda:0 --output runs/new_frames16_run --log-every 10 --checkpoint-every 1000
# 3epoch另开目录，使用frames16_epoch3.json，总步数31329
```

1epoch最终10443与3epoch最终31329均按日程预先确定；不根据测试分数挑checkpoint。对应评测配置的omni/spectral/mcm/train必须与权重一致。

```bash
# 在训练环境执行推理
python tools/mer_omni.py --config config/mer_omni_av_tokens_frames16_evaluation.json infer \
  --dataset MER2023 --checkpoint runs/new_frames16_run/checkpoint_step_0010443.pt \
  --device cuda:0 --sequence-overflow truncate_right --output runs/new_frames16_eval/MER2023
# 在requirements/evaluation.txt对应的环境执行抽取和评分
python tools/mer_omni.py --config config/mer_omni_av_tokens_frames16_evaluation.json extract \
  --dataset MER2023 --predictions runs/new_frames16_eval/MER2023/predictions.jsonl \
  --output runs/new_frames16_eval/MER2023/extraction.json \
  --gpu-memory-utilization 0.36 --max-model-len 4096 --max-num-seqs 8 --single-process-store
python tools/mer_omni.py --config config/mer_omni_av_tokens_frames16_evaluation.json score \
  --dataset MER2023 --predictions runs/new_frames16_eval/MER2023/predictions.jsonl \
  --extraction runs/new_frames16_eval/MER2023/extraction.json \
  --output runs/new_frames16_eval/MER2023/score.json
```

3epoch对应替换为frames16_epoch3_evaluation配置及31329权重，输出使用另一个新目录。MER2024需明确指定MER2024及独立目录，不能混用预测缓存。三个阶段顺序运行，先前模型进程退出后再运行标签抽取，具体可用资源取决于机器与其他任务。

## 只读训练侧诊断

```bash
# 全量31327条，不加--limit；视频ffprobe使用16个线程并行
python tools/audit_native_sequence_lengths.py \
  --manifest data/manifests/mercaptionplus.train.jsonl \
  --tokenizer models/Qwen2.5-Omni-3B --max-frames 16 \
  --output reports/native_sequence_lengths.json
# 固定训练清单前256条；stdout为JSON，stderr为进度，不写runs
python tools/profile_router_weights.py \
  --checkpoint runs/new_frames16_run/checkpoint_step_0010443.pt --device cuda:0
```

长度审计保留旧输出字段，并新增video的nb_frames/decoded_frames、差异计数、时长与 `min(max_frames,frames)//2` 时间位置数；同时计算8帧与16帧，输出min/p50/p90/p95/p99/max及候选L的溢出表。文本tokenizer与soundfile音频估计保持原路径，不加载模型权重。可选--limit用于调试，本文全量结果没有使用它。

路由工具只检查训练样本，恢复前端buffer和校准，eval模式关闭dropout；熵按每条样本计算，单位nats，标准差ddof=0。模态均值>0.8或逐样本最大权重中位数>0.95触发警告，三个band_router分别报告，不把频带选择集中等同于模态塌缩。需要持久化结果时显式重定向stdout到自己的文件。

## 模型边界

Omni-3B 的视觉编码器、音频编码器、文本 embedding 和原始语言权重冻结；语言主干使用 LoRA（r=16、alpha=32、dropout=0.05）。字幕经过 token embedding，音频保留原生时序，视频仅在每个时间位置内平均空间 patch。

当前两组保留方案将三模态独立右补零至96/50/704，不做词级对齐。另有显式omni_resample插值对照；该模式不允许sequence_overflow键，与zero_pad配置不可混用。kernel=5 的卷积投影到 256 维后，序列长度为 92/46/700。谱基和三频带重建沿用 Atsuko，等能量边界在全部 MER-Caption+ 训练输入上统计并冻结；模型不从测试批次重新估计边界。

频带路由后，三模态分别经过 4 层、8 头 Transformer，读取末时间位置。`router_input=pre_attn` 时，模态路由决策使用注意力前序列均值，融合对象仍是注意力后的末位置。融合向量经映射成为一个 Thinker 输入 token。

`prompt.include_subtitle=true` 时，原始字幕另行进入 Thinker 提示词，前端文字分支保持。`prompt.include_av_tokens=true` 且来源为默认 `frontend` 时，从 `frontend.modalities` 取音频索引 2 和视频索引 1，分别经可训练 `Linear(256, 2048)` 得到一个独立 token。它们仍经过完整频谱前端，仅绕过最后的模态加权。原基线不启用这两个开关。

新增投影在原有模块之后初始化，并保存/恢复 PyTorch RNG，使共有参数初始值与原有实验一致。完整训练时各实验从底座和 seed=42 初始化开始。推理会核验检查点的字幕和音视频入口设置，拒绝跨版本混用。

新增原生音视频对照使用 `config/mer_omni_native_av_ablation.json` / `config/mer_omni_native_av_evaluation.json`，显式设置 `prompt.av_token_source=native_mean`。缺省值 `frontend` 保留原有行为。此对照保留融合路径、原始字幕、提示位置及三个多模态 token 数量，仅替换独立音视频入口：冻结编码器的每条未补零原生序列 → 按时间求均值 → 各自无 bias 的 `Linear(2048,256)` → 原有 `Linear(256,2048)`。不增加激活、归一化或 dropout。

新映射在原音视频投影之后、同一 RNG 保护范围内初始化；共有权重和随机数流保持。新增 1,048,576 个可训练参数，总数从 49,588,108 增至 50,636,684，增加约 2.11%。这是整个序列前端的旁路对照，包含参数量变化且均值会丢失时序，不能单独认定补零、卷积或末位置读取的因果贡献。融合前端的配置和初始投影不变，因此生产校准指纹检查通过时复用原频带。

推理时 `truncate_right` 仍只影响融合前端；原生独立音视频均值使用编码器输出的全部有效位置。新推理收据会单列这一处理范围。推理配置和检查点的 `av_token_source` 不同将报错；旧检查点不能作为新结构的继续训练权重。

四段音频对照只进一步设置 `prompt.audio_token_count=4`，默认 `1` 保留此前单均值行为。该选项只允许 `1` 或 `4`；四段模式要求 `av_token_source=native_mean`。原始音频的 T 个有效位置按顺序分成 `min(4,T)` 个连续非空段，各段长度最多相差 1，多余位置依次分给前面的段。每段取均值，并共用现有 `audio_native_in` 与 `audio_in` 投影；段之间不共享均值，也不补零、复制位置或插值。

LLM 提示中的音频部分包含这些按时间顺序排列的 token；视频仍是一个原生全段均值 token，融合路径仍是一个 token，字幕保持完整。参数、初始化及频带校准不变，可训练参数仍为 50,636,684。提示最多增加三个位置，因此这是音频汇聚粒度对照，也改变提示长度，不能单独测出时序顺序或补零的作用。

文本预检查按最多四个音频 token 计数；实际组装按每条样本的非空段数计数，并对所有提示位置屏蔽监督。原有单 token 配置与冻结源码的初始化、loss 和生成逐值兼容。推理及训练恢复均核验 token 数设置；即使参数形状相同，也拒绝用单 token 配置直接评测四段检查点，或反过来混用。

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

上文早期字幕/独立token对照的谱前端与共有投影初始化一致，可复用原训练集校准；新环境需先校准。当前16帧、resample、LoRA64各自使用对应签名校准，更改omni/spectral/seed或前端投影后不能直接复用旧文件。仅改train优化配方或epochs不改变model_signature。周期检查点在独立文件中保存，不中断训练循环。

原生音视频对照同样从底座重新开始、训练完整一轮，统一入口为：

```bash
python tools/mer_omni.py --config config/mer_omni_native_av_ablation.json train \
  --device cuda:0 --output runs/new_native_av_run --log-every 10 --checkpoint-every 1000
```

服务器登记运行使用 `tools/run_native_av_ablation.py`，依赖本机 `reports/native_av_ablation_20261016/` 中的方案、测试及 GPU 预检收据，训练完成后核对与独立音视频对照的样本顺序、学习率和完整覆盖。该调度器只训练，不自动启动任何 benchmark；MER2024 继续暂缓。

四段音频对照同样从底座重新训练完整一轮，统一入口为：

```bash
python tools/mer_omni.py --config config/mer_omni_audio_segments_ablation.json train \
  --device cuda:0 --output runs/new_audio_segments_run --log-every 10 --checkpoint-every 1000
```

服务器调度器为 `tools/run_audio_segments_ablation.py`，依赖本机 `reports/audio_segments_ablation_20261017/` 的登记与预检收据，只训练并检查覆盖和对照顺序，不自动评测。旧实验登记依赖当时的源码哈希，不能用已改变的源码直接重启旧调度方案；各轮修改前的源码已归档，公开复现使用对应配置及统一入口。

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

上面的 `python` 在生成时属于训练环境，在抽取和评分时属于独立评测环境。上述例子对应原8帧AV-token版本，其MER2024只有213条历史部分生成，不能作为正式成绩。16帧rank16一轮已完整评测MER2024共1169条；其他版本不能复用其成绩。命令只评测显式指定的数据集。

`truncate_right` 只在推理时覆盖超长策略，保留原生特征前缀，长度和训练校准不变；逐样本记录受影响模态，退出后恢复模型设置。MER2023 有 1 条、MOSEI 有 29 条超长；MER2024 无超长输入。训练默认仍为 `error`。

正式生成使用 seed=42、temperature=1、top_p=0.9、top_k=20、max_new_tokens=1200、max_length=2000、EOS `[151645,151643]`，并识别 `###`。抽取使用 vLLM 0.6.2 / Transformers 4.49.0、engine seed=0、sampling seed=None、temperature=0.7、top_p=0.8、repetition_penalty=1.05、max_tokens=512、batch=8。需要本机单进程 NCCL rendezvous 时可加 `--single-process-store`。

情感指标只过滤真值 0；预测 neutral/无效值转为 0，再按 `>0` 二分。OV 集合先对样本求 precision/recall 均值，再取调和平均，最后平均五个 wheel。缺失、重复预测和过期抽取缓存报错；九集聚合要求同一检查点与评测设置。

## 测试范围

`python -m unittest discover -s tests -v` 使用合成媒体、微型真实 Transformers Omni 和上传的原版快照，覆盖 CE 掩码/梯度、辅助路径跳过、尾批归一化、完整采样、断点恢复、补零/截断、频带校准及评分规则。不需要实际 3B 权重或数据集。

默认测试只依赖随库提供的参考文件，并核验其哈希。若自行准备了额外原版源码，可通过 `ATSUKO_REFERENCE_ROOT`、`AFFECTGPT_REFERENCE_ROOT` 环境变量显式启用额外函数体对照；不再探测原服务器目录。通过这些测试不能证明模型识别能力或整个实验与原作者完全等价。
