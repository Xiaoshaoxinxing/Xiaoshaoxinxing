# 当前 SFT 运行协议

更新于 **2026-10-26**。当前 LoRA 基线和官方 AffectGPT face 九集评测均已完成。

## 环境

训练及 Omni 推理：`.venv`，Python3.10、torch2.4/cu121、transformers4.52.4。官方 face 及标签提取：`.venv-eval`，transformers4.49.0、vLLM0.6.2。具体依赖见[requirements](../requirements)。保留现有环境和模型位置。

数据与模型是本地资产，不随 GitHub 源码上传。正式数据清单在 `data/manifests/`，固定训练目标及原生媒体元数据在 `data/manifests/sft/`。[数据来源记录](results/sft-data-provenance.json)给出 SHA-256 和目标分配规则。

## 只读检查

```bash
.venv/bin/python -B tools/check_sft_workspace.py
.venv/bin/python -B tools/peer_lora_epoch2.py check
.venv/bin/python -B tools/search_peer_lora_checkpoints.py check
.venv-eval/bin/python -B tools/run_affectgpt_official_face_merunibench.py check
```

第一条核验三个 checkpoint 的 adapter／优化器／RNG、固定筛选记录、九集预测及缓存重评分，并检查 face 的来源、权重和完成结果。它不训练、不生成答案、不修改正式结果。归档核验与测试日志在 `reports/sft_transition_20261026/`。

## 训练和评测入口

| 文件 | 作用 |
| --- | --- |
| `tools/peer_lora_epoch2.py` | `prepare / check / preflight / train / supervise`；当前固定 LoRA 运行 |
| `tools/search_peer_lora_checkpoints.py` | 固定子集筛选、读取选择记录、选中权重的全量九集评测 |
| `tools/evaluate_peer_lora_epoch2.py` | Qwen 推理、标签提取和正式评分的公共实现 |
| `tools/run_affectgpt_official_face_merunibench.py` | 官方 face 的准备、筛选和全量评测 |

当前配置绑定已完成运行。新实验必须登记独立配置、数据身份、审计和输出目录；当前入口不会覆盖原训练产物。训练 supervisor 只衔接当前 LoRA 的筛选／评测流程，已移除实验 A 的自动评测。已完成的全量评测再次进入 pipeline 时只验证已有预测。

活动 checkpoint：step31327、step34000、step62654。其他 30 个完整 checkpoint 位于本地归档，收据保留在当前运行目录。筛选排名可核验；重新生成某个归档候选的答案前需恢复其权重。

## 数值协议

MER-Caption+ 31,327 个唯一视频，每条解释／标签任务分配固定，两轮重复同一目标。无验证集划分。本轮 LoRA 训练配方见[配置](../config/peer_lora_epoch2.json)，数据目标来自原 A 分配，但已独立为数据产物，不读取 A 的实验目录。

LoRA：两帧图像、独立音频与字幕；贪心生成，最多 256 个新 token，原生 im_end 终止。face：官方八帧人脸、音频采样及 sampled generation。两者使用同一套官方数据划分、标签提取及评分函数。完整生成参数见各自配置。

九集包括 MER2023、MER2024、MELD、IEMOCAPFour、CMUMOSI、CMUMOSEI、SIMS、SIMSv2、OVMERDPlus，共 12,799 条。主指标为四 HR、四 weighted F1（按官方规则排除真实中性值）及 OV 五轮 Fₛ，九项等权平均。

两模型均在同一批 180 条测试样本上选一个 checkpoint，再评测完整九集；需要注明测试子集选模偏差。OV-MERD+ 全量 532 条含 99 条训练重叠，非重叠子集结果仅作补充。

## Atsuko 与 RL

公共组件位于 `mer_omni/sft/`，其 CPU 回归覆盖输入／标签隔离、梯度、校准中断恢复、优化器及 checkpoint 恢复。尚未实现和训练当前两帧 LoRA 配方的 Atsuko 接入实验。旧 B3 的十六帧配方和旧校准保存在历史归档，新结构必须重新做训练集校准和真实 GPU 预检。

RL 尚未启动；当前不创建空运行目录或未经验证的训练配置。
