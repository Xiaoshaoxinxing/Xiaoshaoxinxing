# 模型资源

模型权重在本地准备。仓库保存本说明和 [AffectGPT 七权重下载清单](AffectGPT-face/official-files.json)，不包含模型参数或下载分片。

| 本地位置 | 来源 | 用途 |
| --- | --- | --- |
| `models/Qwen2.5-Omni-3B/` | [Qwen/Qwen2.5-Omni-3B](https://huggingface.co/Qwen/Qwen2.5-Omni-3B) | 当前 LoRA 的 Thinker 基座 |
| `models/Qwen2.5-7B-Instruct/` | [Qwen/Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) | 标签提取和评分模型 |
| `models/AffectGPT-face/` | [MERChallenge/AffectGPT](https://huggingface.co/MERChallenge/AffectGPT) | 官方 face 的 epoch30、35、40、45、50、55、60 |

## Qwen 模型

将对应 Hugging Face 仓库的模型分片、分片索引、配置、tokenizer 和 processor 文件完整下载到上表位置，或把已有完整目录链接到该位置。只保存权重分片不足以加载模型。

当前配置默认离线读取本地资产。训练和官方评分使用不同环境，依赖版本见[运行协议](../docs/protocol.md#环境)；本轮 LoRA adapter 保存在 `runs/`，基座权重保存在本目录。

## 官方 AffectGPT face

七个 checkpoint 固定来自 revision `9d82d878cc6f5a530af01c68319a27362c29dc07`。在已准备环境的项目根目录执行：

```bash
.venv/bin/python -B tools/download_affectgpt_face_official.py
```

该脚本读取仓库中的 `official-files.json`，下载到 `models/AffectGPT-face/`，并校验文件大小和 SHA-256。实际 `.pth`、下载分片及下载收据继续被 Git 忽略。

face 还需要 Qwen2.5-7B-Instruct、CLIP ViT-L/14 和 Chinese HuBERT Large。来源见固定版本的[官方说明中的 Model Zoo](../baselines/affectgpt_official_face/upstream/README.md)，加载路径见[官方 config.py](../baselines/affectgpt_official_face/upstream/config.py)。这些骨干模型须能从 `baselines/affectgpt_official_face/upstream/models/` 访问。

现有 face 运行依赖本机共享模型目录，`prepare` 还调用本机的 `/data/user02/AffectGPT/verify_model_integrity.py`。在其他机器启动新运行，需要先配置骨干模型和完整性校验入口，并准备数据清单及 checkpoint 筛选计划；仅下载七个 `.pth` 不能直接运行完整评测。当前整理保留了原运行身份，未修改这套加载流程。
