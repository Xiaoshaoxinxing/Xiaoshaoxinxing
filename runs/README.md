# 运行产物

每个实验使用独立子目录，保存请求和配置快照、训练日志、checkpoint、筛选结果、预测及评分。GitHub 仅发布本目录说明；实际运行产物保存在本地，可公开的汇总见 [docs/results/](../docs/results)。

| 本地运行 | 内容 |
| --- | --- |
| `peer_lora_epoch2_run02/` | 已完成的 Qwen LoRA SFT；保留 step31327、step34000、step62654 |
| `affectgpt_official_face_screen20_run01/` | 七个官方 face checkpoint 的筛选及选中 epoch50 的九集结果 |

Qwen 全量结果在 `peer_lora_epoch2_run02/evaluation_selected_step34000_merunibench_greedy_s20/`。checkpoint 目录同时保存 adapter、优化器状态、随机状态及校验收据；恢复训练需要完整 checkpoint。

其他 LoRA checkpoint 和历史 A/B3 产物保存在本地归档。新实验使用新的配置及输出目录，避免覆盖完成记录。目录恢复方式和当前入口见[运行协议](../docs/protocol.md)。
