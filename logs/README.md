# 运行日志

保存训练、checkpoint 筛选、模型推理和标签提取过程的终端日志。脚本运行时会按实验名称创建子目录；GitHub 只保存本说明。

| 位置 | 用途 |
| --- | --- |
| `training/<实验名>/` | 训练 supervisor、预检及训练终端输出 |
| `evaluation/<实验名>/` | 筛选、全量推理、评分及队列输出 |

日志用于定位任务状态和错误；正式 checkpoint、完成标记和评分文件保存在 [runs/](../runs/README.md)。公开实验结果见[结果索引](../docs/results/experiments-20261026.json)。
