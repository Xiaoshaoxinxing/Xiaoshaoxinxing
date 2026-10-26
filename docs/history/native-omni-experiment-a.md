# 历史实验 A

实验 A 已于 2026-10-26 从活动目录退役。当前 SFT 基线为 Qwen LoRA 的 step34000，详见[当前状态](../experiment-status.md)。

A 使用 Qwen2.5-Omni-3B 原生音视频输入、最多 16 帧、LoRA r16、31,327 条样本的一轮训练，共 10,443 步。正式结果：MER2023 HR 72.073%，MER2024 HR 75.044%。这些成绩使用当时的固定最终权重与生成设置，不等同于当前 LoRA 配方的结果。

A 的最终 checkpoint 约 343 MiB，仍完整保留。原配置、日志、预测、审计和源码快照位于本地 `.local-archive/sft-transition-20261026/`；具体旧、新路径见[归档索引](../results/archive-index-20261026.json)。

A 中仍被使用的加载、回答掩码、原生输入和状态恢复函数已提取为 `mer_omni/sft/` 公共组件。固定解释／标签分配原样保存在 `data/manifests/sft/mercaptionplus-target-plan.json`，当前训练不依赖 A 的运行目录。

历史源码和收据保持原样。若需要重新评测 A，应恢复当时的配置和源码身份；新实验不应重新引入对 A 目录的隐式依赖。
