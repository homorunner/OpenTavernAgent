# TavernAgent v2

独立的 Rust PPO-Clip + GAE 训练项目，使用 FP32 推理。
默认 CPU 后端；`--features cuda` 构建可选的 FP32 CUDA 前向、反向与 AdamW，无需 PyTorch。

实现说明、训练与评估命令、并发配置、实验结果和轨迹分析统一见
[docs/TAVERNAGENT_V2.md](../../docs/TAVERNAGENT_V2.md)。
