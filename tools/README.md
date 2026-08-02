# Arena Isaac 离线工具

该目录保存不属于 ROS 运行节点、Isaac bridge 热路径或场景 profile 的离线工具。

## 约束

- 工具不得在导入时启动 ROS 或 Isaac Sim。
- 原始受限资产和派生人体资产不得提交到仓库。
- 工具输出默认写入 `/home/stardust/resources/arena_ws/arena_assets/`。
- 进入 `ros2isaacsim` 运行链路前，必须先在这里完成离线检查和独立验证。

## SMPL/AMASS

[`smpl_pipeline/README.md`](smpl_pipeline/README.md) 记录 SMPL-H、AMASS CMU 动作索引、
片段提取、纯 NumPy SMPL-H 重建、独立 USD 预览及人工验收流程。

当前阶段与 `toilet_benchmark` 完全隔离，不修改 Director、HuNav、Replay 或 Isaac
行人服务。
