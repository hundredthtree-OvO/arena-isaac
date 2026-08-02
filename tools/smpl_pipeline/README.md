# SMPL-H / AMASS 离线资产流水线

## 目的

本目录用于在不影响 `toilet_benchmark`、HuNav、Replay 和 Isaac bridge 的情况下分析
AMASS 动作，并为后续 SMPL-H USD 角色制作准备可审计的动作片段。

当前阶段已完成：

```text
AMASS NPZ
  -> schema 校验
  -> 确定性运动索引
  -> SMPL-H 模型结构校验
  -> 稳定直行窗口自动筛选
  -> 120 Hz 到 60 Hz 整数抽样
  -> 纯 NumPy SMPL-H LBS 网格重建
  -> 初始 heading / 地面 / root trajectory 归一化
  -> 带时间采样的独立 USD 预览
```

这些工具仍然：

- 不启动 ROS 或 Isaac Sim；
- 不修改 `ros2isaacsim/person.py`；
- 不修改 `toilet_benchmark`；
- 不把 SMPL 角色接入 HuNav 或 Director；
- 不提交受限的人体 NPZ/USD 派生资产。

USD 导出仅调用 Isaac 安装内的 USD Python 库，不启动完整 Kit 或场景。

## 目录边界

```text
tools/smpl_pipeline/
├── analysis/       # AMASS I/O、索引、旋转、速度、转向和脚接触分析
├── builders/       # 标准 clip、preview NPZ、SMPL-H LBS 和 USD 构建
├── config/         # 可审计的动作库 seed recipe
├── runtime/        # motion backend、manifest 验证和隔离的 Isaac mirror
├── tests/          # 不依赖受限资产的回归测试
└── README.md

arena_assets/smpl/derived/
├── indexes/        # 原始 AMASS 索引
├── clips/          # 60 Hz 标准动作片段
├── manifests/      # 动作类别、方向、速度、接触和来源
├── previews/       # USD 构建前的中间数据
└── usd/            # 人工验收 stage
```

`analysis` 和 `builders` 都是离线工具；`runtime` 当前只接入独立 Isaac 空场景 mirror，尚未
接入 HuNav、Director 或 `toilet_benchmark` 主链路。

## 默认资产路径

```text
/home/stardust/resources/arena_ws/arena_assets/smpl/
├── CMU/CMU/               # AMASS CMU 动作，只读
├── smplh/                 # female/male/neutral SMPL-H，只读
└── derived/               # 本工具生成的索引、片段和报告
```

SMPL-H 和 AMASS 受非商业科研及禁止再分发条款约束。原始模型、原始动作和可还原人体
模型的派生资产不得进入公开仓库。

## 1. 检查 SMPL-H

```bash
cd /home/stardust/resources/arena_ws/src/arena/arena-isaac
python3 -m tools.smpl_pipeline.analysis.inspect_smplh
```

当前 AMASS 兼容模型应满足：

```text
52 joints
6890 vertices
13776 faces
16 shape parameters
```

## 2. 检查单条动作

```bash
python3 -m tools.smpl_pipeline.analysis.inspect_amass \
  /home/stardust/resources/arena_ws/arena_assets/smpl/CMU/CMU/91/91_02_poses.npz \
  --dataset-root /home/stardust/resources/arena_ws/arena_assets/smpl/CMU/CMU
```

输出包括帧率、时长、根节点路径长度、净位移、平均速度和静止帧比例。

## 3. 建立 CMU 索引

快速探索时跳过 2082 个文件的 SHA-256：

```bash
python3 -m tools.smpl_pipeline.analysis.build_motion_index --skip-sha256
```

正式索引：

```bash
python3 -m tools.smpl_pipeline.analysis.build_motion_index
```

默认输出：

```text
/home/stardust/resources/arena_ws/arena_assets/smpl/derived/indexes/cmu.json
```

## 4. 提取候选片段（可选）

以下命令只是示例区间。正式区间必须在速度、转角和脚接触分析后确定：

```bash
python3 -m tools.smpl_pipeline.builders.extract_motion_clip \
  /home/stardust/resources/arena_ws/arena_assets/smpl/CMU/CMU/91/91_02_poses.npz \
  --start-sec 2.0 \
  --end-sec 6.0 \
  --output /home/stardust/resources/arena_ws/arena_assets/smpl/derived/clips/91_02_candidate.npz
```

输出 NPZ：

- `source_trans`：原始世界 translation；
- `root_trajectory`：相对片段首帧的 root translation；
- `local_trans`：清除水平 translation、保留原始高度的 translation；
- `poses`、`betas`、`gender`、`dmpls`：对应裁剪数据。

`local_trans` 还不是完整的 in-place 动画，因为 root orientation 仍保留在
`poses[:, 0:3]`。运行时角色仍需进一步实现完整 heading/local pose 分解。

## 5. 生成独立 Isaac 预览

先用普通项目 Python 自动选择稳定直行窗口，并烘焙 60 Hz SMPL-H 网格：

```bash
cd /home/stardust/resources/arena_ws/src/arena/arena-isaac

python3 -m tools.smpl_pipeline.builders.prepare_preview \
  /home/stardust/resources/arena_ws/arena_assets/smpl/CMU/CMU/91/91_02_poses.npz \
  --output /home/stardust/resources/arena_ws/arena_assets/smpl/derived/previews/91_02_walk_preview.npz
```

需要固定时间窗口时可增加 `--start-sec 4.25 --duration-sec 2.0`。默认不写
`--start-sec` 时，工具按位移、速度、移动帧占比、朝向一致度和转角对 2 秒窗口排序。

再调用 Isaac 4.5 自带的 USD 库：

```bash
bash tools/smpl_pipeline/builders/export_preview_usd.sh \
  /home/stardust/resources/arena_ws/arena_assets/smpl/derived/previews/91_02_walk_preview.npz \
  /home/stardust/resources/arena_ws/arena_assets/smpl/derived/usd/91_02_walk_preview.usd
```

如果 Isaac 不在默认位置，可在命令前设置 `ISAAC_ROOT=/path/to/isaac-sim-4.5.0`。

当前已生成并通过结构检查的 stage：

```text
/home/stardust/resources/arena_ws/arena_assets/smpl/derived/usd/91_02_walk_preview.usd
```

它包含：

- `/World/SMPLHPreview`：121 帧、60 Hz 的 time-sampled mesh；
- `/World/Ground`：仅用于肉眼判断脚底高度的灰色参考地面；
- `/World/RootTrajectory`：绿色 root 平面轨迹；
- Z-up、米制单位、0 到 120 的 timeline。

## 6. Isaac Sim 人工验收

这一步只检查人体资产与动作，不运行 bridge、ROS、HuNav 或厕所场景：

1. 打开 Isaac Sim 4.5 GUI。
2. 使用 `File > Open` 打开
   `/home/stardust/resources/arena_ws/arena_assets/smpl/derived/usd/91_02_walk_preview.usd`。
3. 将 timeline 移到第 0 帧，按 `Play`。
4. 必要时选择 `/World/SMPLHPreview` 后按 `F` 聚焦。

应重点记录：

- 人体是否直立且朝大致 `+X` 前进，而非横移或倒置；
- 脚底是否基本贴近灰色地面，是否持续悬空或明显陷地；
- 四肢和手指是否存在爆炸、扭曲或左右肢体互换；
- 步态是否连续，是否有明显脚滑；
- 绿色轨迹是否与身体平移方向一致；
- 约 2 秒动作末尾是否平滑，暂不要求循环首尾无缝。

本预览是 **baked mesh 验证资产**，不是最终运行时角色。它验证的是 AMASS、SMPL-H
模型、坐标约定和 LBS 数学链路。通过后再实现可由外部 root 轨迹驱动的
`UsdSkel`/SMPL avatar，不能直接把每帧 6890 点更新作为多人运行方案。

## 7. 自动测试

测试只使用临时合成 NPZ，不读取或复制受限资产：

```bash
python3 -m unittest discover \
  -s tools/smpl_pipeline/tests \
  -p 'test_*.py'
```

已额外自动验证：

- 所有网格点为有限值；
- 首帧人体包围盒高度约 1.57 m；
- 最大逐帧顶点位移约 0.089 m；
- USD 可重新打开，具有 121 个 points 时间采样和 13776 个面。

## 8. 受控 UsdSkel 预览

第一轮 baked mesh 人工验证通过后，可以生成 root 位姿与局部步态分离的骨骼版本：

```bash
cd /home/stardust/resources/arena_ws/src/arena/arena-isaac

python3 -m tools.smpl_pipeline.builders.prepare_controlled_preview \
  /home/stardust/resources/arena_ws/arena_assets/smpl/CMU/CMU/91/91_02_poses.npz \
  --output /home/stardust/resources/arena_ws/arena_assets/smpl/derived/previews/91_02_controlled_preview.npz

bash tools/smpl_pipeline/builders/export_controlled_usd.sh \
  /home/stardust/resources/arena_ws/arena_assets/smpl/derived/previews/91_02_controlled_preview.npz \
  /home/stardust/resources/arena_ws/arena_assets/smpl/derived/usd/91_02_controlled_usdskel.usd
```

当前已生成：

```text
/home/stardust/resources/arena_ws/arena_assets/smpl/derived/usd/91_02_controlled_usdskel.usd
```

该 stage 使用 52-joint `UsdSkel`、SMPL-H skin weights 和独立
`UsdSkelAnimation`。时间轴为 60 Hz、0 到 360 帧：

| 帧 | 阶段 | 预期 |
|---:|---|---|
| 0–120 | `walk_x` | 沿 +X 自然行走 |
| 121–180 | `hold` | root 和局部姿态完全冻结 |
| 181–240 | `turn` | 位置不变，root yaw 转到 90° |
| 241–360 | `walk_y` | 保持新朝向并沿 +Y 恢复行走 |

脚接触诊断以 SMPL-H 左右脚 joint 的高度和速度计算，用于选择停止姿态；本次动作左右
接触帧占比约为 0.51 和 0.48。第一版为了保留 SMPL pose correctives，mesh points 仍有
time samples；但位置和 yaw 已由 root joint 控制，不再烘焙到世界空间顶点。

### Isaac Sim 人工验收

```bash
cd /home/stardust/resources/isaac-sim-4.5.0
./isaac-sim.sh
```

在 GUI 中使用 `File > Open` 打开
`/home/stardust/resources/arena_ws/arena_assets/smpl/derived/usd/91_02_controlled_usdskel.usd`，
选择 `/World/Avatar/Body` 后按 `F`，从第 0 帧播放。

需要重点确认：

- `0–120` 帧是否保持第一轮预览的自然步态；
- `121–180` 帧是否真正静止，不漂移、不摆腿、不旋转；
- `181–240` 帧是否原地转向，没有画圈和平移；
- `241–360` 帧是否沿 +Y 恢复，而不是继续沿 +X 或横向滑行；
- 转向后骨骼和 mesh 是否保持一致，没有网格撕裂；
- 全程脚底高度是否合理，恢复行走时是否出现明显相位跳变。

自动验证结果：

- Skeleton 和 animation query 均有效；
- 52 个 joint 在关键帧均可计算完整 skeleton transforms；
- 停止段根位置保持 `(2.846, 0.037)`，yaw 保持约 `-3.9°`；
- 转向结束 yaw 为 `90°`；
- 最后一帧 root 到达约 `(2.602, 2.883)`，yaw 约 `86.1°`；
- 6890 个顶点均具有 8 个归一化 joint influences。

## 受控预览通过后的阶段

1. 修正人工验收发现的脚滑、相位切换或坐标问题；
2. 将停止和转向切换改为 contact-aware clip transition；
3. 评估用 blend shapes 或 GPU deformation 替代 pose-corrective points samples；
4. 设计独立 `SmplUsdAvatarBackend`，保持与 `toilet_benchmark` 解耦；
5. 在空场景做 external root command 单角色 mirror；
6. 最后才进入多人性能和 HuNav/Replay 适配。

## 9. 自然 90° 转向对比

`91_02_controlled_usdskel.usd` 中的冻结整体转向只用于验证 root 控制，不作为最终动画。
CMU subject 83 trial 36 的官方描述是
`walk forward, turn 90 degrees left, walk forward`，本地 AMASS 对应：

```text
/home/stardust/resources/arena_ws/arena_assets/smpl/CMU/CMU/83/83_36_poses.npz
```

生成保持原始连续足步的自然转向预览：

```bash
cd /home/stardust/resources/arena_ws/src/arena/arena-isaac

python3 -m tools.smpl_pipeline.builders.prepare_natural_turn_preview \
  /home/stardust/resources/arena_ws/arena_assets/smpl/CMU/CMU/83/83_36_poses.npz \
  --clip-start-sec 1.5 \
  --turn-start-sec 4.0 \
  --turn-end-sec 5.8 \
  --clip-end-sec 8.0 \
  --output /home/stardust/resources/arena_ws/arena_assets/smpl/derived/previews/83_36_walk_turn_walk_preview.npz

bash tools/smpl_pipeline/builders/export_controlled_usd.sh \
  /home/stardust/resources/arena_ws/arena_assets/smpl/derived/previews/83_36_walk_turn_walk_preview.npz \
  /home/stardust/resources/arena_ws/arena_assets/smpl/derived/usd/83_36_walk_turn_walk_usdskel.usd
```

当前已生成：

```text
/home/stardust/resources/arena_ws/arena_assets/smpl/derived/usd/83_36_walk_turn_walk_usdskel.usd
```

时间轴为 60 Hz、0 到 390 帧：

| 帧 | 阶段 | 预期 |
|---:|---|---|
| 0–149 | `walk_in` | 沿 +X 进入转向点 |
| 150–257 | `natural_turn` | 通过真实换步完成约 90° 左转 |
| 258–390 | `walk_out` | 沿 +Y 离开 |

自动检查结果：

- 转向入口约 `(1.977, -0.167)`、yaw `7.6°`；
- 转向出口约 `(2.382, -0.017)`、yaw `90.6°`；
- 最后到达约 `(2.238, 1.716)`、yaw `91.9°`；
- 转向期间位置连续变化，不是冻结后整体旋转；
- 52-joint skeleton、animation query 和 skin binding 均有效；
- 左右脚接触帧占比约为 0.60 和 0.62。

### Isaac Sim 对比验收

打开 Isaac Sim 后依次加载：

```text
# 旧的 root 控制基线：预期能看到冻结姿态整体旋转
/home/stardust/resources/arena_ws/arena_assets/smpl/derived/usd/91_02_controlled_usdskel.usd

# 新的自然转向：本轮重点验收
/home/stardust/resources/arena_ws/arena_assets/smpl/derived/usd/83_36_walk_turn_walk_usdskel.usd
```

重点比较：

- 转向时是否能看到支撑脚和迈步脚交替，而非整个角色像转盘旋转；
- 身体、骨盆、腿和脚的 yaw 是否分阶段变化；
- 入弯和出弯是否连续，有无姿态跳变；
- 转弯半径是否适合作为厕所拐角动作素材；
- 是否存在脚底打滑、浮空、陷地或手臂穿过身体。

`09_12` 的 navigate 动作也已确认包含自然曲线运动，但其局部段会继续形成更大的 U 形
转向，因此保留为后续 forward/backward/sideways 导航动作素材，不作为本轮纯 90°
转向验收基线。

## 10. 最小 Motion Library

seed recipe 位于：

```text
tools/smpl_pipeline/config/motion_library_seed.json
```

当前动作集合：

| ID | 类别 | 来源 | 语义 |
|---|---|---|---|
| `walk_forward_83_37` | `walk/forward` | CMU `83_37` 转弯前 | 接触相位闭合的正常直行循环 |
| `start_walk_83_37` | `resume/forward` | CMU `83_37` 开头 | 完整静止准备、自然起步和直行 |
| `turn_right_90_83_37` | `turn/right` | CMU `83_37` | 自然减速并右转约 90° |
| `turn_left_90_83_36` | `turn/left` | CMU `83_36` | 原生自然减速并左转约 90° |
| `stop_walk_83_36` | `stop/stationary` | CMU `83_36` 末段 | 从约 0.98 m/s 自然减速至站立 |

`IDLE -> RESUME` 直接从 `start_walk_83_37` 第 0 帧开始，完整消费原始起步直行，只有该段
耗尽后才进入稳定步态循环。左右转 clip 截止在转向后的短恢复，不再携带质量较差的长距离
转后行走。停止使用 `83_36` 原生末段姿态；右转前减速只作为速度节奏参考，不把转向骨骼
姿态复制到直线停止。`133_11` 保留为后续步行/小跑混合研究素材。

构建标准 clip 和 manifest：

```bash
cd /home/stardust/resources/arena_ws/src/arena/arena-isaac

python3 -m tools.smpl_pipeline.builders.build_motion_library \
  --recipe tools/smpl_pipeline/config/motion_library_seed.json \
  --output /home/stardust/resources/arena_ws/arena_assets/smpl/derived/manifests/cmu_smplh_locomotion_seed.json
```

验证 schema、clip 存在性和 SHA-256：

```bash
python3 -m tools.smpl_pipeline.runtime.validate_motion_library \
  /home/stardust/resources/arena_ws/arena_assets/smpl/derived/manifests/cmu_smplh_locomotion_seed.json \
  --asset-root /home/stardust/resources/arena_ws/arena_assets/smpl
```

每条 manifest 记录：

- 原始 AMASS 相对路径、source SHA-256 和裁剪区间；
- 标准 clip 相对路径及 SHA-256；
- 类别、方向、官方动作描述；
- FPS、帧数、事件区间和时长；
- root 位移、yaw 变化、起止速度；
- 左右脚接触帧比例。

manifest 是未来 motion graph 的唯一动作发现接口；运行时不应扫描 `clips/` 或根据文件名
猜测动作语义。

## 11. Contact-aware Motion Graph 预览

最小 graph recipe：

```text
tools/smpl_pipeline/config/motion_graph_seed.json
```

组合顺序：

```text
turn_right_90_83_37
  -> quaternion blend
  -> stop_walk_83_36
  -> hold 1.0 s
  -> quaternion blend
  -> start_walk_83_37
```

构建与导出：

```bash
cd /home/stardust/resources/arena_ws/src/arena/arena-isaac

python3 -m tools.smpl_pipeline.builders.compose_motion_graph_preview \
  --graph tools/smpl_pipeline/config/motion_graph_seed.json \
  --manifest /home/stardust/resources/arena_ws/arena_assets/smpl/derived/manifests/cmu_smplh_locomotion_seed.json \
  --output /home/stardust/resources/arena_ws/arena_assets/smpl/derived/previews/right_turn_stop_resume_graph.npz

bash tools/smpl_pipeline/builders/export_controlled_usd.sh \
  /home/stardust/resources/arena_ws/arena_assets/smpl/derived/previews/right_turn_stop_resume_graph.npz \
  /home/stardust/resources/arena_ws/arena_assets/smpl/derived/usd/right_turn_stop_resume_graph.usd
```

transition selector 在右转尾部 2 秒与停止动作头部 0.5 秒内联合最小化：

- 22 个身体 joint 的局部旋转差；
- 平面速度差；
- 左右脚接触模式差。

合成器还会在停止动作末端自动选择 hold 姿态，评分同时考虑：

- root 平面速度和双脚速度；
- 双脚是否同时接触地面；
- 左右膝屈曲角；
- 骨盆到躯干的倾斜角。

当前重建结果：

- 右转 clip frame `155` → 停止 clip frame `28`；
- 速度差约 `0.1243 m/s`；
- 脚接触模式差为 `0`；
- 使用 0.12 秒 shortest-path quaternion blend，并锁定共同支撑脚；
- hold 采用 `83_36` 原始 frame `1020`：双脚接触、root 速度约
  `0.033 m/s`、平均膝屈曲约 `16.1°`、躯干倾斜约 `3.7°`；
- hold 到完整 `83_37` 起步 clip frame `4` 的速度差约 `0.0014 m/s`，脚接触模式一致，
  使用 0.2 秒姿态与速度过渡。

生成的 USD：

```text
/home/stardust/resources/arena_ws/arena_assets/smpl/derived/usd/right_turn_stop_resume_graph.usd
```

时间轴为 60 Hz、0 到 538 帧：

| 帧 | 阶段 |
|---:|---|
| 0–155 | 自然右转动作 |
| 156–162 | 支撑脚锁定的右转到停止 blend |
| 163–260 | 正常步行减速 |
| 261–320 | 直立姿态完全静止 |
| 321–332 | 静止到完整起步动作的 quaternion blend |
| 333–538 | `83_37` 完整自然起步与直行 |

自动连续性检查：

- hold 内部 root drift 为 `0`；
- 最大 root 单帧平面步长约 `0.021 m`；
- 右转→减速 blend 内共同右支撑脚速度小于 `0.00002 m/s`；
- hold→恢复 blend 边界最大 joint 变化约 `1.4°`；
- Skeleton、AnimationQuery 和 664 个 mesh time samples 均有效。

### Isaac Sim 人工验收

打开 `right_turn_stop_resume_graph.usd`，重点观察：

- 右转完成后切入减速动作时是否有身体或脚的跳变；
- 减速过程是否自然，而不是突然冻结；
- 437–496 帧是否完全静止且接近自然直立；
- 恢复时是否由支撑姿态自然迈出第一步；
- 是否出现脚滑、骨盆弹跳、网格撕裂或瞬时速度跳变。

## 12. 单角色命令运行时

独立运行时位于：

```text
tools/smpl_pipeline/runtime/avatar_backend.py
tools/smpl_pipeline/runtime/isaac_avatar_adapter.py
tools/smpl_pipeline/runtime/mirror_single_avatar.py
```

边界保持为：

- `SmplUsdAvatarBackend` 只读取 motion-library manifest，接收
  `set_command(speed_mps, yaw_rate_rps)` 并输出 root、yaw、52-joint rotations 和状态；
- 状态图为
  `IDLE -> RESUME -> WALK -> TURN_LEFT/TURN_RIGHT -> WALK -> STOP -> IDLE`；
- clip 切换联合使用姿态、速度和脚接触匹配；
- walk loop 在原周期尾部 7 帧内与下一周期头部重叠混合，不再额外插入冻结帧；
- 行走、转向和转后恢复之间的姿态混合继承目标动作的 root 速度，腿部换相期间不再把
  root 暂停在原地；
- 可兼容的切换锁定共同支撑脚；若锁脚需要把骨盆校正超过 `0.15 m`，则自动退化为
  短原地姿态过渡，避免角色跳变；
- 停止入口只在动作标注的自然减速区间内做姿态匹配，优先保持入口速度连续，并使用
  `0.12 s` 接触感知混合；接触校正的平面单帧位移限制为 `0.025 m`；
- `IsaacSmplAvatarAdapter` 只负责把 backend 输出写入 `/World/Avatar` 的 UsdSkel，
  不依赖 ROS、HuNav、`toilet_benchmark` 或现有 Isaac bridge。

当前 seed 对连续命令采用离散动作映射：

- `speed_mps <= 0.08`：自然减速并进入 `IDLE`；
- `speed_mps > 0.08`：自然起步并维持正常步行；
- `yaw_rate_rps < -0.25` 的下降沿：触发一次自然右转；
- yaw 回到阈值一半以内后，允许下一次右转。

当前动作仍使用 mocap 原始播放速率；`speed_mps` 已控制停止/行走状态，但尚未进行任意速度
time-warp。这样可先验证状态边界不滑步，避免速度缩放与接触问题混在同一阶段。

### 自动验证

```bash
cd /home/stardust/resources/arena_ws/src/arena/arena-isaac

python3 -m unittest discover \
  -s tools/smpl_pipeline/tests \
  -p 'test_*.py'

tools/smpl_pipeline/runtime/run_single_avatar_mirror.sh \
  --headless \
  --exit-after-sec 24
```

固定序列为：

| 仿真时间 | 命令 |
|---:|---|
| 0–1 s | 初始站立 |
| 1–5 s | 完整起步并直行 |
| 5–5.35 s | 右转触发 |
| 5.35–13 s | 完成右转并继续直行 |
| 13–17 s | 自然停止并保持 |
| 17–22 s | 恢复直行 |
| 22 s 后 | 最终停止 |

离线状态序列测试结果：

- 完整经过 `IDLE/RESUME/WALK/TURN_RIGHT/STOP`；
- 最大 root 单帧平面位移约 `0.021 m`；
- 持续步行阶段 5% 分位速度约 `0.80 m/s`，周期回环不再产生零速帧；
- 最终 `IDLE` 连续 100 帧 root drift 为 `0`；
- in-memory USD adapter smoke 已成功创建并更新
  `/World/Avatar/Body` 与 `/World/Avatar/Animation`；
- 宿主机 RTX 4070 SUPER 上完整 24 秒 headless SimulationApp 序列正常退出。

### Isaac Sim GUI 人工验收

```bash
cd /home/stardust/resources/arena_ws/src/arena/arena-isaac

tools/smpl_pipeline/runtime/run_single_avatar_mirror.sh
```

窗口会保持运行，完成固定序列后角色最终停止；关闭 Isaac 窗口或在终端按 `Ctrl+C` 退出。
默认相机只设置一次初始视角，之后可用右键和 WASD 自由移动，不会被运行时拉回。需要连续
跟随角色时显式运行：

```bash
tools/smpl_pipeline/runtime/run_single_avatar_mirror.sh --follow-camera
```

重点观察：

- `RESUME -> WALK` 是否自然迈步，walk loop 是否存在周期性顿挫；
- 5 秒后的右转是否由腿部迈步完成，而非冻结整体旋转；
- `TURN_RIGHT -> WALK` 是否出现 yaw 跳变或横向滑动；
- 13 秒后的 `WALK -> STOP -> IDLE` 是否自然减速、支撑脚稳定；
- 17 秒恢复时是否无蹲起、瞬移和脚底滑动。

### 流畅度曲线

难以仅凭观感描述顿挫时，可生成与固定命令序列逐帧对齐的诊断：

```bash
python3 -m tools.smpl_pipeline.analysis.analyze_runtime_smoothness \
  --output-prefix /tmp/smpl_runtime_smoothness
```

输出 `/tmp/smpl_runtime_smoothness.svg` 和同名 CSV。SVG 从上到下显示瞬时 root 速度、完整
步态周期平均速度和 jerk 绝对值，背景色对应动作状态。周期平均速度用于过滤人体正常落脚造成的
骨盆速度起伏；只有固定发生在 phase 回绕处且伴随视觉跳变的尖峰才应判为 loop 边界问题。
CSV 额外保存命令速度、动画限加速度后的速度、gait phase、motion id 和 clip frame，可以把
视觉上不自然的仿真时间精确映射回源动作帧。

## 13. 多速度步态候选

连续调速不能只缩放 root 或只快放动画。当前先建立三档动作候选，再在人工验收后实现相同
gait phase 上的步幅与步频联合插值：

| 档位 | CMU 来源 | 中值速度 | 用途 |
|---|---|---:|---|
| slow | Subject 37 Trial 1 `slow walk` | 约 0.89 m/s | 慢速行走候选 |
| normal | Subject 83 Trial 37 | 约 0.97 m/s | 当前已验收主步态 |
| fast | Subject 38 Trial 1 `walk` | 约 1.24 m/s | 快速行走候选 |

自动候选扫描命令：

```bash
python3 -m tools.smpl_pipeline.analysis.find_gait_speed_candidates
```

输出：

```text
/home/stardust/resources/arena_ws/arena_assets/smpl/derived/indexes/cmu_gait_speed_candidates.json
```

扫描结果只用于预筛。root 速度符合要求但脚接触异常、带负重或属于 stylized motion 的片段
不能直接进入 runtime。

当前人工验收 USD：

```text
/home/stardust/resources/arena_ws/arena_assets/smpl/derived/usd/gait_slow_candidate.usd
/home/stardust/resources/arena_ws/arena_assets/smpl/derived/usd/gait_normal_candidate.usd
/home/stardust/resources/arena_ws/arena_assets/smpl/derived/usd/gait_fast_candidate.usd
```

manifest 中的 walk 动作现已包含 `gait.nominal_speed_mps`、周期时长、单周期位移和已验证
缩放范围；clip NPZ 同时包含 `[0, 1]` 的 `gait_phase`。runtime 已在统一 gait phase 上混合
slow/normal/fast 三档，并同时插值周期频率和单周期 root 位移；速度变化受加速度上限约束，
超出样本区间时只允许在 manifest 的小范围内缩放，不用单纯快放动画来覆盖大速度范围。

离线检查三档速度变化：

```bash
python3 -m tools.smpl_pipeline.analysis.analyze_runtime_smoothness \
  --scenario speed-ramp \
  --output-prefix /tmp/smpl_speed_ramp_smoothness
```

Isaac Sim 空场景人工检查：

```bash
tools/smpl_pipeline/runtime/run_single_avatar_mirror.sh --scenario speed-ramp
```

场景依次执行 `0.75 -> 0.95 -> 1.20 -> 0.75 m/s -> STOP`。重点观察速度切换时步频与
步幅是否同步变化、相位是否连续，以及脚底是否出现持续滑移。目前尚未加入逐脚 IK/contact
lock，因此不同表演者动作混合区间仍需以人工视觉验收为准。

实时键盘命令 mirror：

```bash
tools/smpl_pipeline/runtime/run_single_avatar_mirror.sh --scenario interactive
```

| 按键 | 作用 |
|---|---|
| `I` | 开始或恢复行走 |
| `K` / `Space` | 自然停止 |
| `J` / `L` | 触发左转 / 右转 |
| `1` / `2` / `3` | 选择 `0.75 / 0.95 / 1.20 m/s` |

控制键避开 `WASD`，因此仍可使用 Isaac 原生视角操作。interactive 模式只验证单角色实时
运动后端，不订阅 ROS，也不接入 HuNav、Director 或厕所 benchmark。

`IDLE -> RESUME` 使用独立的 `0.30 s` smoothstep 姿态融合，吸收静止动作 `83_36` 与起步
动作 `83_37` 的骨盆和支撑腿差异。`RESUME -> WALK` 在完整 gait cycle 中按姿态、速度与
脚接触匹配接入相位，不再固定跳回 walk 第 0 帧；其他动作边界仍保持较短的默认融合时间。
