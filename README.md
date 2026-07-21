# arena-isaac：PhysX differential contact navigation

后续开发主链路为 **`physx_diff_contact`**：四个轮关节由 PhysX articulation
驱动，机器人运动来自轮地接触和纵向/侧向轮胎力，不直接写 root pose，也不使用
voxel guard 代替底层碰撞。该链路同时启用 RTX LiDAR、People/AnimGraph 服务和
`toilet_benchmark` 行人事件编排。

原 **`physx_wheels` + `rtx_scan`** 链路继续保留，作为稳定兼容和回归对照，不再是
后续轮地物理开发的主链路。

## 1. 当前设计

控制链：

```text
键鼠 /cmd_vel                  -> 使用 vx/wz，忽略 vy
手柄 /cmd_vel_gamepad_diff     -> 差速 vx/wz 映射（vy 强制为 0）
  -> 同时运行时按住手柄 deadman 优先
  -> velocity smoother / acceleration limiter
  -> 4 个 wheel joint velocity targets
  -> PhysX contact + longitudinal/lateral tire force
  -> actual /odom + odom->base_link TF
```

保留内容：

```text
/odom
/tf: odom -> base_link
/tf_static: base_link -> front_laser_link, base_link -> rear_laser_link, stamp=0
/front_scan
/rear_scan
/cmd_vel_applied
manual LiDAR mount: front_x=0.30, rear_x=-0.30, z=0.20
/isaac/spawn_pedestrian
/isaac/move_pedestrians
/isaac/pedestrian_states
```

关键变化：

```text
physx_diff_contact：真实读取 articulation 状态，以轮速目标、轮地接触和轮胎力驱动。
physx_wheels：保留 voxel guard 和旧 root-motion 辅助，作为回退基线。
```

## 2. 重要说明

`physx_diff_contact` 中 `voxel guard` 默认关闭：

```text
PhysX CollisionAPI：底层接触、阻挡、防穿模。
轮胎力模型：分别限制纵向牵引力和侧向力，使差速转向可控。
场景修复：只恢复 profile 白名单内门体和门框的碰撞。
```

`physx_wheels` 回退模式仍启用 voxel guard，用于和旧导航表现对照。

机器人仍保留 semantic footprint：

```text
不是给 PhysX 碰撞用；
是给 voxel guard / 后续行人预测 / benchmark 评估用。
```

PhysX 真实碰撞看的是 URDF/USD collision geometry，不等于 visual mesh，也不等于 guard footprint。

## 3. 安装/替换

```bash
cd ~/resources/arena_ws/src/arena
mv arena-isaac arena-isaac_backup_before_v18_physx
unzip /path/to/arena-isaac-v18-physx-root-velocity.zip -d /tmp/arena_v18
rsync -av /tmp/arena_v18/arena-isaac/ ~/resources/arena_ws/src/arena/arena-isaac/
```

重编译：

```bash
bash --noprofile --norc
source /opt/ros/humble/setup.bash
cd ~/resources/arena_ws
. arena.bash
rm -rf build/ros2isaacsim install/ros2isaacsim log
colcon build --packages-select ros2isaacsim --symlink-install
source install/setup.bash
```

## 4. 启动

### 主链路：physx_diff_contact

终端 1：bridge（PhysX contact + RTX LiDAR + People services）

```bash
cd ~/resources/arena_ws
source /opt/ros/humble/setup.bash
. arena.bash
source install/setup.bash
cd ~/resources/arena_ws/src/arena/arena-isaac
python3 scripts/arena_scene_profile.py --profile scripts/profiles/shenxinfu_841837.yaml bridge physx_diff_contact
```

`scripts/profiles/shenxinfu_841837.yaml` 里的 `lidar.range_offset_m` 会写入自定义 RTX lidar profile 的 `rangeOffset` 参数，可用于减轻近场自遮挡。

终端 2：按 contact profile 生成机器人

```bash
cd ~/resources/arena_ws
source /opt/ros/humble/setup.bash
. arena.bash
source install/setup.bash
cd ~/resources/arena_ws/src/arena/arena-isaac
python3 scripts/arena_scene_profile.py --profile scripts/profiles/shenxinfu_841837.yaml spawn --phase physx_diff_contact
```

终端 3A：差速手柄控制（推荐）

```bash
cd ~/resources/arena_ws/src/arena/arena-isaac
python3 scripts/arena_scene_profile.py --profile scripts/profiles/shenxinfu_841837.yaml gamepad --phase physx_diff_contact
```

终端 3B：键鼠控制（可选）

```bash
cd ~/resources/arena_ws/src/arena/arena-isaac
python3 scripts/arena_scene_profile.py --profile scripts/profiles/shenxinfu_841837.yaml teleop
```

`physx_diff_contact` 是差速模式，只使用前后速度 `vx` 和角速度 `wz`；横移
`vy` 不参与执行。键鼠窗口需获得焦点，使用 `W/S` 前后、`Q/E` 转向。

终端 4：厕所行人 benchmark（可选，等待 bridge 完全启动后运行）

```bash
cd ~/resources/arena_ws
source /opt/ros/humble/setup.bash
. arena.bash
source install/setup.bash
ros2 run toilet_benchmark toilet_director_node --initial-agents 2
```

该命令复用现有 entrance -> urinal -> service -> exit -> recycle 事件流。
`physx_diff_contact` phase 已启用 `replicator_agent_core`、Character services 和
行人状态 topic，但保持 `enable_navmesh: false`，不会改变 toilet benchmark 的现有规划逻辑。

终端 5：RViz（可选）

```bash
rviz2 --ros-args -p use_sim_time:=true
```

### 手柄说明

`gamepad` 命令同时启动 ROS `joy_node` 和 `gamepad_diff_teleop`。默认左摇杆纵轴控制 `vx`、横轴控制 `wz`，按住 LB（默认 button 4）才发送命令；控制器根据 `0.345 m` 轮距执行差速公式，且始终令 `vy=0`。

操作方式：

- 启动命令前先连接手柄，并确认 Linux 中存在 `/dev/input/js0`。
- 按住 LB 后，左摇杆前后控制机器人前进/后退，左右控制原地或行进转向。
- 松开 LB 会立即发送一次停止命令；不按 deadman 时手柄不会持续占用控制权。
- 摇杆中心小幅漂移由 `gamepad.deadzone` 过滤；速度由 `linear_scale` 和 `angular_scale` 调整。
- 如果前后或转向方向相反，可以把对应 `scale` 改为负数；如果摇杆无响应，使用 `/joy` 输出确认 axis/button 编号。

键鼠和手柄命令可以分别启动，也可以同时运行。同时运行时，按住 deadman 的手柄优先；松开后键鼠恢复。如果按键或摇杆编号不匹配，先检查：

```bash
ros2 topic echo /joy
```

然后修改 `scripts/profiles/shenxinfu_841837.yaml` 中的 `gamepad.linear_axis`、`angular_axis` 和 `enable_button`。

### physx_wheels 兼容回退

需要对照旧 voxel guard 行为时，bridge 和 spawn 分别改为：

```bash
cd ~/resources/arena_ws/src/arena/arena-isaac
python3 scripts/arena_scene_profile.py --profile scripts/profiles/shenxinfu_841837.yaml bridge rtx_scan
python3 scripts/arena_scene_profile.py --profile scripts/profiles/shenxinfu_841837.yaml spawn
```

### Synthetic laser 回退链路

Synthetic laser 不再是默认主链路，仅在 RTX 不可用或需要静态地图对齐诊断时使用。终端 1 改为：

```bash
python3 scripts/arena_scene_profile.py --profile scripts/profiles/shenxinfu_841837.yaml bridge social_nav
```

spawn 完成后另开终端：

```bash
python3 scripts/arena_scene_profile.py --profile scripts/profiles/shenxinfu_841837.yaml synthetic_laser
```

也可以打印 synthetic 回退链路的专用顺序：

```bash
python3 scripts/arena_scene_profile.py --profile scripts/profiles/shenxinfu_841837.yaml scheme1 steps
```

以及做一次离线几何对齐检查：

```bash
python3 scripts/arena_scene_profile.py --profile scripts/profiles/shenxinfu_841837.yaml scheme1 check
```

## 5. Profile 关键参数

后续主链路的模式参数位于
`scripts/profiles/modes/physx_diff_contact.yaml`，核心配置为：

```yaml
runtime:
  enable_guard: false
  guard_backend: none

robot:
  model: mecanum730_xms5_lidar_physx_diff_contact
  x: -4.4
  y: -1.0
  z: 0.03

motion:
  wheel_radius: 0.08
  tire_force_enabled: true
  tire_longitudinal_stiffness: 60.0
  tire_lateral_stiffness: 20.0
```

主 profile 中对应 phase 还负责启用行人服务：

```yaml
phases:
  physx_diff_contact:
    enable_people_stack: true
    enable_character_services: true
    people_extension_mode: replicator_agent_core
    enable_navmesh: false
    lidar_backend: rtx
```

## 6. 验证

确认主链路命令组合正确：

```bash
python3 scripts/arena_scene_profile.py --profile scripts/profiles/shenxinfu_841837.yaml --dry-run bridge physx_diff_contact
python3 scripts/arena_scene_profile.py --profile scripts/profiles/shenxinfu_841837.yaml --dry-run spawn --phase physx_diff_contact
```

应看到：

```text
--robot-model mecanum730_xms5_lidar_physx_diff_contact
enable_kinematic_collision_guard:=false
enable_people_stack:=true
enable_character_services:=true
people_extension_mode:=replicator_agent_core
```

bridge 日志应出现：

```text
mecanum controller initialized: mode=physx_diff_contact
```

确认行人接口：

```bash
ros2 service list | grep -E 'spawn_pedestrian|move_pedestrians'
ros2 topic list | grep pedestrian
```

方案①静态地图检查：

```bash
python3 scripts/arena_scene_profile.py --profile scripts/profiles/shenxinfu_841837.yaml scheme1 check
```

当前 `shenxinfu_841837` 已生成：

```text
/home/stardust/resources/arena_ws/arena_assets/collision_configs/shenxinfu_841837.map.yaml
/home/stardust/resources/arena_ws/arena_assets/collision_configs/shenxinfu_841837.map.pgm
```

`scheme1 check` 通过时应看到：

```text
resolution_match: true
missing_world_cells: 0
extra_world_cells: 0
scan_alignment_ok: true
aligned: true
```

检查 TF：

```bash
ros2 topic echo /tf_static --once
ros2 run tf2_ros tf2_echo odom base_link
```

`/tf_static` 中 LiDAR stamp 应为 0。

检查速度输出：

```bash
ros2 topic echo /cmd_vel_applied
```

`physx_diff_contact` 不通过 voxel guard 裁剪墙前速度；碰撞阻挡应由 PhysX
接触产生，并反映在实际 `/odom` 与轮地诊断中。

检查 contact backend 是否真正生效：

```text
mecanum controller initialized: mode=physx_diff_contact
```

## 7. 本分支修改点

```text
ros2isaacsim/isaac_utils/mecanum_teleop.py
  - 新增 mode=physx_wheels
  - 新增 velocity smoother / acceleration limiter
  - PhysX wheels 模式回读实际 stage pose 发布 odom/tf
  - 新增 height/upright velocity-level stabilizer
  - 保留 /clock 同步 odom/tf
  - 保留 static sensor TF stamp=0

ros2isaacsim/isaac_utils/services/UrdfToUsd.py
  - 支持 robot_model 中的 physx_wheels 模式
  - physx_wheels 模式优先引用离线 USD 资产
  - physx 模式不调用 _disable_physics_tree

ros2isaacsim/ros2isaacsim/spawn_v10_scene_lidar_validation.py
  - 新增 --robot-model 参数
  - 不再硬编码 mecanum730_xms5_lidar_kinematic

scripts/arena_scene_profile.py
  - spawn 时从 profile 传入 robot.model
  - bridge 时注入 motion / lidar publish_points 等环境变量

scripts/profiles/shenxinfu_841837.yaml
  - 默认 robot.model 改为 mecanum730_xms5_lidar_physx_wheels
  - 默认双 lidar 关闭 point cloud / 可视化
```

## 8. 下一步

`physx_diff_contact` 已接入 RTX LiDAR 和 toilet benchmark 行人链路。后续功能、
碰撞修复和社会导航集成均以该模式为主；`physx_wheels` 只作为回归基线保留。

当前重点：

```text
1. 持续校准轮胎力、地面摩擦和实际差速响应。
2. 验证机器人、门、隔板及卫浴设施的 PhysX 碰撞完整性。
3. 在同一主链路上验证 RTX scan、实际 odom/TF 和 toilet benchmark。
4. 后续再评估 HuNav 或其他社会运动层，不改写 toilet director 的事件职责。
```
