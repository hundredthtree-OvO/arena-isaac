# arena-isaac：Stage-1 PhysX wheels navigation path

当前推荐目标：在 v17 的稳定导航链路上，增加 **PhysX wheels** 底盘执行模式。它不追求真实麦轮 roller 接触，而是通过 PhysX articulation + 4 个主轮关节速度目标驱动整机，并保留场景碰撞响应与 ROS 接口兼容。

## 1. 当前设计

控制链：

```text
/cmd_vel
  -> velocity smoother / acceleration limiter
  -> voxel guard 上层安全过滤
  -> wheel joint velocity targets
  -> PhysX 积分与 CollisionAPI 接触响应
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
voxel guard
manual LiDAR mount: front_x=0.30, rear_x=-0.30, z=0.20
```

关键变化：

```text
v17 kinematic：直接写 USD root pose。
stage1 physx_wheels：写 4 个主轮关节速度，期望由 PhysX articulation 积分和处理接触。
```

## 2. 重要说明

`voxel guard` 仍然保留，但定位变为上层安全过滤：

```text
PhysX CollisionAPI：底层接触、阻挡、防穿模。
voxel guard：提前阻止危险速度、提供可解释 blocked 日志、后续融合 dynamic actor guard。
```

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

终端 1：bridge（方案①推荐：`social_nav`）

```bash
cd ~/resources/arena_ws
source /opt/ros/humble/setup.bash
. arena.bash
source install/setup.bash
cd ~/resources/arena_ws/src/arena/arena-isaac
python3 scripts/arena_scene_profile.py --profile scripts/profiles/shenxinfu_841837.yaml bridge social_nav
```

如果要单独恢复真实传感器 scan（RTX）做验证，改用：

```bash
python3 scripts/arena_scene_profile.py --profile scripts/profiles/shenxinfu_841837.yaml bridge rtx_scan
```

`scripts/profiles/shenxinfu_841837.yaml` 里的 `lidar.range_offset_m` 会写入自定义 RTX lidar profile 的 `rangeOffset` 参数，可用于减轻近场自遮挡。

终端 2：spawn

```bash
cd ~/resources/arena_ws
source /opt/ros/humble/setup.bash
. arena.bash
source install/setup.bash
cd ~/resources/arena_ws/src/arena/arena-isaac
python3 scripts/arena_scene_profile.py --profile scripts/profiles/shenxinfu_841837.yaml spawn
```

终端 3：synthetic laser（方案①静态 map + 动态行人叠加）

```bash
cd ~/resources/arena_ws
source /opt/ros/humble/setup.bash
. arena.bash
source install/setup.bash
cd ~/resources/arena_ws/src/arena/arena-isaac
python3 scripts/arena_scene_profile.py --profile scripts/profiles/shenxinfu_841837.yaml synthetic_laser
```

终端 4：pedestrians（可选）

```bash
cd ~/resources/arena_ws
source /opt/ros/humble/setup.bash
. arena.bash
source install/setup.bash
cd ~/resources/arena_ws/src/arena/arena-isaac
python3 scripts/arena_scene_profile.py --profile scripts/profiles/shenxinfu_841837.yaml pedestrians
```

终端 5：teleop（可选）

```bash
cd ~/resources/arena_ws/src/arena/arena-isaac
python3 scripts/arena_scene_profile.py --profile scripts/profiles/shenxinfu_841837.yaml teleop
```

RViz：

```bash
rviz2 --ros-args -p use_sim_time:=true
```

也可以先打印方案①专用顺序：

```bash
python3 scripts/arena_scene_profile.py --profile scripts/profiles/shenxinfu_841837.yaml scheme1 steps
```

以及做一次离线几何对齐检查：

```bash
python3 scripts/arena_scene_profile.py --profile scripts/profiles/shenxinfu_841837.yaml scheme1 check
```

## 5. Profile 关键参数

当前默认：

```yaml
robot:
  model: mecanum730_xms5_lidar_physx_wheels

motion:
  smoothing_enabled: true
  max_linear_accel: 0.4
  max_lateral_accel: 0.4
  max_angular_accel: 0.8
  max_linear_decel: 0.8
  max_lateral_decel: 0.8
  max_angular_decel: 1.2

physx_root_velocity:
  height_kp: 8.0
  height_max_vel: 0.35
  upright_kp: 10.0
  upright_max_ang_vel: 1.5
  fallback_kinematic: true
```

回退到 v17 直接 kinematic：

```yaml
robot:
  model: mecanum730_xms5_lidar_kinematic
```

## 6. 验证

确认 spawn 使用 PhysX 模式：

```bash
python3 scripts/arena_scene_profile.py --profile scripts/profiles/shenxinfu_841837.yaml --dry-run spawn
```

应看到：

```text
--robot-model mecanum730_xms5_lidar_physx_wheels
```

bridge 日志应出现：

```text
mecanum controller initialized: mode=physx_wheels
collision_guard initialized: backend=voxel
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

如果撞墙或 guard 阻挡，`/cmd_vel_applied` 应被裁剪或置零。

检查 PhysX wheels 是否真正生效：

```text
mecanum controller initialized: mode=physx_wheels
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

先不要合并动态行人。建议先单独验证：

```text
1. PhysX wheels 是否真生效，且不再依赖 root velocity fallback。
2. 机器人是否稳定，不跳、不倒、不穿地。
3. 撞墙时 PhysX 是否阻挡，/odom 是否不再继续穿墙漂移。
4. voxel guard 是否仍能提前 blocked。
5. acceleration limiter 是否让速度缓慢变化。
```

通过后，再进入：

```text
v19 animated pedestrians + semantic capsule + dynamic actor guard
```
