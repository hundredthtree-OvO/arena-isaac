# arena-isaac v17 cleanup：shenxinfu_841837 / voxel guard / actual odom-TF

本版本用于当前 Isaac Sim 4.5.0 + ROS 2 Humble 的 `shenxinfu_841837` 场景验证。默认路线是：

```text
Isaac controller 发布真实 /odom 与 TF
voxel guard 负责静态场景碰撞过滤
/cmd_vel 是期望速度
/cmd_vel_applied 是 guard 裁剪后的实际执行速度
front/rear LiDAR 使用手动标定位置 ±0.30m
```

已移除旧的外部 odom 积分节点，避免和 Isaac controller 重复发布 `odom -> base_link`。

---

## 1. 代码安装与重编译

把本目录放到：

```bash
~/resources/arena_ws/src/arena/arena-isaac
```

然后重编译：

```bash
bash --noprofile --norc
source /opt/ros/humble/setup.bash
cd ~/resources/arena_ws
. arena.bash
rm -rf build/ros2isaacsim install/ros2isaacsim log
colcon build --packages-select ros2isaacsim --symlink-install
source install/setup.bash
```

确认包路径：

```bash
ros2 pkg prefix ros2isaacsim
```

期望输出类似：

```text
/home/stardust/resources/arena_ws/install/ros2isaacsim
```

---

## 2. 当前关键配置

配置文件：

```bash
~/resources/arena_ws/src/arena/arena-isaac/scripts/profiles/shenxinfu_841837.yaml
```

当前推荐参数：

```yaml
robot:
  z: 0.0
  cmd_vel_topic: /cmd_vel
  odom_topic: /odom
  odom_frame: odom
  base_frame: base_link
  cmd_vel_applied_topic: /cmd_vel_applied

robot_geometry:
  auto_footprint: false

lidar:
  mount_source: manual
  front_x: 0.30
  rear_x: -0.30
  y: 0.0
  z: 0.20
  front_frame: front_laser_link
  rear_frame: rear_laser_link
  visualize: true

odom_tf:
  publish_actual: true
  publish_sensor_static_tf: true

guard:
  backend: voxel
  length: 0.70
  width: 0.42
  margin: 0.02
  overlap_policy: escape
```

LiDAR 世界高度约等于 `robot.z + lidar.z`。当前是 `0.0 + 0.20 = 0.20m`。

---

## 3. 构建或重建 voxel map

只有在场景文件、voxel 参数、skip/include 规则变了之后才需要重建 voxel map。

### 终端 1：启动轻量 bridge

```bash
pkill -f rviz2 || true
pkill -f "isaac-sim-4.5.0/kit/python/bin/python3" || true
pkill -f "run_isaacsim" || true

source /opt/ros/humble/setup.bash
cd ~/resources/arena_ws
. arena.bash
source install/setup.bash
cd ~/resources/arena_ws/src/arena/arena-isaac

python3 scripts/arena_scene_profile.py --profile scripts/profiles/shenxinfu_841837.yaml bridge voxel_build
```

### 终端 2：导入场景和机器人

```bash
source /opt/ros/humble/setup.bash
cd ~/resources/arena_ws
. arena.bash
source install/setup.bash
cd ~/resources/arena_ws/src/arena/arena-isaac

python3 scripts/arena_scene_profile.py --profile scripts/profiles/shenxinfu_841837.yaml spawn
```

### 终端 2：生成 map / svg / pcd

```bash
python3 scripts/arena_scene_profile.py --profile scripts/profiles/shenxinfu_841837.yaml voxel build
python3 scripts/arena_scene_profile.py --profile scripts/profiles/shenxinfu_841837.yaml voxel render
python3 scripts/arena_scene_profile.py --profile scripts/profiles/shenxinfu_841837.yaml voxel summary
```

输出文件：

```text
/home/stardust/resources/arena_ws/arena_assets/collision_configs/shenxinfu_841837.voxel.json.gz
/home/stardust/resources/arena_ws/arena_assets/collision_configs/shenxinfu_841837.voxel.svg
/home/stardust/resources/arena_ws/arena_assets/collision_configs/shenxinfu_841837.voxel.pcd
```

---

## 4. 正常运行 voxel guard

### 终端 1：启动 guard bridge

```bash
pkill -f rviz2 || true
pkill -f "isaac-sim-4.5.0/kit/python/bin/python3" || true
pkill -f "run_isaacsim" || true

source /opt/ros/humble/setup.bash
cd ~/resources/arena_ws
. arena.bash
source install/setup.bash
cd ~/resources/arena_ws/src/arena/arena-isaac

python3 scripts/arena_scene_profile.py --profile scripts/profiles/shenxinfu_841837.yaml bridge voxel_guard
```

### 终端 2：spawn 场景、机器人、LiDAR

```bash
source /opt/ros/humble/setup.bash
cd ~/resources/arena_ws
. arena.bash
source install/setup.bash
cd ~/resources/arena_ws/src/arena/arena-isaac

python3 scripts/arena_scene_profile.py --profile scripts/profiles/shenxinfu_841837.yaml spawn
```

bridge 日志应出现类似：

```text
[xms_mecanum] collision_guard initialized: backend=voxel, length=..., width=..., margin=...
[xms_mecanum] voxel_guard loaded: path=...
[xms_mecanum] mecanum controller initialized: ... collision_guard=enabled
[xms_mecanum] published static sensor TFs ...
```

如果没有 `collision_guard=enabled` 或 `voxel_guard loaded`，说明不是按 `bridge voxel_guard` 启动，或者旧 Isaac 进程没有清干净。

### 终端 3：键盘控制

```bash
source /opt/ros/humble/setup.bash
cd ~/resources/arena_ws
. arena.bash
source install/setup.bash
cd ~/resources/arena_ws/src/arena/arena-isaac

python3 scripts/arena_scene_profile.py --profile scripts/profiles/shenxinfu_841837.yaml teleop
```

---

## 5. RViz

RViz 必须使用仿真时间：

```bash
source /opt/ros/humble/setup.bash
cd ~/resources/arena_ws
. arena.bash
source install/setup.bash
rviz2 --ros-args -p use_sim_time:=true
```

推荐设置：

```text
Fixed Frame: odom
Add: TF
Add: Odometry /odom
Add: LaserScan /front_scan
Add: LaserScan /rear_scan
LaserScan Decay Time: 0
```

---

## 6. 验证 checklist

### TF 发布者

```bash
ros2 topic info /tf -v
```

期望 `/tf` 只有一个动态 TF 发布者：

```text
Node name: isaac_controller
```

不应再出现外部 odom 积分节点。

### TF 链

```bash
ros2 run tf2_ros tf2_echo odom base_link
ros2 run tf2_ros tf2_echo base_link front_laser_link
ros2 run tf2_ros tf2_echo base_link rear_laser_link
```

期望 LiDAR 静态 TF 约为：

```text
front_laser_link: x = +0.300, z = 0.200
rear_laser_link : x = -0.300, z = 0.200, yaw = 180deg
```

### /tf_static 时间戳

```bash
ros2 topic echo /tf_static --once
```

期望 sensor static TF stamp 为 0：

```yaml
stamp:
  sec: 0
  nanosec: 0
```

### scan 与 /clock

```bash
ros2 topic echo /clock --once
ros2 topic echo /rear_scan --once | head -20
```

`/rear_scan.header.stamp` 应接近 `/clock.clock`。

### collision guard

撞墙时看：

```bash
ros2 topic echo /cmd_vel_applied
```

期望：

```text
/cmd_vel          非零
/cmd_vel_applied  接近 0 或被裁剪
/odom             不继续漂移
```

---

## 7. 常用调参

### LiDAR 位置

```yaml
lidar:
  mount_source: manual
  front_x: 0.30
  rear_x: -0.30
```

如果 LiDAR 太靠内、被车体遮挡，可以试：

```yaml
front_x: 0.32
rear_x: -0.32
```

如果 LiDAR 仍可能伸出 guard，则往内收：

```yaml
front_x: 0.28
rear_x: -0.28
```

### guard footprint

```yaml
guard:
  length: 0.70
  width: 0.42
  margin: 0.02
```

如果通道太窄进不去，优先微调 `width` 或 `margin`。如果 LiDAR 穿墙，不要优先拉大 body guard，而是先调 LiDAR `front_x/rear_x`。

---

## 8. 当前代码整理点

- 删除外部 odom 积分节点入口与源码，避免重复发布 `odom -> base_link`。
- `arena_scene_profile.py` 删除 `odom` 子命令。
- `mecanum_teleop.py` 由 Isaac USD pose 发布真实 `/odom` 与动态 `odom -> base_link`。
- 动态 TF 与 `/odom` stamp 使用 `/clock` topic，同步 Isaac sim time。
- 静态 LiDAR TF 发布到 `/tf_static`，stamp 固定为 0。
- LiDAR TF 从 USD prim 实际 pose 读取，和 Isaac 中的 LiDAR marker/prim 保持一致。
- `/cmd_vel_applied` 保留，用于记录 guard 过滤后的真实执行速度。

---

## 9. 数据采集建议 topic

基础版：

```bash
ros2 bag record \
  /clock \
  /cmd_vel \
  /cmd_vel_applied \
  /odom \
  /tf \
  /tf_static \
  /front_scan \
  /rear_scan
```

后续加入行人后再记录行人 ground truth / tracks / collision events。
