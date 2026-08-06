# arena-isaac: physx_diff_contact 主链路

`shenxinfu_841837` 现在以 `physx_diff_contact` 作为默认主链路，`rtx_scan` 也切到同一模式。
profile surface 保留场景 `proxy` 修复、RTX LiDAR、手动/作者碰撞修复与 synthetic fallback，
不再公开旧的兼容入口。

## 运行

终端 1: bridge

```bash
cd ~/resources/arena_ws
source /opt/ros/humble/setup.bash
. arena.bash
source install/setup.bash
cd ~/resources/arena_ws/src/arena/arena-isaac
python3 scripts/arena_scene_profile.py --profile scripts/profiles/shenxinfu_841837.yaml bridge physx_diff_contact
```

终端 2: spawn

```bash
cd ~/resources/arena_ws/src/arena/arena-isaac
python3 scripts/arena_scene_profile.py --profile scripts/profiles/shenxinfu_841837.yaml spawn --phase physx_diff_contact
```

终端 3: gamepad

```bash
cd ~/resources/arena_ws/src/arena/arena-isaac
python3 scripts/arena_scene_profile.py --profile scripts/profiles/shenxinfu_841837.yaml gamepad --phase physx_diff_contact
```

终端 4: 键鼠

```bash
cd ~/resources/arena_ws/src/arena/arena-isaac
python3 scripts/arena_scene_profile.py --profile scripts/profiles/shenxinfu_841837.yaml teleop
```

## 关键配置

```yaml
default_mode: physx_diff_contact
mode_files:
  physx_diff_contact: modes/physx_diff_contact.yaml

robot_footprint:
  length: 0.7
  width: 0.42
  margin: 0.02
```

`robot_footprint` 会导出到：

```text
ARENA_ISAAC_ROBOT_FOOTPRINT_LENGTH
ARENA_ISAAC_ROBOT_FOOTPRINT_WIDTH
ARENA_ISAAC_ROBOT_FOOTPRINT_MARGIN
```

如果后续 runtime 需要方向覆盖，可额外使用：

```text
ARENA_ISAAC_ROBOT_FOOTPRINT_FORWARD
ARENA_ISAAC_ROBOT_FOOTPRINT_REAR
ARENA_ISAAC_ROBOT_FOOTPRINT_LEFT
ARENA_ISAAC_ROBOT_FOOTPRINT_RIGHT
```

这组 footprint 与 kinematic collision guard 的参数解耦，避免动态行人接触半径被缩回 mecanum 默认尺寸。

## 保留行为

- `proxy` 场景 collision repair 继续启用，用于恢复白名单内的门体和门框碰撞。
- `lidar.backend=rtx` 仍是主扫描源，`range_offset_m` 继续用于压制近场自遮挡。
- `social_nav` 保留为 synthetic fallback 和 map/TF 对齐检查入口。
- `scheme1 steps` 和 `scheme1 check` 仍可用于离线几何对齐诊断。

## 验证

```bash
python3 scripts/arena_scene_profile.py --profile scripts/profiles/shenxinfu_841837.yaml --dry-run bridge physx_diff_contact
python3 scripts/arena_scene_profile.py --profile scripts/profiles/shenxinfu_841837.yaml --dry-run spawn --phase physx_diff_contact
python3 scripts/arena_scene_profile.py --profile scripts/profiles/shenxinfu_841837.yaml --dry-run bridge rtx_scan
```

期望看到：

```text
--robot-model mecanum730_xms5_lidar_physx_diff_contact
enable_people_stack:=true
enable_character_services:=true
```
