# PhysX Differential Contact Roadmap

## Goal

Use a differential-drive backend that moves the robot through wheel drives and
PhysX contacts, not through root pose writes. The scene profile should keep the
proxy collision repair and RTX LiDAR surfaces, while the robot footprint stays
separate from any collision-guard parameters.

## Current Status

- `physx_diff_contact` is the default scene mode.
- `rtx_scan` also resolves to `physx_diff_contact`.
- The scene profile now exports `ARENA_ISAAC_ROBOT_FOOTPRINT_*` for the robot
  contact footprint.
- 旧兼容入口、2D collision surface 和 robot voxel-guard profile/CLI surface
  已从该场景工作流中移除。
- Runtime migration is still pending for any code that should consume the new
  footprint env vars directly.

## What Still Matters

- Keep `proxy` scene collision repair enabled so authored door and frame fixes
  remain in place.
- Keep RTX LiDAR as the main scan source for the contact profile.
- Keep `social_nav` as the synthetic/manual fallback path for map and TF checks.
- Keep the arm hold, gamepad, and benchmark launch flow stable while contact
  tuning continues.

## Remaining Work

- Migrate runtime consumers from collision-guard footprint values to
  `ARENA_ISAAC_ROBOT_FOOTPRINT_*`.
- Tune wheel drive damping, tire forces, and contact offsets against the real
  Isaac bridge.
- Validate blocking, wall contact, and door-threshold traversal in the scene.
- Confirm odometry and scan alignment under slip and impact.

## Acceptance

- The robot footprint used for pedestrian contact remains at the explicit
  profile values, not the mecanum default geometry.
- `physx_diff_contact` and `rtx_scan` produce the same contact profile surface.
- 主工作流不再需要旧兼容命令或 2D collision 命令。
