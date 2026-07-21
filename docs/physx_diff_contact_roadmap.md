# PhysX Differential Contact Roadmap

## Goal

Add a differential-drive mode in which Isaac Sim moves the robot exclusively
through wheel joint drives and PhysX contacts. The mode must not write the
robot root pose or root velocity, and it must remain isolated from the existing
`physx_wheels` fallback.

## Architecture

- Keep ROS subscriptions, command arbitration, arm hold, TF, and lifecycle in
  `MecanumRobot`.
- Keep Twist-to-wheel conversion in `drive_kinematics.py`.
- Add a dedicated `physx_diff_contact` backend that only configures wheel
  drives, applies wheel velocity targets, and reads measured articulation state.
- Select the backend explicitly from the scene profile instead of relying only
  on a robot-model suffix.
- Keep the existing `physx_wheels` mode available as a rollback baseline.

## Asset Strategy

Generate an Isaac-specific derivative from the current complete Isaac URDF.
Do not maintain a hand-copied Genesis URDF.

The derivative must:

- preserve the manipulator, gripper, lidar links, and joint names;
- rewrite external mesh paths to repository-owned meshes;
- replace the legacy `base_link` collision-sphere cloud with simple chassis
  box colliders;
- add one cylinder collider to each wheel core;
- leave mecanum rollers visual-only during the differential-drive experiment;
- use wheel-joint damping suitable for velocity drive rather than the legacy
  locked-wheel value;
- leave the source URDF unchanged and be reproducible from a deterministic
  builder.

## Phases

## Current Status

- Phase 1 builder, repository-local URDF, deterministic regeneration, and XML
  regression tests are complete.
- The Phase 2 wheel-target-only backend and experimental profile wiring are
  complete.
- The contact URDF bakes the compact navigation arm and gripper posture into
  fixed joints while preserving their collision geometry.
- Runtime USD import inspection and contact tuning remain pending because they
  require an active Isaac Sim 4.5 bridge.
- The default `rtx_scan` phase remains on `physx_wheels`; use the explicit
  `physx_diff_contact` phase for experiments.

### Phase 1: Reproducible URDF Variant

- Add the derivative builder and generated URDF.
- Add XML tests for collision geometry, local mesh paths, wheel joints, and
  preservation of the arm and sensor structure.
- Import the derivative into USD and inspect articulation and collision schemas.

Acceptance: all four wheels have primitive collision, no external mesh path or
legacy base collision sphere remains, and regeneration produces no diff.

### Phase 2: Contact Drive Backend

- Add `physx_diff_contact` as an isolated backend.
- Accept only `vx` and `wz`; force `vy` to zero.
- Apply velocity targets to FL/FR/RL/RR wheel joints.
- Prohibit root pose writes, root velocity writes, kinematic fallback, and
  voxel-guard filtering in this backend.
- Add unit tests for backend selection and command mapping.

Acceptance: the base remains stationary with zero wheel target and moves only
when PhysX resolves wheel-ground contact.

### Phase 3: Contact and Stability Tuning

- Configure wheel, floor, and wall physics materials.
- Tune wheel drive damping and maximum force, chassis mass/inertia, acceleration
  limits, solver iterations, contact offsets, and simulation substeps.
- Test straight motion, rotation, braking, wall contact, oblique contact, and
  door-threshold traversal.

Acceptance: no root correction is active, no persistent penetration occurs,
and bounded commands do not overturn the robot in the benchmark scene.

### Phase 4: Physical Odometry and Lidar

- Continue publishing pose from the actual articulation root.
- Publish twist from measured PhysX root velocity or pose differentiation rather
  than the command velocity.
- Validate RTX scan alignment during acceleration, slip, blocking, and impact.

Acceptance: odometry reports zero or reduced motion when wheels slip or the
robot is blocked, while lidar remains aligned in RViz.

### Phase 5: Benchmark Integration

- Add a dedicated profile phase with robot voxel guard disabled but scene
  CollisionAPI repair retained.
- Validate gamepad and Nav2 differential commands.
- Verify arm holding does not inject unstable forces into the articulation.
- Verify robot contact with static geometry and pedestrian collision proxies.

Acceptance: the existing benchmark can switch between legacy and contact modes
without modifying source code or regenerating the scene.

## Risks

- The imported articulation may still contain authored root or joint settings
  that override physical motion.
- Existing arm hard-sync can inject energy into the full articulation.
- Simplified cylinders intentionally approximate mecanum wheels as differential
  tires and therefore do not support physical lateral motion.
- Disabling voxel guard is safe only after every required scene collider has
  been verified.
- Genesis force, friction, and solver values are starting points, not directly
  transferable Isaac Sim tuning values.
