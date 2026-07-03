# Isaac Sim `scan + map_server` Route Plan

## Goal

Move `arena-isaac` away from the current `voxel/grid -> synthetic LaserScan` main path and toward the same high-level split used by many ROS navigation stacks:

- `scan` comes from a real simulator sensor pipeline
- `map` comes from a static occupancy-map generation pipeline
- dynamic actors are not baked into the static map

For this repository, the target state is:

```text
Isaac Sim lidar sensor
  -> /front_scan, /rear_scan (or /scan)

Static occupancy map generator
  -> map.yaml + map.pgm
  -> nav2 map_server
  -> /map OccupancyGrid

Dynamic pedestrians
  -> visible in scan at runtime
  -> excluded from static map
```

## Recommendation

Primary recommendation:

- use real sensor `LaserScan`
- use a static occupancy-map pipeline for `map_server`
- make both outputs derive from the same scene geometry semantics

Preferred order:

1. `RTX lidar + occupancy map from visible render meshes`
2. if RTX self-occlusion is too unstable, fall back to `PhysX lidar + occupancy map from collision approximations`

Do not use the current `voxel -> map -> synthetic scan` path as the long-term main route.

## Why This Route

The current synthetic scan path is stable, but its scan quality is bounded by grid discretization:

```text
voxel columns
  -> occupancy cells
  -> DDA raycast
  -> stair-step scan boundaries
```

That makes it hard to get the "continuous scene contour" effect expected from a real lidar.

By contrast, Arena-Rosnav-style systems separate:

- `LaserScan` for observations
- `map_server` / `OccupancyGrid` for navigation state

That split is the right long-term model here as well.

## Key Isaac Sim Insight

Isaac Sim's official occupancy-map tooling already distinguishes geometry source depending on lidar type:

- for PhysX lidar, use collision geometry
- for RTX lidar, use original USD meshes

This implies the core engineering rule for this project:

> `scan` and `map` must come from the same geometry meaning.

Bad combinations:

- RTX scan + voxel-derived map
- RTX scan + collision-only occupancy map
- PhysX scan + render-mesh occupancy map

Good combinations:

- RTX scan + render-mesh occupancy map
- PhysX scan + collision-geometry occupancy map

## Proposed Architecture

### Option A: Preferred

```text
RTX lidar
  -> ROS2RtxLidarHelper
  -> /front_scan, /rear_scan

Occupancy map generator
  -> visible structural meshes only
  -> map.yaml + map.pgm
  -> map_server
  -> /map
```

Use this when:

- self-occlusion can be controlled
- robot self-mesh can be hidden from the lidar render path
- scene structural meshes are clean enough to generate a useful static map

### Option B: Conservative fallback

```text
PhysX lidar
  -> LaserScan

Occupancy map generator
  -> collision approximations
  -> map.yaml + map.pgm
  -> map_server
```

Use this when:

- RTX self-occlusion is not reliably fixable with acceptable effort
- collision geometry is already much cleaner than render geometry

## What Must Not Enter The Static Map

The static map should contain only navigation-relevant static geometry:

- walls
- permanent partitions
- fixed door frames
- large fixed cabinets or fixtures only if they genuinely constrain motion

The static map should exclude:

- the robot itself
- pedestrians
- moveable objects
- door leaves that may open/close
- decorative assets
- overly detailed bathroom fixtures if they are not desired as navigation barriers

## Implementation Phases

## Phase 0: Freeze Baseline

Objective:

- preserve the current working voxel/synthetic path as a fallback

Actions:

- keep `synthetic_2d_laser` intact as a backup path
- keep current generated `shenxinfu_841837.map.yaml` only for debug/reference
- document current known-good startup sequence

Exit criteria:

- team can always fall back to the current synthetic scan path

## Phase 1: Build True Static Map Pipeline

Objective:

- generate the map for `map_server` from the proper static scene geometry, not from voxel columns

Actions:

1. choose geometry source:
   - Option A: render meshes
   - Option B: collision approximations
2. use Isaac Sim occupancy map tooling to export:
   - `map.yaml`
   - `map.pgm`
3. exclude robot and dynamic actors from the exported map
4. verify:
   - map origin
   - map resolution
   - wall continuity
   - no robot silhouette in map

Exit criteria:

- a clean static map exists without voxel-like stair-step artifacts dominating major walls

## Phase 2: Fix Sensor Self-Occlusion

Objective:

- make real lidar scan usable as the main observation source

Actions for RTX route:

1. confirm which prims are visible to lidar render product
2. inspect whether the robot body mesh or imported collision mesh is causing self-returns
3. test asset-layer fixes:
   - hide robot self-mesh from lidar render path
   - preserve PhysX collision while suppressing render visibility for lidar
   - move sensor mounting slightly outward only if necessary
4. verify:
   - robot does not draw a large self silhouette
   - nearby walls are still visible
   - scan remains stable while rotating in place

Actions for PhysX fallback:

1. simplify collision geometry if necessary
2. mount lidar on clean frame outside the main body volume
3. verify scan coverage and continuity

Exit criteria:

- real lidar scan can be used for navigation without dominant self-occlusion

## Phase 3: Wire Map Server

Objective:

- publish a standard static map through `map_server`

Actions:

1. add a launch path for `map_server` using the generated `map.yaml`
2. expose `/map`
3. ensure `frame_id` and origin conventions are stable
4. keep robot scan frame consistent with localization stack expectations

Exit criteria:

- `/map` is live and usable by AMCL/Nav2 or equivalent consumers

## Phase 4: Align Scan and Map

Objective:

- prove the real scan geometry is consistent with the static map

Checks:

1. place robot at known poses
2. compare scan to map walls in RViz
3. rotate robot in place
4. drive parallel to walls
5. confirm no systematic:
   - yaw bias
   - origin offset
   - mirrored scan
   - front/rear scan inversion

Exit criteria:

- walls in scan and map overlap within expected discretization tolerance

## Phase 5: Dynamic Actor Handling

Objective:

- keep dynamic humans visible in scan without polluting the static map

Actions:

- let real lidar see runtime actors naturally
- keep pedestrians out of the `map.yaml`
- if needed, add dynamic obstacle layer at navigation stack level

Exit criteria:

- humans show up in scan at runtime
- static map remains clean and reusable

## Experimental Order

Recommended execution order:

1. export a render-mesh occupancy map with robot excluded
2. visually inspect `map.yaml + pgm`
3. enable real lidar scan only
4. test self-occlusion in an empty/static scene
5. if self-occlusion is acceptable, proceed with RTX route
6. if not, test collision-based occupancy map + PhysX lidar fallback
7. only after scan quality is acceptable, integrate `map_server`

This order avoids wasting effort on localization/map-server plumbing before the lidar itself is trustworthy.

## Validation Checklist

### Static map validation

- map does not contain robot body
- map excludes pedestrians
- outer walls are continuous
- door openings match intended navigation behavior
- no strong voxel/block artifacts on major boundaries

### Scan validation

- no dominant self-hit near zero range
- front and rear scan orientation correct
- wall ranges stable over multiple frames
- rotating in place does not create scan discontinuities unrelated to environment

### Alignment validation

- `/scan` overlaps `/map` in RViz
- scan does not appear mirrored
- scan does not appear globally rotated
- scan origin matches expected base frame

## Risks

### Main risks

- RTX lidar still sees robot self-mesh even after visibility cleanup
- scene render meshes are too detailed/noisy for a good static occupancy map
- collision approximations are too coarse if RTX path is abandoned
- map generated from one geometry subset does not match scan generated from another

### Practical mitigation

- keep the current synthetic scan path as fallback
- evaluate RTX and PhysX routes as parallel options early
- prefer structural geometry over visual clutter in map generation

## Deliverables

Minimum viable deliverables for this route:

1. `map.yaml + map.pgm` generated from scene geometry, not voxel columns
2. real lidar `LaserScan` pipeline selected as primary
3. documented startup sequence for:
   - bridge
   - spawn
   - lidar
   - map_server
   - optional pedestrians
4. RViz evidence that scan and map align

## Recommended First Engineering Task

Before touching AMCL or Nav2 integration, do this first:

> produce two static maps for the same scene:
> 1. render-mesh occupancy map
> 2. collision-geometry occupancy map
>
> then compare each against a real lidar scan from the same robot pose.

That experiment will decide the route quickly:

- if render-map and RTX scan agree, choose Option A
- if collision-map and PhysX scan agree better, choose Option B

## Final Recommendation

Proceed with this roadmap:

1. treat the current synthetic scan path as fallback only
2. prototype `RTX scan + render-mesh occupancy map`
3. if self-occlusion blocks progress, switch to `PhysX scan + collision occupancy map`
4. only after scan quality is validated, standardize `map_server` integration

This is the most credible route to achieve:

- continuous-looking lidar contours
- clean static maps
- compatibility with ROS navigation conventions
- conceptual alignment with Arena-Rosnav
