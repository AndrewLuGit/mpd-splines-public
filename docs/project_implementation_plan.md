# Real-World MPD Deployment Plan

This document maps the project described in `CS593 Midterm Report.docx` onto the current repository.

## Goal

Build an online planning stack for a Franka Panda arm:

1. Capture depth images from simulated or real cameras.
2. Reconstruct a 3D scene with nvblox.
3. Convert the reconstructed scene into obstacle geometry usable by MPD.
4. Plan a collision-free Panda joint trajectory with the pretrained warehouse MPD model.
5. Execute the trajectory in simulation first, then on the real robot.

## What Already Exists In This Repo

The current codebase already provides most of the offline planning stack:

- Panda robot model and kinematics:
  - `mpd/torch_robotics/torch_robotics/robots/robot_panda.py`
- Warehouse environment:
  - `mpd/torch_robotics/torch_robotics/environments/env_warehouse.py`
- Obstacle primitives and signed-distance support:
  - `mpd/torch_robotics/torch_robotics/environments/primitives.py`
  - `mpd/torch_robotics/torch_robotics/environments/env_base.py`
  - `mpd/torch_robotics/torch_robotics/environments/grid_map_sdf.py`
- MPD model loading and inference:
  - `mpd/inference/inference.py`
  - `scripts/inference/inference.py`
- Trajectory execution in Isaac Gym:
  - `mpd/torch_robotics/torch_robotics/isaac_gym_envs/motion_planning_envs.py`

Important existing capability:

- The environment already supports `obj_extra_list`, which is exactly the hook needed for online obstacles.
- Collision costs can target only extra objects or all objects.
- `GenerativeOptimizationPlanner.plan_trajectory(...)` already plans from arbitrary start and goal joint states.

## Biggest Gap

The repo does **not** yet contain the online perception bridge:

- No SAPIEN scene/camera integration
- No nvblox ingestion or TSDF reader
- No conversion from TSDF/voxels to MPD obstacle primitives
- No real Franka trajectory follower
- No online inverse kinematics stage for turning a desired end-effector target into a feasible `q_goal`

## Key Constraint From The Current Planner

The warehouse model was trained with end-effector goal pose context, but the inference pipeline still expects:

- `q_pos_start`
- `q_pos_goal`
- `ee_pose_goal`

So a real deployment needs an **IK or goal-sampling step** before calling MPD. The existing IK example is:

- `mpd/torch_robotics/examples/inverse_kinematics.py`

## Recommended Architecture

Use the following runtime pipeline:

1. `SAPIEN / RealSense adapter`
   - Produces synchronized depth images and camera poses in the robot/world frame.
2. `nvblox wrapper`
   - Builds a TSDF or occupancy volume from incoming depth frames.
3. `scene-to-primitives converter`
   - Converts the TSDF into a compact set of axis-aligned boxes or occupied voxels.
4. `MPD planning service`
   - Rebuilds the environment with `EnvWarehouse(..., obj_extra_list=...)`.
   - Runs IK for the target EE pose.
   - Calls `GenerativeOptimizationPlanner.plan_trajectory(...)`.
5. `trajectory follower`
   - First: replay in simulator.
   - Later: stream time-parameterized joint commands to the Franka controller.

## Implementation Phases

### Phase 1: Online Planning In A Static Simulated Scene

Objective:
Replace the hard-coded extra boxes in `EnvWarehouseExtraObjectsV00` with obstacles loaded at runtime.

Steps:

1. Add a utility that builds `ObjectField` / `MultiBoxField` instances from a list of boxes.
2. Add a planning entrypoint that accepts:
   - current Panda joint state
   - target EE pose
   - extra obstacle boxes
3. Create an environment with:
   - `EnvWarehouse(obj_extra_list=[...])`
4. Run IK to generate candidate goal joint states.
5. Run MPD with the pretrained warehouse checkpoint.
6. Visualize or replay the result in Isaac Gym or PyBullet.

Deliverable:
An end-to-end script that plans around runtime-specified obstacles without retraining the model.

### Phase 2: SAPIEN Depth Integration

Objective:
Swap manual obstacle boxes for depth-derived scene geometry.

Steps:

1. Build a SAPIEN scene matching the warehouse/table/shelf geometry.
2. Attach one or more virtual depth cameras with known extrinsics.
3. Export:
   - depth image
   - camera intrinsics
   - camera pose in world frame
4. Save a frame bundle format that can be fed to nvblox.

Deliverable:
A SAPIEN script that produces realistic depth observations for the planning scene.

### Phase 3: nvblox Reconstruction Bridge

Objective:
Turn depth images into a world-aligned voxel or TSDF representation.

Steps:

1. Wrap nvblox as an external process or service.
2. Feed depth frames and camera poses into nvblox.
3. Read back one of:
   - occupied voxel centers
   - TSDF slices / dense volume
   - mesh blocks / ESDF if available
4. Convert the output into MPD-friendly geometry.

Preferred first version:

- Start with occupied voxels.
- Merge neighboring voxels into larger boxes to keep the primitive count manageable.

Deliverable:
A converter that outputs `List[ObjectField]` for use as `obj_extra_list`.

### Phase 4: Online Planning Loop

Objective:
Make planning reactive instead of single-shot.

Loop:

1. Read latest joint state.
2. Reconstruct scene from recent depth frames.
3. Build/update extra obstacles.
4. Solve IK for the target EE pose.
5. Plan with MPD.
6. Check trajectory validity.
7. Send the selected trajectory to the follower.

Recommended behavior:

- Replan at low frequency, such as 1-2 Hz initially.
- Only replace the active trajectory if the new plan is collision-free and materially better.

### Phase 5: Real Franka Execution

Objective:
Replace the simulator follower with a real robot controller.

Steps:

1. Add a Franka interface layer that can:
   - read joint states
   - send joint position or joint impedance targets
   - stop safely
2. Resample MPD output to the controller rate.
3. Enforce:
   - joint limits
   - velocity limits
   - acceleration limits
4. Add watchdogs for:
   - stale perception
   - invalid IK
   - no valid trajectory
   - execution divergence

Deliverable:
A guarded trajectory executor for the physical Panda arm.

## Proposed Repo Additions

The cleanest way to implement this in this repo is to add a deployment package.

Suggested layout:

```text
mpd/deployment/
  __init__.py
  scene_primitives.py
  goal_ik.py
  online_planner.py
  trajectory_follower.py
  sapien_depth_adapter.py
  nvblox_bridge.py
  scene_voxelizer.py

scripts/deployment/
  run_online_planner_sim.py
  run_sapien_depth_capture.py
  run_nvblox_bridge.py
  run_real_franka_planner.py
```

### Module Responsibilities

`scene_primitives.py`

- Convert voxels or boxes into `ObjectField([MultiBoxField(...)], name=...)`
- Merge voxels into larger boxes
- Filter noise and tiny disconnected components

`goal_ik.py`

- Use the Panda differentiable model to generate candidate `q_goal`
- Rank IK solutions by distance from current joint state and collision validity

`online_planner.py`

- Own environment construction
- Inject runtime obstacle primitives through `obj_extra_list`
- Initialize dataset and `GenerativeOptimizationPlanner`
- Choose the best valid trajectory

`trajectory_follower.py`

- Simulator replay first
- Real Franka command publisher later
- Time resampling and safety clamps

`sapien_depth_adapter.py`

- Render depth frames
- Export intrinsics/extrinsics and depth tensors

`nvblox_bridge.py`

- Handle process/service boundary to nvblox
- Translate frame bundles in and volume data out

`scene_voxelizer.py`

- Convert nvblox volume output into a box set or voxel occupancy list

## Minimal Viable Path

If the goal is to get a demo working quickly, do this first:

1. Skip SAPIEN.
2. Skip nvblox.
3. Feed synthetic extra boxes directly into `EnvWarehouse`.
4. Add IK for a user-specified EE target.
5. Run MPD and replay in Isaac Gym.

That gives a full online planning prototype with the smallest number of moving parts.

Then add:

1. SAPIEN depth
2. nvblox reconstruction
3. real robot execution

## Main Technical Risks

### 1. Goal representation mismatch

The current planner still needs `q_goal`, not only an EE goal. If IK is unstable or returns goals in collision, MPD will fail before planning starts.

Mitigation:

- Generate multiple IK candidates.
- Collision-check them with the current environment.
- Run MPD for the top few candidates.

### 2. Too many obstacle primitives

A dense voxel map converted directly into thousands of boxes will make SDF generation expensive.

Mitigation:

- Downsample voxels.
- Merge contiguous voxels into coarse boxes.
- Restrict the planning volume to the robot workspace.

### 3. Domain gap between training and deployment scene

The warehouse model was trained on box-primitive scenes. A noisy reconstructed TSDF may not match that distribution well.

Mitigation:

- Convert reconstruction output into box primitives, not arbitrary meshes.
- Keep obstacle geometry coarse and conservative.
- Start with scenes that resemble the training distribution.

### 4. Real-time execution and replanning

Depth, reconstruction, IK, and planning together may be too slow for aggressive online control.

Mitigation:

- Start with stop-plan-execute.
- Move to slow replanning later.
- Cache the planner and keep model warm.

## First Concrete Coding Task

The best first implementation step in this repo is:

1. Add `mpd/deployment/scene_primitives.py`
2. Add `mpd/deployment/goal_ik.py`
3. Add `mpd/deployment/online_planner.py`
4. Add `scripts/deployment/run_online_planner_sim.py`

This first version should:

- take manual extra boxes
- take a target EE pose
- solve IK for `q_goal`
- inject runtime obstacles into `EnvWarehouse`
- run MPD
- replay the selected trajectory in simulation

Once that works, the SAPIEN and nvblox pieces become adapters around a planning core that is already validated.
