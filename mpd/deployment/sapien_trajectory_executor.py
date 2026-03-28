from __future__ import annotations

from dataclasses import asdict

import numpy as np

from mpd.deployment.sapien_depth_adapter import (
    _create_scene,
    _load_robot_articulation,
    _matrix_to_sapien_pose,
    environment_to_scene_boxes,
    make_camera_pose_look_at,
)


def build_scene_spec_from_planning_env(env):
    return {
        "environment": env.__class__.__name__,
        "workspace_limits": np.asarray(env.limits.detach().cpu(), dtype=float).tolist(),
        "boxes": [asdict(box_spec) for box_spec in environment_to_scene_boxes(env)],
    }


def _as_numpy_trajectory(q_traj):
    if hasattr(q_traj, "detach"):
        q_traj = q_traj.detach().cpu().numpy()
    q_traj = np.asarray(q_traj, dtype=np.float32)
    if q_traj.ndim != 2:
        raise ValueError(f"Expected q_traj with shape (T, dof), got {q_traj.shape}")
    return q_traj


def _as_numpy_timesteps(timesteps, num_points, trajectory_duration):
    if timesteps is None:
        return np.linspace(0.0, float(trajectory_duration), num_points, dtype=np.float32)

    if hasattr(timesteps, "detach"):
        timesteps = timesteps.detach().cpu().numpy()
    timesteps = np.asarray(timesteps, dtype=np.float32).reshape(-1)
    if timesteps.shape[0] != num_points:
        raise ValueError(f"Expected {num_points} trajectory timesteps, got {timesteps.shape[0]}")
    return timesteps


def _expand_joint_parameter(value, dof, name):
    array = np.asarray(value, dtype=np.float32)
    if array.ndim == 0:
        return np.full((dof,), float(array), dtype=np.float32)
    if array.shape != (dof,):
        raise ValueError(f"{name} must be a scalar or have shape ({dof},), got {array.shape}")
    return array


def _set_drive_targets(active_joints, q_target, qvel_target=None):
    if qvel_target is None:
        qvel_target = np.zeros_like(q_target)
    for joint, target, velocity_target in zip(active_joints, q_target, qvel_target):
        joint.set_drive_target(float(target))
        joint.set_drive_velocity_target(float(velocity_target))


def _compute_segment_step_count(segment_duration, scene_timestep):
    if segment_duration <= 0.0:
        return 1
    return max(1, int(np.ceil(segment_duration / scene_timestep)))


def _collect_box_bounds(scene_spec):
    if not scene_spec.get("boxes"):
        mins = np.array([-0.5, -0.5, 0.0], dtype=np.float32)
        maxs = np.array([0.5, 0.5, 1.0], dtype=np.float32)
        return mins, maxs

    mins = []
    maxs = []
    for box in scene_spec["boxes"]:
        center = np.asarray(box["center"], dtype=np.float32)
        half_size = 0.5 * np.asarray(box["size"], dtype=np.float32)
        mins.append(center - half_size)
        maxs.append(center + half_size)
    return np.min(np.stack(mins, axis=0), axis=0), np.max(np.stack(maxs, axis=0), axis=0)


def _configure_viewer(viewer, scene_spec, sapien_module, view_preset="isaac_gym_default"):
    if view_preset == "isaac_gym_default":
        camera_position = np.array([1.2, -1.2, 0.9], dtype=np.float32)
        camera_target = np.array([0.1, 0.1, 0.5], dtype=np.float32)
    else:
        mins, maxs = _collect_box_bounds(scene_spec)
        center = 0.5 * (mins + maxs)
        extent = float(np.max(maxs - mins))
        extent = max(extent, 1.0)
        camera_position = np.array(
            [
                center[0] - 1.5 * extent,
                center[1] - 1.5 * extent,
                center[2] + 1.1 * extent,
            ],
            dtype=np.float32,
        )
        camera_target = center

    camera_pose_world = make_camera_pose_look_at(camera_position, camera_target)
    viewer.set_camera_pose(_matrix_to_sapien_pose(sapien_module, camera_pose_world))


def execute_trajectory_in_sapien(
    q_traj,
    timesteps,
    scene_spec,
    robot_cfg=None,
    render_viewer=True,
    add_ground=False,
    scene_timestep=1.0 / 240.0,
    render_every_n_steps=4,
    stiffness=200.0,
    damping=40.0,
    force_limit=1000.0,
    drive_mode="force",
    balance_passive_force=True,
    compensate_gravity=True,
    compensate_coriolis_and_centrifugal=True,
    n_pre_steps=5,
    n_post_steps=10,
    viewer_preset="isaac_gym_default",
):
    q_traj_np = _as_numpy_trajectory(q_traj)
    timesteps_np = _as_numpy_timesteps(
        timesteps=timesteps,
        num_points=q_traj_np.shape[0],
        trajectory_duration=max(float(q_traj_np.shape[0] - 1), 1.0),
    )

    robot_cfg = dict(robot_cfg or {})
    robot_cfg["enabled"] = True
    robot_cfg["fix_root_link"] = bool(robot_cfg.get("fix_root_link", True))
    robot_cfg["qpos"] = q_traj_np[0].tolist()

    sapien, scene, _ = _create_scene(scene_spec=scene_spec, add_ground=add_ground)
    viewer = None
    try:
        scene.set_timestep(float(scene_timestep))
        robot = _load_robot_articulation(scene, sapien, robot_cfg)
        if robot is None:
            raise RuntimeError("Failed to create the SAPIEN robot articulation for trajectory replay")

        if q_traj_np.shape[1] != robot.dof:
            raise ValueError(
                f"Trajectory dof mismatch: planner produced {q_traj_np.shape[1]} joints but SAPIEN robot has {robot.dof}"
            )

        if hasattr(robot, "set_qvel"):
            robot.set_qvel(np.zeros((robot.dof,), dtype=np.float32))

        active_joints = robot.get_active_joints()
        if len(active_joints) != robot.dof:
            raise RuntimeError(
                f"Unexpected active joint count for replay: expected {robot.dof}, got {len(active_joints)}"
            )

        stiffness_per_joint = _expand_joint_parameter(stiffness, robot.dof, "stiffness")
        damping_per_joint = _expand_joint_parameter(damping, robot.dof, "damping")
        force_limit_per_joint = _expand_joint_parameter(force_limit, robot.dof, "force_limit")

        for joint_idx, joint in enumerate(active_joints):
            joint.set_drive_property(
                stiffness=float(stiffness_per_joint[joint_idx]),
                damping=float(damping_per_joint[joint_idx]),
                force_limit=float(force_limit_per_joint[joint_idx]),
                mode=str(drive_mode),
            )

        if render_viewer:
            viewer = scene.create_viewer()
            _configure_viewer(viewer, scene_spec, sapien, view_preset=viewer_preset)

        q_target_history = []
        tracking_error_history = []
        sim_steps = 0

        def _step_once(q_target):
            nonlocal sim_steps
            _set_drive_targets(active_joints, q_target)
            if balance_passive_force:
                qf = robot.compute_passive_force(
                    gravity=bool(compensate_gravity),
                    coriolis_and_centrifugal=bool(compensate_coriolis_and_centrifugal),
                )
                robot.set_qf(np.asarray(qf, dtype=np.float32))
            scene.step()
            sim_steps += 1
            qpos = np.asarray(robot.get_qpos(), dtype=np.float32)
            q_target_history.append(np.asarray(q_target, dtype=np.float32))
            tracking_error_history.append(float(np.linalg.norm(qpos - q_target)))
            if viewer is not None and sim_steps % max(int(render_every_n_steps), 1) == 0:
                scene.update_render()
                viewer.render()

        for _ in range(max(int(n_pre_steps), 0)):
            if viewer is not None and getattr(viewer, "closed", False):
                break
            _step_once(q_traj_np[0])

        for waypoint_idx in range(q_traj_np.shape[0] - 1):
            if viewer is not None and getattr(viewer, "closed", False):
                break

            q_start = q_traj_np[waypoint_idx]
            q_goal = q_traj_np[waypoint_idx + 1]
            dt_segment = float(timesteps_np[waypoint_idx + 1] - timesteps_np[waypoint_idx])
            n_segment_steps = _compute_segment_step_count(dt_segment, float(scene_timestep))
            for step_idx in range(n_segment_steps):
                if viewer is not None and getattr(viewer, "closed", False):
                    break
                alpha = float(step_idx + 1) / float(n_segment_steps)
                q_target = (1.0 - alpha) * q_start + alpha * q_goal
                _step_once(q_target)

        for _ in range(max(int(n_post_steps), 0)):
            if viewer is not None and getattr(viewer, "closed", False):
                break
            _step_once(q_traj_np[-1])

        if viewer is not None:
            scene.update_render()
            viewer.render()

        q_target_history = np.asarray(q_target_history, dtype=np.float32)
        tracking_error_history = np.asarray(tracking_error_history, dtype=np.float32)

        return {
            "scene_timestep": float(scene_timestep),
            "trajectory_duration": float(timesteps_np[-1] - timesteps_np[0]) if timesteps_np.size > 1 else 0.0,
            "num_waypoints": int(q_traj_np.shape[0]),
            "num_sim_steps": int(sim_steps),
            "max_tracking_error_l2": float(np.max(tracking_error_history)) if tracking_error_history.size else 0.0,
            "mean_tracking_error_l2": float(np.mean(tracking_error_history)) if tracking_error_history.size else 0.0,
            "final_tracking_error_l2": float(tracking_error_history[-1]) if tracking_error_history.size else 0.0,
            "final_qpos": np.asarray(robot.get_qpos(), dtype=np.float32).tolist(),
            "final_q_target": q_target_history[-1].tolist() if q_target_history.size else q_traj_np[-1].tolist(),
            "viewer_closed_early": bool(viewer is not None and getattr(viewer, "closed", False)),
        }
    finally:
        if viewer is not None:
            del viewer
        del scene
        del sapien
