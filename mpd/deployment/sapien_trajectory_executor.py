from __future__ import annotations

from dataclasses import asdict
import gc
import json
import os

import matplotlib.pyplot as plt
import numpy as np

from mpd.deployment.sapien_depth_adapter import (
    _build_box_actor,
    _create_scene,
    _create_render_cameras,
    _load_robot_articulation,
    _matrix_to_sapien_pose,
    DepthCameraSpec,
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


def _capture_camera_rgb(scene, camera):
    scene.update_render()
    camera.take_picture()
    color = np.asarray(camera.get_picture("Color"), dtype=np.float32)
    if color.ndim != 3 or color.shape[2] < 3:
        raise RuntimeError(f"Expected SAPIEN Color picture with shape (H, W, C>=3), got {color.shape}")
    return np.clip(color[..., :3], 0.0, 1.0)


def _sample_trajectory_evenly(q_traj_np, timesteps_np, num_samples):
    num_samples = max(int(num_samples), 1)
    if q_traj_np.shape[0] == 1 or num_samples == 1:
        return q_traj_np[[0]], np.asarray([timesteps_np[0]], dtype=np.float32)

    if np.any(np.diff(timesteps_np) <= 0.0):
        sampled_indices = np.linspace(0, q_traj_np.shape[0] - 1, num_samples)
        sample_times = np.linspace(float(timesteps_np[0]), float(timesteps_np[-1]), num_samples, dtype=np.float32)
        sampled_q = np.stack(
            [
                np.interp(sampled_indices, np.arange(q_traj_np.shape[0], dtype=np.float32), q_traj_np[:, dof_idx])
                for dof_idx in range(q_traj_np.shape[1])
            ],
            axis=-1,
        )
        return sampled_q.astype(np.float32), sample_times

    sample_times = np.linspace(float(timesteps_np[0]), float(timesteps_np[-1]), num_samples, dtype=np.float32)
    sampled_q = np.stack(
        [np.interp(sample_times, timesteps_np, q_traj_np[:, dof_idx]) for dof_idx in range(q_traj_np.shape[1])],
        axis=-1,
    )
    return sampled_q.astype(np.float32), sample_times


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


def _with_box_pose_world(box_spec):
    if "pose_world" in box_spec and box_spec["pose_world"] is not None:
        return dict(box_spec)

    pose_world = np.eye(4, dtype=np.float32)
    pose_world[:3, 3] = np.asarray(box_spec["center"], dtype=np.float32)
    enriched = dict(box_spec)
    enriched["pose_world"] = pose_world.tolist()
    enriched.setdefault("source", "reconstructed")
    return enriched


def save_trajectory_composite_image_in_sapien(
    q_traj,
    timesteps,
    scene_spec,
    camera_spec,
    save_path,
    robot_cfg=None,
    num_samples=32,
    add_ground=False,
    alpha=0.35,
    alpha_ramp=True,
    robot_mask_threshold=0.03,
):
    q_traj_np = _as_numpy_trajectory(q_traj)
    timesteps_np = _as_numpy_timesteps(
        timesteps=timesteps,
        num_points=q_traj_np.shape[0],
        trajectory_duration=max(float(q_traj_np.shape[0] - 1), 1.0),
    )
    sampled_q, sample_times = _sample_trajectory_evenly(q_traj_np, timesteps_np, num_samples)

    if isinstance(camera_spec, dict):
        camera_spec = DepthCameraSpec(**camera_spec)
    if not isinstance(camera_spec, DepthCameraSpec):
        raise TypeError(f"camera_spec must be a DepthCameraSpec or dict, got {type(camera_spec).__name__}")

    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)

    background_sapien = None
    background_scene = None
    sapien = None
    scene = None
    background_cameras = None
    cameras = None
    robot = None
    try:
        background_sapien, background_scene, _ = _create_scene(scene_spec=scene_spec, add_ground=add_ground)
        sapien, scene, _ = _create_scene(scene_spec=scene_spec, add_ground=add_ground)
        background_cameras = _create_render_cameras(background_scene, background_sapien, [camera_spec])
        cameras = _create_render_cameras(scene, sapien, [camera_spec])

        robot_cfg = dict(robot_cfg or {})
        robot_cfg["enabled"] = True
        robot_cfg["fix_root_link"] = bool(robot_cfg.get("fix_root_link", True))
        robot_cfg["qpos"] = sampled_q[0].tolist()
        robot = _load_robot_articulation(scene, sapien, robot_cfg)
        if robot is None:
            raise RuntimeError("Failed to create the SAPIEN robot articulation for trajectory composite rendering")
        if sampled_q.shape[1] != robot.dof:
            raise ValueError(
                f"Trajectory dof mismatch: planner produced {sampled_q.shape[1]} joints but SAPIEN robot has {robot.dof}"
            )

        background_rgb = _capture_camera_rgb(background_scene, background_cameras[0])
        composite = background_rgb.copy()
        mask_fractions = []
        alpha = float(alpha)

        for sample_idx, q_sample in enumerate(sampled_q):
            robot.set_qpos(q_sample)
            if hasattr(robot, "set_qvel"):
                robot.set_qvel(np.zeros((robot.dof,), dtype=np.float32))
            robot_rgb = _capture_camera_rgb(scene, cameras[0])
            robot_mask = np.linalg.norm(robot_rgb - background_rgb, axis=-1) > float(robot_mask_threshold)
            mask_fractions.append(float(robot_mask.mean()))
            if not np.any(robot_mask):
                continue

            if alpha_ramp and sampled_q.shape[0] > 1:
                sample_alpha = alpha * (0.45 + 0.55 * (sample_idx / float(sampled_q.shape[0] - 1)))
            else:
                sample_alpha = alpha
            sample_alpha = float(np.clip(sample_alpha, 0.0, 1.0))
            composite[robot_mask] = (1.0 - sample_alpha) * composite[robot_mask] + sample_alpha * robot_rgb[robot_mask]

        plt.imsave(save_path, np.clip(composite, 0.0, 1.0))
        return {
            "save_path": save_path,
            "camera_name": camera_spec.name,
            "image_width": int(camera_spec.width),
            "image_height": int(camera_spec.height),
            "num_samples_requested": int(num_samples),
            "num_samples_rendered": int(sampled_q.shape[0]),
            "sample_times": sample_times.tolist(),
            "alpha": alpha,
            "alpha_ramp": bool(alpha_ramp),
            "robot_mask_threshold": float(robot_mask_threshold),
            "mean_robot_mask_fraction": float(np.mean(mask_fractions)) if mask_fractions else 0.0,
            "max_robot_mask_fraction": float(np.max(mask_fractions)) if mask_fractions else 0.0,
        }
    finally:
        if robot is not None:
            del robot
        if cameras is not None:
            del cameras
        if background_cameras is not None:
            del background_cameras
        if scene is not None:
            del scene
        if background_scene is not None:
            del background_scene
        if sapien is not None:
            del sapien
        if background_sapien is not None:
            del background_sapien


def _add_sphere_actor(scene, sapien_module, sphere_spec, color=(0.15, 0.8, 0.2)):
    center = np.asarray(sphere_spec["center"], dtype=np.float32)
    radius = float(sphere_spec["radius"])
    builder = scene.create_actor_builder()
    builder.add_sphere_visual(radius=radius, material=list(color))
    actor = builder.build_static(name=sphere_spec.get("name", "debug_sphere"))
    pose = np.eye(4, dtype=np.float32)
    pose[:3, 3] = center
    actor.set_pose(_matrix_to_sapien_pose(sapien_module, pose))
    return actor


def visualize_esdf_debug_in_sapien(
    scene_spec,
    robot_cfg,
    esdf_points,
    esdf_values,
    render_viewer=True,
    add_ground=False,
    viewer_preset="isaac_gym_default",
    step_physics=False,
    point_radius=0.015,
):
    esdf_points = np.asarray(esdf_points, dtype=np.float32)
    esdf_values = np.asarray(esdf_values, dtype=np.float32).reshape(-1)

    if esdf_points.ndim != 2 or esdf_points.shape[1] != 3:
        raise ValueError(f"Expected esdf_points with shape (N, 3), got {esdf_points.shape}")
    if esdf_points.shape[0] != esdf_values.shape[0]:
        raise ValueError(f"esdf_points/esdf_values size mismatch: {esdf_points.shape[0]} vs {esdf_values.shape[0]}")

    if not render_viewer:
        return {
            "n_esdf_points": int(esdf_points.shape[0]),
            "esdf_min": float(esdf_values.min()) if esdf_values.size else 0.0,
            "esdf_max": float(esdf_values.max()) if esdf_values.size else 0.0,
        }

    sapien, scene, _ = _create_scene(scene_spec=scene_spec, add_ground=add_ground)
    viewer = None
    try:
        robot_cfg = dict(robot_cfg or {})
        robot_cfg["enabled"] = True
        _load_robot_articulation(scene, sapien, robot_cfg)

        if esdf_values.size:
            esdf_min = float(esdf_values.min())
            esdf_max = float(esdf_values.max())
            denom = max(esdf_max - esdf_min, 1e-6)
            colors = plt.get_cmap("plasma")((esdf_values - esdf_min) / denom)[:, :3]
        else:
            esdf_min = 0.0
            esdf_max = 0.0
            colors = np.zeros((0, 3), dtype=np.float32)

        for idx, (point, color) in enumerate(zip(esdf_points, colors)):
            _add_sphere_actor(
                scene,
                sapien,
                {
                    "name": f"esdf_debug_point_{idx}",
                    "center": point.tolist(),
                    "radius": float(point_radius),
                },
                color=tuple(float(v) for v in color),
            )

        viewer = scene.create_viewer()
        _configure_viewer(viewer, scene_spec, sapien, view_preset=viewer_preset)

        while not getattr(viewer, "closed", False):
            if step_physics:
                scene.step()
            scene.update_render()
            viewer.render()

        return {
            "n_esdf_points": int(esdf_points.shape[0]),
            "esdf_min": esdf_min,
            "esdf_max": esdf_max,
            "viewer_closed_early": bool(getattr(viewer, "closed", False)),
        }
    finally:
        if viewer is not None:
            del viewer
        del scene
        del sapien


def visualize_box_collision_debug_in_sapien(
    scene_spec,
    robot_cfg,
    colliding_boxes,
    noncolliding_boxes=None,
    robot_collision_spheres=None,
    render_viewer=True,
    add_ground=False,
    viewer_preset="isaac_gym_default",
    step_physics=False,
):
    if not render_viewer:
        return {
            "n_colliding_boxes": len(colliding_boxes or []),
            "n_noncolliding_boxes": len(noncolliding_boxes or []),
            "n_robot_collision_spheres": len(robot_collision_spheres or []),
        }

    sapien, scene, _ = _create_scene(scene_spec=scene_spec, add_ground=add_ground)
    viewer = None
    try:
        robot_cfg = dict(robot_cfg or {})
        robot_cfg["enabled"] = True
        _load_robot_articulation(scene, sapien, robot_cfg)

        for box_spec in noncolliding_boxes or []:
            _build_box_actor(
                scene,
                sapien,
                _with_box_pose_world(box_spec),
                add_collision=False,
                color_by_source=False,
                color_override=[0.2, 0.6, 0.95],
            )

        for box_spec in colliding_boxes or []:
            _build_box_actor(
                scene,
                sapien,
                _with_box_pose_world(box_spec),
                add_collision=False,
                color_by_source=False,
                color_override=[0.9, 0.15, 0.15],
            )

        for sphere_spec in robot_collision_spheres or []:
            _add_sphere_actor(scene, sapien, sphere_spec, color=(0.15, 0.8, 0.2))

        viewer = scene.create_viewer()
        _configure_viewer(viewer, scene_spec, sapien, view_preset=viewer_preset)

        while not getattr(viewer, "closed", False):
            if step_physics:
                scene.step()
            scene.update_render()
            viewer.render()

        return {
            "n_colliding_boxes": len(colliding_boxes or []),
            "n_noncolliding_boxes": len(noncolliding_boxes or []),
            "n_robot_collision_spheres": len(robot_collision_spheres or []),
            "viewer_closed_early": bool(getattr(viewer, "closed", False)),
        }
    finally:
        if viewer is not None:
            del viewer
        del scene
        del sapien


def visualize_phase3_scene_debug_in_sapien(
    scene_spec,
    robot_cfg,
    reconstructed_boxes,
    robot_collision_spheres=None,
    render_viewer=True,
    add_ground=False,
    viewer_preset="isaac_gym_default",
    step_physics=False,
):
    if not render_viewer:
        return {
            "n_reconstructed_boxes": len(reconstructed_boxes or []),
            "n_robot_collision_spheres": len(robot_collision_spheres or []),
        }

    sapien, scene, _ = _create_scene(scene_spec=scene_spec, add_ground=add_ground)
    viewer = None
    try:
        robot_cfg = dict(robot_cfg or {})
        robot_cfg["enabled"] = True
        _load_robot_articulation(scene, sapien, robot_cfg)

        for box_spec in reconstructed_boxes or []:
            _build_box_actor(
                scene,
                sapien,
                _with_box_pose_world(box_spec),
                add_collision=False,
                color_by_source=False,
                color_override=[0.9, 0.15, 0.15],
            )

        for sphere_spec in robot_collision_spheres or []:
            _add_sphere_actor(scene, sapien, sphere_spec, color=(0.15, 0.8, 0.2))

        viewer = scene.create_viewer()
        _configure_viewer(viewer, scene_spec, sapien, view_preset=viewer_preset)

        while not getattr(viewer, "closed", False):
            if step_physics:
                scene.step()
            scene.update_render()
            viewer.render()

        return {
            "n_reconstructed_boxes": len(reconstructed_boxes or []),
            "n_robot_collision_spheres": len(robot_collision_spheres or []),
            "viewer_closed_early": bool(getattr(viewer, "closed", False)),
        }
    finally:
        if viewer is not None:
            del viewer
        del scene
        del sapien


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
            "median_tracking_error_l2": (
                float(np.median(tracking_error_history)) if tracking_error_history.size else 0.0
            ),
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


class PersistentSapienTrajectoryExecutor:
    def __init__(self, viewer_preset="isaac_gym_default"):
        self.viewer_preset = viewer_preset
        self.viewer = None
        self.scene = None
        self.sapien = None
        self.robot = None
        self._scene_cache_key = None

    def close(self):
        if self.viewer is not None:
            del self.viewer
            self.viewer = None
        if self.scene is not None:
            del self.scene
            self.scene = None
        self.sapien = None
        self.robot = None
        self._scene_cache_key = None
        gc.collect()

    @staticmethod
    def _json_default(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")

    def _make_scene_cache_key(self, scene_spec, robot_cfg, add_ground):
        cache_robot_cfg = dict(robot_cfg or {})
        cache_robot_cfg.pop("qpos", None)
        return json.dumps(
            {
                "scene_spec": scene_spec,
                "robot_cfg": cache_robot_cfg,
                "add_ground": bool(add_ground),
            },
            sort_keys=True,
            default=self._json_default,
        )

    def execute_trajectory(
        self,
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
        viewer_preset=None,
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

        scene_cache_key = self._make_scene_cache_key(scene_spec, robot_cfg, add_ground)
        scene_needs_rebuild = (
            self.scene is None or self.sapien is None or self.robot is None or self._scene_cache_key != scene_cache_key
        )

        if scene_needs_rebuild:
            new_sapien, new_scene, _ = _create_scene(scene_spec=scene_spec, add_ground=add_ground)
            new_scene.set_timestep(float(scene_timestep))
            robot = _load_robot_articulation(new_scene, new_sapien, robot_cfg)
            if robot is None:
                del new_scene
                raise RuntimeError("Failed to create the SAPIEN robot articulation for trajectory replay")

            old_scene = self.scene
            self.scene = new_scene
            self.sapien = new_sapien
            self.robot = robot
            self._scene_cache_key = scene_cache_key

            if old_scene is not None:
                del old_scene
        else:
            new_sapien = self.sapien
            new_scene = self.scene
            robot = self.robot
            new_scene.set_timestep(float(scene_timestep))
            if hasattr(robot, "set_qpos"):
                robot.set_qpos(q_traj_np[0])

        if q_traj_np.shape[1] != robot.dof:
            if scene_needs_rebuild:
                self.close()
            raise ValueError(
                f"Trajectory dof mismatch: planner produced {q_traj_np.shape[1]} joints but SAPIEN robot has {robot.dof}"
            )

        if hasattr(robot, "set_qvel"):
            robot.set_qvel(np.zeros((robot.dof,), dtype=np.float32))

        active_joints = robot.get_active_joints()
        if len(active_joints) != robot.dof:
            if scene_needs_rebuild:
                self.close()
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
            if self.viewer is None or getattr(self.viewer, "closed", False):
                self.viewer = new_scene.create_viewer()
            elif scene_needs_rebuild:
                self.viewer.set_scene(new_scene)
            _configure_viewer(
                self.viewer,
                scene_spec,
                new_sapien,
                view_preset=viewer_preset or self.viewer_preset,
            )

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
            new_scene.step()
            sim_steps += 1
            qpos = np.asarray(robot.get_qpos(), dtype=np.float32)
            q_target_history.append(np.asarray(q_target, dtype=np.float32))
            tracking_error_history.append(float(np.linalg.norm(qpos - q_target)))
            if self.viewer is not None and sim_steps % max(int(render_every_n_steps), 1) == 0:
                new_scene.update_render()
                self.viewer.render()

        for _ in range(max(int(n_pre_steps), 0)):
            if self.viewer is not None and getattr(self.viewer, "closed", False):
                break
            _step_once(q_traj_np[0])

        for waypoint_idx in range(q_traj_np.shape[0] - 1):
            if self.viewer is not None and getattr(self.viewer, "closed", False):
                break
            q_start = q_traj_np[waypoint_idx]
            q_goal = q_traj_np[waypoint_idx + 1]
            dt_segment = float(timesteps_np[waypoint_idx + 1] - timesteps_np[waypoint_idx])
            n_segment_steps = _compute_segment_step_count(dt_segment, float(scene_timestep))
            for step_idx in range(n_segment_steps):
                if self.viewer is not None and getattr(self.viewer, "closed", False):
                    break
                alpha = float(step_idx + 1) / float(n_segment_steps)
                q_target = (1.0 - alpha) * q_start + alpha * q_goal
                _step_once(q_target)

        for _ in range(max(int(n_post_steps), 0)):
            if self.viewer is not None and getattr(self.viewer, "closed", False):
                break
            _step_once(q_traj_np[-1])

        if self.viewer is not None:
            new_scene.update_render()
            self.viewer.render()

        q_target_history = np.asarray(q_target_history, dtype=np.float32)
        tracking_error_history = np.asarray(tracking_error_history, dtype=np.float32)

        return {
            "scene_timestep": float(scene_timestep),
            "trajectory_duration": float(timesteps_np[-1] - timesteps_np[0]) if timesteps_np.size > 1 else 0.0,
            "num_waypoints": int(q_traj_np.shape[0]),
            "num_sim_steps": int(sim_steps),
            "max_tracking_error_l2": float(np.max(tracking_error_history)) if tracking_error_history.size else 0.0,
            "median_tracking_error_l2": (
                float(np.median(tracking_error_history)) if tracking_error_history.size else 0.0
            ),
            "mean_tracking_error_l2": float(np.mean(tracking_error_history)) if tracking_error_history.size else 0.0,
            "final_tracking_error_l2": float(tracking_error_history[-1]) if tracking_error_history.size else 0.0,
            "final_qpos": np.asarray(robot.get_qpos(), dtype=np.float32).tolist(),
            "final_q_target": q_target_history[-1].tolist() if q_target_history.size else q_traj_np[-1].tolist(),
            "viewer_closed_early": bool(self.viewer is not None and getattr(self.viewer, "closed", False)),
        }
