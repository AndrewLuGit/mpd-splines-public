from mpd.utils.patches import numpy_monkey_patch

numpy_monkey_patch()

import argparse
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib
import numpy as np
import torch
import viser
import yaml
from scipy.spatial.transform import Rotation
from viser.extras import ViserUrdf

from mpd.deployment.goal_ik import build_ee_pose_goal, build_ee_pose_goal_from_dict
from mpd.deployment.online_planner import OnlineMPDPlanner
from mpd.deployment.sapien_trajectory_executor import (
    PersistentSapienTrajectoryExecutor,
    build_scene_spec_from_planning_env,
)
from mpd.paths import REPO_PATH
from mpd.utils.loaders import load_params_from_yaml, save_to_yaml
from torch_robotics.torch_utils.seed import fix_random_seed
from torch_robotics.torch_utils.torch_utils import get_torch_device, to_torch

matplotlib.use("Agg", force=True)


def _resolve_repo_path(path):
    if os.path.isabs(path):
        return path
    return os.path.join(REPO_PATH, path)


def _load_extra_boxes(cfg):
    extra_boxes = list(cfg.get("extra_boxes") or [])
    extra_boxes_path = cfg.get("extra_boxes_path")
    if not extra_boxes_path:
        return extra_boxes

    with open(_resolve_repo_path(extra_boxes_path), "r") as stream:
        loaded_data = yaml.safe_load(stream) or {}

    if isinstance(loaded_data, dict):
        loaded_boxes = list(loaded_data.get("extra_boxes") or [])
    elif isinstance(loaded_data, list):
        loaded_boxes = list(loaded_data)
    else:
        raise ValueError("extra_boxes_path must point to a YAML list or a mapping with an 'extra_boxes' field")

    if cfg.get("append_extra_boxes_from_path", False):
        return extra_boxes + loaded_boxes
    return loaded_boxes


def _pose_to_viser_wxyz_and_position(ee_pose_goal):
    if hasattr(ee_pose_goal, "detach"):
        pose = ee_pose_goal.detach().cpu().numpy()
    else:
        pose = np.asarray(ee_pose_goal, dtype=np.float32)

    pose = np.asarray(pose, dtype=np.float32)
    if pose.shape == (3, 4):
        pose_h = np.eye(4, dtype=np.float32)
        pose_h[:3, :] = pose
        pose = pose_h
    elif pose.shape != (4, 4):
        raise ValueError(f"Expected goal pose with shape (3, 4) or (4, 4), got {pose.shape}")

    quat_xyzw = Rotation.from_matrix(pose[:3, :3]).as_quat()
    wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=np.float32)
    position = pose[:3, 3].astype(np.float32)
    return wxyz, position


def _viser_wxyz_to_xyzw(wxyz):
    wxyz = np.asarray(wxyz, dtype=np.float32).reshape(4)
    return np.array([wxyz[1], wxyz[2], wxyz[3], wxyz[0]], dtype=np.float32)


def _as_planner_ee_goal_pose(ee_pose_goal):
    if hasattr(ee_pose_goal, "detach"):
        pose = ee_pose_goal.detach().clone()
    else:
        pose = torch.as_tensor(ee_pose_goal, dtype=torch.float32)

    if pose.shape[-2:] == (3, 4):
        return pose
    if pose.shape[-2:] == (4, 4):
        return pose[..., :3, :4].clone()
    raise ValueError(f"Expected EE goal pose with shape (..., 3, 4) or (..., 4, 4), got {tuple(pose.shape)}")


def _safe_metric_scalar(value):
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return None
        return float(np.asarray(value).reshape(-1)[0])
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    return None


def _format_seconds(seconds):
    if seconds is None:
        return "-"
    return f"{seconds:.3f} s"


def _synchronize_torch_device(device):
    device = torch.device(device)
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)
    elif device.type == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "synchronize"):
        torch.mps.synchronize()


def _get_reference_subset_and_length(planner, selection):
    subset = planner.val_subset if selection == "validation" else planner.train_subset
    return subset, len(subset)


@dataclass
class RuntimeBoxWidget:
    name: str
    gizmo: object
    visual: object
    enabled: object
    size: object

    def current_box_spec(self):
        if not bool(self.enabled.value):
            return None
        center = np.asarray(self.gizmo.position, dtype=np.float32).reshape(3)
        size = np.asarray(self.size.value, dtype=np.float32).reshape(3)
        return {
            "name": self.name,
            "center": center.tolist(),
            "size": size.tolist(),
        }


def _make_runtime_box_widget(server, idx, box_spec):
    name = str(box_spec.get("name", f"runtime_box_{idx}"))
    center = np.asarray(box_spec.get("center", [0.6, 0.0, 0.2]), dtype=np.float32)
    size = np.asarray(box_spec.get("size", [0.2, 0.2, 0.2]), dtype=np.float32)

    gizmo = server.scene.add_transform_controls(
        f"/runtime_boxes/{name}",
        scale=0.18,
        position=center,
        disable_rotations=True,
    )
    visual = server.scene.add_box(
        f"/runtime_boxes/{name}/visual",
        color=(70, 150, 255),
        dimensions=size,
        opacity=0.55,
    )

    enabled = server.gui.add_checkbox(f"{name} enabled", True)
    size_handle = server.gui.add_vector3(
        f"{name} size",
        tuple(float(v) for v in size),
        min=(0.02, 0.02, 0.02),
        max=(1.5, 1.5, 1.5),
        step=0.01,
    )

    @enabled.on_update
    def _(_event):
        visible = bool(enabled.value)
        gizmo.visible = visible
        visual.visible = visible

    @size_handle.on_update
    def _(_event):
        visual.dimensions = tuple(float(v) for v in size_handle.value)

    return RuntimeBoxWidget(
        name=name,
        gizmo=gizmo,
        visual=visual,
        enabled=enabled,
        size=size_handle,
    )


def _add_fixed_scene_to_viser(server, scene_spec):
    server.scene.set_up_direction("+z")
    server.scene.add_grid(
        "/scene/grid",
        width=3.0,
        height=3.0,
        plane="xy",
        cell_size=0.1,
        section_size=0.5,
        cell_color=(220, 220, 220),
        section_color=(170, 170, 170),
        plane_opacity=0.0,
    )
    for box in scene_spec.get("boxes", []):
        center = np.asarray(box["center"], dtype=np.float32)
        size = np.asarray(box["size"], dtype=np.float32)
        source = box.get("source", "fixed")
        color = (170, 170, 170) if source == "fixed" else (120, 170, 255)
        opacity = 0.25 if source == "fixed" else 0.35
        server.scene.add_box(
            f"/scene/{source}/{box['name']}",
            color=color,
            dimensions=size,
            opacity=opacity,
            position=center,
        )


def _add_robot_q_init_spheres_to_viser(server, robot, q_init, sphere_margin=0.0):
    q_t = q_init
    if not hasattr(q_t, "detach"):
        q_t = to_torch(q_t, **robot.tensor_args)

    sphere_centers = robot.fk_map_collision(q_t).detach().cpu().numpy().reshape(-1, 3)
    sphere_radii = robot.link_collision_spheres_radii.detach().cpu().numpy().reshape(-1)

    group_name = "/robot_q_init"
    server.scene.add_frame(
        f"{group_name}/base",
        show_axes=False,
        axes_length=0.1,
        axes_radius=0.005,
        position=(0.0, 0.0, 0.0),
    )

    for idx, (center, radius) in enumerate(zip(sphere_centers, sphere_radii)):
        server.scene.add_icosphere(
            f"{group_name}/sphere_{idx}",
            radius=float(radius + sphere_margin),
            color=(80, 200, 110),
            opacity=0.45,
            position=center.astype(np.float32),
            subdivisions=2,
        )


def _add_robot_q_init_urdf_to_viser(server, robot, q_init, show_collision=False):
    q_t = q_init
    if hasattr(q_t, "detach"):
        q_np = q_t.detach().cpu().numpy().reshape(-1)
    else:
        q_np = np.asarray(q_t, dtype=np.float32).reshape(-1)

    robot_root = server.scene.add_frame("/robot_q_init", show_axes=False)
    viser_urdf = ViserUrdf(
        server,
        urdf_or_path=Path(robot.robot_urdf_file_raw),
        root_node_name="/robot_q_init",
        load_meshes=True,
        load_collision_meshes=bool(show_collision),
        collision_mesh_color_override=(1.0, 0.0, 0.0, 0.35),
    )
    viser_urdf.update_cfg(q_np)
    return robot_root, viser_urdf


def _update_robot_q_init_urdf(viser_urdf, q_init):
    q_t = q_init
    if hasattr(q_t, "detach"):
        q_np = q_t.detach().cpu().numpy().reshape(-1)
    else:
        q_np = np.asarray(q_t, dtype=np.float32).reshape(-1)
    viser_urdf.update_cfg(q_np)


def _refresh_robot_q_init_spheres(server, robot, q_init, sphere_margin=0.0):
    try:
        server.scene._handle_from_node_name["/robot_q_init"].remove()
    except Exception:
        pass
    _add_robot_q_init_spheres_to_viser(server, robot, q_init=q_init, sphere_margin=sphere_margin)


def _save_runtime_state(path, q_start, ee_pose_goal, extra_boxes):
    q_start_np = q_start.detach().cpu().numpy() if hasattr(q_start, "detach") else np.asarray(q_start, dtype=np.float32)
    ee_pose_np = (
        ee_pose_goal.detach().cpu().numpy()
        if hasattr(ee_pose_goal, "detach")
        else np.asarray(ee_pose_goal, dtype=np.float32)
    )
    save_to_yaml(
        {
            "q_start": np.asarray(q_start_np, dtype=float).tolist(),
            "ee_goal_pose_matrix": np.asarray(ee_pose_np, dtype=float).tolist(),
            "extra_boxes": list(extra_boxes),
        },
        path,
    )


def _remove_scene_node_if_present(server, path):
    try:
        handle = server.scene._handle_from_node_name.get(path)
        if handle is not None:
            handle.remove()
    except Exception:
        pass


def _clear_ik_debug_viser(server):
    _remove_scene_node_if_present(server, "/ik_debug")


def _add_robot_spheres_to_viser_group(server, robot, q, group_name, color, opacity=0.22, sphere_margin=0.0):
    q_t = q
    if not hasattr(q_t, "detach"):
        q_t = to_torch(q_t, **robot.tensor_args)

    sphere_centers = robot.fk_map_collision(q_t).detach().cpu().numpy().reshape(-1, 3)
    sphere_radii = robot.link_collision_spheres_radii.detach().cpu().numpy().reshape(-1)
    server.scene.add_frame(group_name, show_axes=False)
    for idx, (center, radius) in enumerate(zip(sphere_centers, sphere_radii)):
        server.scene.add_icosphere(
            f"{group_name}/sphere_{idx}",
            radius=float(radius + sphere_margin),
            color=color,
            opacity=float(opacity),
            position=center.astype(np.float32),
            subdivisions=1,
        )


def _visualize_ik_debug_in_viser(
    server,
    planner,
    max_collision_free_to_render=4,
    max_colliding_to_render=4,
    max_ik_valid_to_render=4,
    max_ik_refined_to_render=4,
    sphere_margin=0.0,
):
    _clear_ik_debug_viser(server)
    if planner.last_ik_debug_data is None:
        return

    ik_debug_data = planner.last_ik_debug_data
    robot = planner.planning_task.robot
    ee_pose_goal = ik_debug_data["ee_pose_goal"]
    ee_pose_goal_np = ee_pose_goal.detach().cpu().numpy() if hasattr(ee_pose_goal, "detach") else np.asarray(ee_pose_goal)
    if ee_pose_goal_np.shape == (3, 4):
        pose_h = np.eye(4, dtype=np.float32)
        pose_h[:3, :] = ee_pose_goal_np
        ee_pose_goal_np = pose_h

    quat_xyzw = Rotation.from_matrix(ee_pose_goal_np[:3, :3]).as_quat()
    wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=np.float32)
    position = ee_pose_goal_np[:3, 3].astype(np.float32)

    server.scene.add_frame(
        "/ik_debug/goal_pose",
        position=position,
        wxyz=wxyz,
        axes_length=0.18,
        axes_radius=0.01,
    )
    server.scene.add_label("/ik_debug/goal_pose_label", "IK goal", position=position + np.array([0.0, 0.0, 0.08]))

    q_candidates_collision_free = ik_debug_data["q_candidates_collision_free"]
    q_candidates_colliding = ik_debug_data["q_candidates_colliding"]
    q_candidates_ik_valid = ik_debug_data.get("q_candidates_ik_valid")
    q_candidates_ik_refined = ik_debug_data.get("q_candidates_ik_refined")
    idx_valid_refined_se3 = ik_debug_data.get("idx_valid_refined_se3")

    q_candidates_refined_se3_valid = None
    q_candidates_refined_se3_invalid = None
    if q_candidates_ik_refined is not None and idx_valid_refined_se3 is not None:
        idx_valid_refined_se3 = torch.atleast_1d(idx_valid_refined_se3).reshape(-1)
        if q_candidates_ik_refined.shape[0] > 0:
            refined_valid_mask = torch.zeros(
                q_candidates_ik_refined.shape[0],
                dtype=torch.bool,
                device=q_candidates_ik_refined.device,
            )
            if idx_valid_refined_se3.numel() > 0:
                refined_valid_mask[idx_valid_refined_se3] = True
            q_candidates_refined_se3_valid = q_candidates_ik_refined[refined_valid_mask]
            q_candidates_refined_se3_invalid = q_candidates_ik_refined[~refined_valid_mask]

    rendered_any_candidates = False

    for idx, q in enumerate(q_candidates_colliding[:max_colliding_to_render]):
        group_name = f"/ik_debug/colliding_{idx}"
        _add_robot_spheres_to_viser_group(
            server,
            robot,
            q,
            group_name=group_name,
            color=(235, 80, 80),
            opacity=0.18,
            sphere_margin=sphere_margin,
        )
        rendered_any_candidates = True

    for idx, q in enumerate(q_candidates_collision_free[:max_collision_free_to_render]):
        group_name = f"/ik_debug/collision_free_{idx}"
        _add_robot_spheres_to_viser_group(
            server,
            robot,
            q,
            group_name=group_name,
            color=(245, 170, 65),
            opacity=0.16,
            sphere_margin=sphere_margin,
        )
        rendered_any_candidates = True

    if not rendered_any_candidates and q_candidates_refined_se3_valid is not None:
        for idx, q in enumerate(q_candidates_refined_se3_valid[:max_ik_refined_to_render]):
            group_name = f"/ik_debug/refined_pose_valid_{idx}"
            _add_robot_spheres_to_viser_group(
                server,
                robot,
                q,
                group_name=group_name,
                color=(90, 150, 255),
                opacity=0.14,
                sphere_margin=sphere_margin,
            )
            rendered_any_candidates = True

    if not rendered_any_candidates and q_candidates_refined_se3_invalid is not None:
        for idx, q in enumerate(q_candidates_refined_se3_invalid[:max_ik_refined_to_render]):
            group_name = f"/ik_debug/refined_pose_miss_{idx}"
            _add_robot_spheres_to_viser_group(
                server,
                robot,
                q,
                group_name=group_name,
                color=(160, 120, 255),
                opacity=0.12,
                sphere_margin=sphere_margin,
            )
            rendered_any_candidates = True

    if not rendered_any_candidates and q_candidates_ik_valid is not None:
        for idx, q in enumerate(q_candidates_ik_valid[:max_ik_valid_to_render]):
            group_name = f"/ik_debug/ik_valid_{idx}"
            _add_robot_spheres_to_viser_group(
                server,
                robot,
                q,
                group_name=group_name,
                color=(170, 110, 255),
                opacity=0.12,
                sphere_margin=sphere_margin,
            )


def main():
    parser = argparse.ArgumentParser(description="Interactive warehouse demo: viser UI + SAPIEN execution")
    parser.add_argument(
        "--cfg",
        default="scripts/deployment/cfgs/interactive_viser_demo_warehouse.yaml",
        help="Path to the interactive demo YAML config",
    )
    args = parser.parse_args()

    cfg_path = _resolve_repo_path(args.cfg)
    cfg = load_params_from_yaml(cfg_path)

    fix_random_seed(cfg.get("seed", 0))

    device = get_torch_device(cfg.get("device", "cuda:0"))
    tensor_args = {"device": device, "dtype": torch.float32}

    results_dir = _resolve_repo_path(cfg.get("results_dir", "logs/interactive_viser_demo"))
    os.makedirs(results_dir, exist_ok=True)
    save_to_yaml(cfg, os.path.join(results_dir, "config.yaml"))

    planner = OnlineMPDPlanner(
        cfg_inference_path=cfg["cfg_inference_path"],
        extra_boxes=[],
        device=cfg.get("device", "cuda:0"),
        debug=cfg.get("debug", False),
        results_dir=results_dir,
        env_id_override=cfg.get("env_id_override", "EnvWarehouse"),
        env_sdf_cell_size=cfg.get("env_sdf_cell_size"),
    )
    reference_sample = planner.get_reference_sample(
        index=cfg.get("reference_index", 0),
        selection=cfg.get("reference_split", "validation"),
    )
    reference_split = cfg.get("reference_split", "validation")
    _, n_reference_samples = _get_reference_subset_and_length(planner, reference_split)
    scene_spec = build_scene_spec_from_planning_env(planner.planning_task.env)

    q_start = reference_sample.q_start
    if cfg.get("q_start") is not None:
        q_start = to_torch(cfg["q_start"], **tensor_args)

    ee_pose_goal = reference_sample.ee_goal_pose
    if cfg.get("ee_goal_pose") is not None:
        ee_pose_goal = _as_planner_ee_goal_pose(
            build_ee_pose_goal_from_dict(cfg["ee_goal_pose"], tensor_args=tensor_args)
        )

    server = viser.ViserServer(
        host=cfg.get("viser_host", "0.0.0.0"),
        port=int(cfg.get("viser_port", 8080)),
        label=cfg.get("viser_label", "MPD Interactive Demo"),
    )

    _add_fixed_scene_to_viser(server, scene_spec)
    robot_viser_handle = None
    robot_viser_mode = None
    if cfg.get("show_robot_q_init", True):
        show_robot_collision = bool(cfg.get("show_robot_q_init_collision", False))
        try:
            robot_root, robot_urdf_handle = _add_robot_q_init_urdf_to_viser(
                server,
                planner.planning_task.robot,
                q_init=q_start,
                show_collision=show_robot_collision,
            )
            robot_viser_handle = (robot_root, robot_urdf_handle)
            robot_viser_mode = "urdf"
        except Exception:
            _add_robot_q_init_spheres_to_viser(
                server,
                planner.planning_task.robot,
                q_init=q_start,
                sphere_margin=float(cfg.get("robot_q_init_sphere_margin", 0.0)),
            )
            robot_viser_mode = "spheres"

    status = server.gui.add_text("Status", "Idle", disabled=True)
    ik_time_text = server.gui.add_text("IK total time", "-", disabled=True)
    ik_coarse_time_text = server.gui.add_text("IK coarse time", "-", disabled=True)
    ik_refined_time_text = server.gui.add_text("IK refined time", "-", disabled=True)
    planning_time_text = server.gui.add_text("Planning time", "-", disabled=True)
    total_time_text = server.gui.add_text("IK + planning time", "-", disabled=True)
    dataset_goal_slider = server.gui.add_slider(
        f"{reference_split} ee_goal index",
        min=0,
        max=max(n_reference_samples - 1, 0),
        step=1,
        initial_value=min(int(cfg.get("reference_index", 0)), max(n_reference_samples - 1, 0)),
    )
    dataset_goal_info = server.gui.add_text(
        "dataset goal source",
        f"{reference_split} split ({n_reference_samples} samples)",
        disabled=True,
    )
    q_start_text = server.gui.add_text(
        "q_start",
        np.array2string(q_start.detach().cpu().numpy(), precision=3, separator=", "),
        disabled=True,
    )
    auto_execute = server.gui.add_checkbox("Execute In SAPIEN", bool(cfg.get("run_sapien", True)))
    plan_button = server.gui.add_button("Plan And Execute")

    initial_boxes = _load_extra_boxes(cfg)
    runtime_widgets = [_make_runtime_box_widget(server, idx, box_spec) for idx, box_spec in enumerate(initial_boxes)]

    goal_wxyz, goal_position = _pose_to_viser_wxyz_and_position(ee_pose_goal)
    goal_gizmo = server.scene.add_transform_controls(
        "/goal_pose",
        scale=float(cfg.get("goal_gizmo_scale", 0.22)),
        position=goal_position,
        wxyz=goal_wxyz,
    )
    server.scene.add_frame(
        "/goal_pose/frame",
        axes_length=float(cfg.get("goal_axes_length", 0.16)),
        axes_radius=float(cfg.get("goal_axes_radius", 0.01)),
    )
    server.scene.add_label("/goal_pose/label", "EE Goal", position=(0.0, 0.0, 0.12))

    @dataset_goal_slider.on_update
    def _(_event):
        if n_reference_samples == 0:
            return
        sample = planner.get_reference_sample(
            index=int(dataset_goal_slider.value),
            selection=reference_split,
        )
        goal_pose = _as_planner_ee_goal_pose(sample.ee_goal_pose)
        goal_wxyz_new, goal_position_new = _pose_to_viser_wxyz_and_position(goal_pose)
        goal_gizmo.position = goal_position_new
        goal_gizmo.wxyz = goal_wxyz_new
        _clear_ik_debug_viser(server)
        ik_time_text.value = "-"
        ik_coarse_time_text.value = "-"
        ik_refined_time_text.value = "-"
        planning_time_text.value = "-"
        total_time_text.value = "-"
        status.value = f"Loaded EE goal from {reference_split}[{int(dataset_goal_slider.value)}]"

    persistent_executor = PersistentSapienTrajectoryExecutor(
        viewer_preset=cfg.get("sapien_viewer_preset", "isaac_gym_default")
    )

    lock = threading.Lock()
    busy = {"value": False}
    demo_state = {"q_start": q_start.detach().clone()}

    @plan_button.on_click
    def _(_event):
        with lock:
            if busy["value"]:
                return
            busy["value"] = True
            plan_button.disabled = True

        status.value = "Planning..."
        ik_time_text.value = "-"
        ik_coarse_time_text.value = "-"
        ik_refined_time_text.value = "-"
        planning_time_text.value = "-"
        total_time_text.value = "-"

        ik_time = None
        planning_time = None
        try:
            current_extra_boxes = [box for widget in runtime_widgets if (box := widget.current_box_spec()) is not None]
            _clear_ik_debug_viser(server)
            current_goal = build_ee_pose_goal(
                position=np.asarray(goal_gizmo.position, dtype=np.float32),
                orientation_quat_xyzw=_viser_wxyz_to_xyzw(goal_gizmo.wxyz),
                tensor_args=tensor_args,
            )
            current_goal = _as_planner_ee_goal_pose(current_goal)

            _save_runtime_state(
                os.path.join(results_dir, cfg.get("runtime_state_filename", "interactive_demo_state.yaml")),
                q_start=demo_state["q_start"],
                ee_pose_goal=current_goal,
                extra_boxes=current_extra_boxes,
            )

            planner.update_extra_boxes(current_extra_boxes)

            _synchronize_torch_device(tensor_args["device"])
            ik_t0 = time.perf_counter()
            q_goal_candidates = planner.solve_goal_ik(
                q_start=demo_state["q_start"],
                ee_pose_goal=current_goal,
                batch_size=cfg.get("n_ik_candidates", 64),
                max_iterations=cfg.get("ik_max_iterations", 500),
                lr=cfg.get("ik_lr", 2e-1),
                se3_eps=cfg.get("ik_se3_eps", 5e-2),
                max_candidates=cfg.get("max_goal_candidates", 4),
                debug=cfg.get("debug", False),
            )
            _synchronize_torch_device(tensor_args["device"])
            ik_time = time.perf_counter() - ik_t0
            ik_debug_data = planner.last_ik_debug_data or {}
            ik_time_total = float(ik_debug_data.get("t_ik_total_wall", ik_time))
            ik_time_coarse = float(ik_debug_data.get("t_ik_coarse_wall", ik_time_total))
            ik_time_refined = float(ik_debug_data.get("t_ik_refined_wall", max(ik_time_total - ik_time_coarse, 0.0)))
            ik_time_text.value = _format_seconds(ik_time_total)
            ik_coarse_time_text.value = _format_seconds(ik_time_coarse)
            ik_refined_time_text.value = _format_seconds(ik_time_refined)
            status.value = (
                f"IK finished in {_format_seconds(ik_time_total)} "
                f"(coarse {_format_seconds(ik_time_coarse)}, refined {_format_seconds(ik_time_refined)}). "
                f"Planning trajectory..."
            )

            if q_goal_candidates.numel() == 0:
                _visualize_ik_debug_in_viser(
                    server,
                    planner,
                    max_collision_free_to_render=cfg.get("max_collision_free_to_render", 4),
                    max_colliding_to_render=cfg.get("max_colliding_to_render", 4),
                    max_ik_valid_to_render=cfg.get("max_ik_valid_to_render", 4),
                    max_ik_refined_to_render=cfg.get("max_ik_refined_to_render", 4),
                    sphere_margin=float(cfg.get("robot_q_init_sphere_margin", 0.0)),
                )
                raise RuntimeError("No collision-free IK candidate was found for the requested EE goal pose")

            ik_debug_data = planner.last_ik_debug_data
            _synchronize_torch_device(tensor_args["device"])
            planning_t0 = time.perf_counter()
            try:
                results_single_plan = planner.plan_to_ee_goal(
                    q_start=demo_state["q_start"],
                    ee_pose_goal=current_goal,
                    q_goal_candidates=q_goal_candidates,
                    n_trajectory_samples=100,
                    n_ik_candidates=cfg.get("n_ik_candidates", 64),
                    max_goal_candidates=cfg.get("max_goal_candidates", 4),
                    ik_max_iterations=cfg.get("ik_max_iterations", 500),
                    ik_lr=cfg.get("ik_lr", 2e-1),
                    ik_se3_eps=cfg.get("ik_se3_eps", 5e-2),
                    debug=cfg.get("debug", False),
                )
            finally:
                planner.last_ik_debug_data = ik_debug_data
                _synchronize_torch_device(tensor_args["device"])
                planning_time = time.perf_counter() - planning_t0
                planning_time_text.value = _format_seconds(planning_time)
                total_time_text.value = _format_seconds(ik_time_total + planning_time)
            results_single_plan.t_ik_wall = ik_time_total
            results_single_plan.t_ik_coarse_wall = ik_time_coarse
            results_single_plan.t_ik_refined_wall = ik_time_refined
            results_single_plan.t_planning_wall = planning_time
            results_single_plan.t_ik_and_planning_wall = ik_time_total + planning_time
            print(
                f"IK time: {_format_seconds(ik_time_total)} "
                f"(coarse {_format_seconds(ik_time_coarse)}, refined {_format_seconds(ik_time_refined)}) | "
                f"Planning time: {_format_seconds(planning_time)} | "
                f"IK + planning: {_format_seconds(ik_time_total + planning_time)}"
            )

            torch.save(
                results_single_plan,
                os.path.join(results_dir, cfg.get("plan_filename", "interactive_plan.pt")),
                _use_new_zipfile_serialization=True,
            )

            if results_single_plan.q_trajs_pos_best is None:
                status.value = (
                    "Planning finished, but no valid trajectory was found. "
                    f"IK: {_format_seconds(ik_time_total)} "
                    f"(coarse {_format_seconds(ik_time_coarse)}, refined {_format_seconds(ik_time_refined)}), "
                    f"planning: {_format_seconds(planning_time)}"
                )
                return

            path_length = _safe_metric_scalar(results_single_plan.metrics.get("trajs_best", {}).get("path_length"))
            status.value = (
                f"Plan found. IK: {_format_seconds(ik_time_total)} "
                f"(coarse {_format_seconds(ik_time_coarse)}, refined {_format_seconds(ik_time_refined)}), "
                f"planning: {_format_seconds(planning_time)}"
                f", inference: {results_single_plan.t_inference_total:.2f}s"
                + (f", path length: {path_length:.3f}" if path_length is not None else "")
            )

            demo_state["q_start"] = results_single_plan.q_trajs_pos_best[-1].detach().clone()
            q_start_text.value = np.array2string(
                demo_state["q_start"].detach().cpu().numpy(), precision=3, separator=", "
            )
            if cfg.get("show_robot_q_init", True):
                if robot_viser_mode == "urdf" and robot_viser_handle is not None:
                    _update_robot_q_init_urdf(robot_viser_handle[1], demo_state["q_start"])
                elif robot_viser_mode == "spheres":
                    _refresh_robot_q_init_spheres(
                        server,
                        planner.planning_task.robot,
                        demo_state["q_start"],
                        sphere_margin=float(cfg.get("robot_q_init_sphere_margin", 0.0)),
                    )

            if bool(auto_execute.value):
                status.value = "Executing in SAPIEN..."
                sapien_stats = persistent_executor.execute_trajectory(
                    q_traj=results_single_plan.q_trajs_pos_best,
                    timesteps=results_single_plan.timesteps,
                    scene_spec=build_scene_spec_from_planning_env(planner.planning_task.env),
                    render_viewer=cfg.get("render_sapien_viewer", True),
                    add_ground=cfg.get("sapien_add_ground", False),
                    scene_timestep=cfg.get("sapien_scene_timestep", 1.0 / 240.0),
                    render_every_n_steps=cfg.get("sapien_render_every_n_steps", 4),
                    stiffness=cfg.get("sapien_drive_stiffness", 200.0),
                    damping=cfg.get("sapien_drive_damping", 40.0),
                    force_limit=cfg.get("sapien_force_limit", 1000.0),
                    drive_mode=cfg.get("sapien_drive_mode", "force"),
                    balance_passive_force=cfg.get("sapien_balance_passive_force", True),
                    compensate_gravity=cfg.get("sapien_compensate_gravity", True),
                    compensate_coriolis_and_centrifugal=cfg.get("sapien_compensate_coriolis_and_centrifugal", True),
                    n_pre_steps=cfg.get("sapien_n_pre_steps", 5),
                    n_post_steps=cfg.get("sapien_n_post_steps", 10),
                    robot_cfg=cfg.get("sapien_robot"),
                    viewer_preset=cfg.get("sapien_viewer_preset", "isaac_gym_default"),
                )
                status.value = (
                    f"Execution finished. Final tracking error: "
                    f"{float(sapien_stats['final_tracking_error_l2']):.4f}. "
                    f"IK: {_format_seconds(ik_time_total)} "
                    f"(coarse {_format_seconds(ik_time_coarse)}, refined {_format_seconds(ik_time_refined)}), "
                    f"planning: {_format_seconds(planning_time)}"
                )
        except Exception as exc:
            timing_parts = []
            if ik_time is not None:
                ik_debug_data = planner.last_ik_debug_data or {}
                ik_time_total = float(ik_debug_data.get("t_ik_total_wall", ik_time))
                ik_time_coarse = float(ik_debug_data.get("t_ik_coarse_wall", ik_time_total))
                ik_time_refined = float(ik_debug_data.get("t_ik_refined_wall", max(ik_time_total - ik_time_coarse, 0.0)))
                ik_time_text.value = _format_seconds(ik_time_total)
                ik_coarse_time_text.value = _format_seconds(ik_time_coarse)
                ik_refined_time_text.value = _format_seconds(ik_time_refined)
                timing_parts.append(
                    f"IK: {_format_seconds(ik_time_total)} "
                    f"(coarse {_format_seconds(ik_time_coarse)}, refined {_format_seconds(ik_time_refined)})"
                )
            if planning_time is not None:
                timing_parts.append(f"planning: {_format_seconds(planning_time)}")
            status.value = f"Error: {exc}" + (f" ({', '.join(timing_parts)})" if timing_parts else "")
            if planner.last_ik_debug_data is not None:
                _visualize_ik_debug_in_viser(
                    server,
                    planner,
                    max_collision_free_to_render=cfg.get("max_collision_free_to_render", 4),
                    max_colliding_to_render=cfg.get("max_colliding_to_render", 4),
                    max_ik_valid_to_render=cfg.get("max_ik_valid_to_render", 4),
                    max_ik_refined_to_render=cfg.get("max_ik_refined_to_render", 4),
                    sphere_margin=float(cfg.get("robot_q_init_sphere_margin", 0.0)),
                )
            if cfg.get("save_ik_debug_plot_on_failure", True) and planner.last_ik_debug_data is not None:
                planner.save_last_ik_debug_visualization(
                    save_path=os.path.join(results_dir, cfg.get("ik_debug_plot_filename", "ik_debug.png")),
                    show=cfg.get("show_ik_debug_plot", False),
                    max_collision_free_to_render=cfg.get("max_collision_free_to_render", 4),
                    max_colliding_to_render=cfg.get("max_colliding_to_render", 4),
                    draw_collision_spheres=cfg.get("draw_collision_spheres", False),
                )
        finally:
            with lock:
                busy["value"] = False
                plan_button.disabled = False

    print("\n----------------INTERACTIVE VISER DEMO----------------")
    print(f"cfg: {cfg_path}")
    print(f"results_dir: {results_dir}")
    print(f"viser: http://{cfg.get('viser_host', '127.0.0.1')}:{int(cfg.get('viser_port', 8080))}")
    print("Move the EE goal gizmo and runtime box gizmos in the browser, then click 'Plan And Execute'.")

    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        planner.cleanup()
        persistent_executor.close()


if __name__ == "__main__":
    main()
