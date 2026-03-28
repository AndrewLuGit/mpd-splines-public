from collections.abc import Mapping

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.spatial.transform import Rotation

from torch_robotics.torch_kinematics_tree.geometrics.frame import Frame
from torch_robotics.torch_utils.torch_utils import DEFAULT_TENSOR_ARGS, to_torch
from torch_robotics.visualizers.plot_utils import create_fig_and_axes, plot_coordinate_frame


def _ensure_homogeneous_pose_matrix(ee_pose_goal, tensor_args=DEFAULT_TENSOR_ARGS):
    ee_pose_goal = to_torch(ee_pose_goal, **tensor_args)

    if ee_pose_goal.shape[-2:] == (4, 4):
        return ee_pose_goal

    if ee_pose_goal.shape[-2:] == (3, 4):
        output_shape = ee_pose_goal.shape[:-2] + (4, 4)
        pose_h = torch.zeros(output_shape, **tensor_args)
        pose_h[..., :3, :4] = ee_pose_goal
        pose_h[..., 3, 3] = 1.0
        return pose_h

    raise ValueError(f"Expected ee_pose_goal with shape (..., 3, 4) or (..., 4, 4), got {ee_pose_goal.shape}")


def build_ee_pose_goal(
    position,
    orientation_matrix=None,
    orientation_quat_xyzw=None,
    orientation_euler_xyz_deg=None,
    tensor_args=DEFAULT_TENSOR_ARGS,
):
    orientation_specs = [
        orientation_matrix is not None,
        orientation_quat_xyzw is not None,
        orientation_euler_xyz_deg is not None,
    ]
    if sum(orientation_specs) > 1:
        raise ValueError("Provide only one orientation representation")

    pose = np.eye(4, dtype=float)
    pose[:3, 3] = np.asarray(position, dtype=float)

    if orientation_matrix is not None:
        rotation_matrix = np.asarray(orientation_matrix, dtype=float)
    elif orientation_quat_xyzw is not None:
        rotation_matrix = Rotation.from_quat(np.asarray(orientation_quat_xyzw, dtype=float)).as_matrix()
    elif orientation_euler_xyz_deg is not None:
        rotation_matrix = Rotation.from_euler("xyz", np.asarray(orientation_euler_xyz_deg, dtype=float), degrees=True).as_matrix()
    else:
        rotation_matrix = np.eye(3, dtype=float)

    if rotation_matrix.shape != (3, 3):
        raise ValueError(f"orientation must define a 3x3 rotation matrix, got {rotation_matrix.shape}")

    pose[:3, :3] = rotation_matrix
    return to_torch(pose, **tensor_args)


def build_ee_pose_goal_from_dict(ee_goal_pose_cfg, tensor_args=DEFAULT_TENSOR_ARGS):
    if ee_goal_pose_cfg is None:
        return None
    if not isinstance(ee_goal_pose_cfg, Mapping):
        raise TypeError("ee_goal_pose config must be a mapping")

    if "matrix" in ee_goal_pose_cfg:
        matrix = np.asarray(ee_goal_pose_cfg["matrix"], dtype=float)
        if matrix.shape == (3, 4):
            pose = np.eye(4, dtype=float)
            pose[:3, :] = matrix
        elif matrix.shape == (4, 4):
            pose = matrix
        else:
            raise ValueError(f"ee_goal_pose.matrix must have shape (3, 4) or (4, 4), got {matrix.shape}")
        return to_torch(pose, **tensor_args)

    return build_ee_pose_goal(
        position=ee_goal_pose_cfg["position"],
        orientation_matrix=ee_goal_pose_cfg.get("orientation_matrix"),
        orientation_quat_xyzw=ee_goal_pose_cfg.get("orientation_quat_xyzw"),
        orientation_euler_xyz_deg=ee_goal_pose_cfg.get("orientation_euler_xyz_deg"),
        tensor_args=tensor_args,
    )


def solve_panda_goal_ik(
    planning_task,
    q_start,
    ee_pose_goal,
    batch_size=32,
    max_iterations=500,
    lr=2e-1,
    se3_eps=5e-2,
    q0_noise=torch.pi / 8,
    max_candidates=8,
    debug=False,
    return_debug_data=False,
):
    q_start = to_torch(q_start, **planning_task.tensor_args)
    ee_pose_goal = _ensure_homogeneous_pose_matrix(ee_pose_goal, tensor_args=planning_task.tensor_args)

    q0 = q_start.unsqueeze(0).repeat(batch_size, 1)
    q_solutions, idx_valid = planning_task.robot.diff_panda.inverse_kinematics(
        ee_pose_goal,
        link_name=planning_task.robot.link_name_ee,
        batch_size=batch_size,
        max_iters=max_iterations,
        lr=lr,
        se3_eps=se3_eps,
        q0=q0,
        q0_noise=q0_noise,
        eps_joint_lim=torch.pi / 64,
        print_freq=-1,
        debug=debug,
    )

    if idx_valid.ndim == 0:
        idx_valid = idx_valid.unsqueeze(0)
    idx_valid = torch.atleast_1d(idx_valid)

    debug_data = dict(
        q_start=q_start,
        ee_pose_goal=ee_pose_goal,
        q_solutions_all=q_solutions.detach().clone(),
        idx_valid_ik=idx_valid.detach().clone(),
        q_candidates_ik_valid=torch.empty((0, planning_task.robot.q_dim), **planning_task.tensor_args),
        in_collision_ik_valid=torch.empty((0,), dtype=torch.bool, device=planning_task.tensor_args["device"]),
        q_candidates_collision_free=torch.empty((0, planning_task.robot.q_dim), **planning_task.tensor_args),
        q_candidates_colliding=torch.empty((0, planning_task.robot.q_dim), **planning_task.tensor_args),
        q_candidates_selected=torch.empty((0, planning_task.robot.q_dim), **planning_task.tensor_args),
    )

    if idx_valid.numel() == 0:
        empty = torch.empty((0, planning_task.robot.q_dim), **planning_task.tensor_args)
        if return_debug_data:
            return empty, debug_data
        return empty

    q_candidates = q_solutions[idx_valid]
    debug_data["q_candidates_ik_valid"] = q_candidates.detach().clone()
    in_collision = planning_task.compute_collision(q_candidates).bool().reshape(-1)
    debug_data["in_collision_ik_valid"] = in_collision.detach().clone()
    debug_data["q_candidates_colliding"] = q_candidates[in_collision].detach().clone()
    q_candidates = q_candidates[~in_collision]
    debug_data["q_candidates_collision_free"] = q_candidates.detach().clone()
    if q_candidates.numel() == 0:
        empty = torch.empty((0, planning_task.robot.q_dim), **planning_task.tensor_args)
        if return_debug_data:
            return empty, debug_data
        return empty

    distances = planning_task.robot.distance_q(q_candidates, q_start)
    sorted_indices = torch.argsort(distances)
    q_candidates_selected = q_candidates[sorted_indices[:max_candidates]]
    debug_data["q_candidates_selected"] = q_candidates_selected.detach().clone()

    if return_debug_data:
        return q_candidates_selected, debug_data
    return q_candidates_selected


def render_panda_goal_ik_debug(
    planning_task,
    ik_debug_data,
    save_path=None,
    show=False,
    max_collision_free_to_render=4,
    max_colliding_to_render=4,
    draw_collision_spheres=False,
):
    fig, ax = create_fig_and_axes(dim=planning_task.env.dim)
    planning_task.env.render(ax)

    q_start = ik_debug_data["q_start"]
    ee_pose_goal = ik_debug_data["ee_pose_goal"]
    q_candidates_collision_free = ik_debug_data["q_candidates_collision_free"]
    q_candidates_colliding = ik_debug_data["q_candidates_colliding"]
    idx_valid_ik = ik_debug_data["idx_valid_ik"]
    q_solutions_all = ik_debug_data["q_solutions_all"]

    planning_task.robot.render(
        ax,
        q=q_start,
        color="green",
        draw_links_spheres=draw_collision_spheres,
    )

    for q in q_candidates_colliding[:max_colliding_to_render]:
        planning_task.robot.render(
            ax,
            q=q,
            color="red",
            draw_links_spheres=draw_collision_spheres,
        )

    for q in q_candidates_collision_free[:max_collision_free_to_render]:
        planning_task.robot.render(
            ax,
            q=q,
            color="orange",
            draw_links_spheres=draw_collision_spheres,
        )

    # Draw the target last so it is not hidden behind the robots.
    frame_target = Frame(rot=ee_pose_goal[:3, :3], trans=ee_pose_goal[:3, 3], device=planning_task.tensor_args["device"])
    plot_coordinate_frame(
        ax,
        frame_target,
        arrow_length=0.20,
        arrow_linewidth=3.0,
        tensor_args=planning_task.tensor_args,
    )

    target_pos = ee_pose_goal[:3, 3].detach().cpu().numpy()
    ax.scatter(
        target_pos[0],
        target_pos[1],
        target_pos[2],
        color="black",
        marker="x",
        s=180,
        linewidths=3,
        depthshade=False,
    )
    ax.scatter(
        target_pos[0],
        target_pos[1],
        target_pos[2],
        color="magenta",
        marker="*",
        s=260,
        edgecolors="black",
        linewidths=1.5,
        depthshade=False,
    )
    ax.text(
        target_pos[0],
        target_pos[1],
        target_pos[2] + 0.06,
        "EE goal",
        color="magenta",
        fontsize=10,
        horizontalalignment="center",
    )

    n_total = int(q_solutions_all.shape[0])
    n_valid_ik = int(idx_valid_ik.numel())
    n_collision_free = int(q_candidates_collision_free.shape[0])
    n_colliding = int(q_candidates_colliding.shape[0])
    ax.set_title(
        "IK Debug\n"
        f"sampled={n_total}, ik_valid={n_valid_ik}, collision_free={n_collision_free}, colliding={n_colliding}"
    )

    if save_path is not None:
        fig.savefig(save_path, dpi=160, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)
