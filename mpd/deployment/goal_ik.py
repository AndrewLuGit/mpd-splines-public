import time
from collections.abc import Mapping

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.spatial.transform import Rotation

from torch_robotics.torch_kinematics_tree.geometrics.frame import Frame
from torch_robotics.torch_kinematics_tree.geometrics.utils import SE3_distance
from torch_robotics.torch_utils.torch_utils import DEFAULT_TENSOR_ARGS, to_torch
from torch_robotics.visualizers.plot_utils import create_fig_and_axes, plot_coordinate_frame


def _synchronize_tensor_device(tensor_args):
    device = tensor_args.get("device")
    if device is None:
        return
    device = torch.device(device)
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)
    elif device.type == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "synchronize"):
        torch.mps.synchronize()


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


def _refine_panda_goal_ik_with_collision(
    planning_task,
    q_candidates,
    q_start,
    ee_pose_goal,
    max_iterations=80,
    lr=5e-2,
    se3_weight=1.0,
    collision_weight=20.0,
    q_start_weight=0.05,
    collision_margin=0.03,
    debug=False,
):
    if q_candidates.numel() == 0 or max_iterations <= 0:
        return q_candidates

    diff_panda = planning_task.robot.diff_panda
    link_name = planning_task.robot.link_name_ee
    lower, upper, _, _ = diff_panda.get_joint_limit_array()
    lower = to_torch(lower, **planning_task.tensor_args)
    upper = to_torch(upper, **planning_task.tensor_args)

    q_refined = q_candidates.detach().clone().requires_grad_(True)
    optimizer = torch.optim.LBFGS(
        [q_refined],
        lr=lr,
        max_iter=max_iterations,
        max_eval=max_iterations * 2,
        history_size=10,
        tolerance_grad=1e-7,
        tolerance_change=1e-9,
        line_search_fn="strong_wolfe",
    )
    q_start_expanded = q_start.unsqueeze(0).expand_as(q_refined)
    ee_pose_goal_expanded = ee_pose_goal.unsqueeze(0).expand(q_refined.shape[0], -1, -1)

    def _compute_metrics(q_eval):
        H = diff_panda.compute_forward_kinematics_all_links(q_eval, link_list=[link_name]).squeeze(1)
        err_se3 = SE3_distance(H, ee_pose_goal_expanded, w_pos=1.0, w_rot=1.0)
        collision_cost = planning_task.compute_collision_cost(q_eval, margin=collision_margin).reshape(-1)
        err_q_start = torch.linalg.norm(q_eval - q_start_expanded, dim=-1)
        return err_se3, collision_cost, err_q_start

    debug_state = {"n_closure_calls": 0}

    def closure():
        optimizer.zero_grad()
        q_eval = torch.clamp(q_refined, lower, upper)
        err_se3, collision_cost, err_q_start = _compute_metrics(q_eval)
        loss_per_q = se3_weight * err_se3 + collision_weight * collision_cost + q_start_weight * err_q_start
        loss = loss_per_q.sum()
        loss.backward()
        debug_state["n_closure_calls"] += 1
        if debug and debug_state["n_closure_calls"] == 1:
            print(
                "IK collision refine "
                "iter=0 "
                f"se3_mean={float(err_se3.mean()):.4f} "
                f"coll_mean={float(collision_cost.mean()):.4f} "
                f"qstart_mean={float(err_q_start.mean()):.4f}"
            )
        return loss

    optimizer.step(closure)
    with torch.no_grad():
        q_refined.clamp_(lower, upper)

    if debug:
        with torch.no_grad():
            err_se3, collision_cost, err_q_start = _compute_metrics(q_refined)
        print(
            "IK collision refine "
            f"iter={max(0, debug_state['n_closure_calls'] - 1)} "
            f"se3_mean={float(err_se3.mean()):.4f} "
            f"coll_mean={float(collision_cost.mean()):.4f} "
            f"qstart_mean={float(err_q_start.mean()):.4f}"
        )

    return q_refined.detach()


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
    collision_refine_max_iterations=80,
    collision_refine_lr=5e-2,
    collision_refine_se3_weight=1.0,
    collision_refine_weight=20.0,
    collision_refine_q_start_weight=0.05,
    collision_refine_margin=0.03,
    debug=False,
    return_debug_data=False,
):
    q_start = to_torch(q_start, **planning_task.tensor_args)
    ee_pose_goal = _ensure_homogeneous_pose_matrix(ee_pose_goal, tensor_args=planning_task.tensor_args)
    diff_panda = planning_task.robot.diff_panda
    link_name = planning_task.robot.link_name_ee
    coarse_ik_time = 0.0
    fine_ik_time = 0.0
    collision_refine_time = 0.0

    use_coarse_to_fine = batch_size > max_candidates and max_iterations > 40
    q_solutions = None
    idx_valid = None
    if use_coarse_to_fine:
        coarse_iterations = min(16, max_iterations - 4)
        coarse_iterations = max(coarse_iterations, 8)
        fine_iterations = max(max_iterations - coarse_iterations, 1)
        coarse_se3_eps = max(se3_eps * 1.5, 0.075)
        coarse_topk = min(batch_size, max(max_candidates * 4, 24))

        lower, upper, _, _ = diff_panda.get_joint_limit_array()
        lower = to_torch(lower, **planning_task.tensor_args)
        upper = to_torch(upper, **planning_task.tensor_args)
        q0_coarse = lower + torch.rand(batch_size, planning_task.robot.q_dim, **planning_task.tensor_args) * (upper - lower)
        q0_coarse[0] = torch.clamp(q_start, lower, upper)

        _synchronize_tensor_device(planning_task.tensor_args)
        coarse_t0 = time.perf_counter()
        q_solutions_coarse, idx_valid_coarse = diff_panda.inverse_kinematics(
            ee_pose_goal,
            link_name=link_name,
            batch_size=batch_size,
            max_iters=coarse_iterations,
            lr=lr,
            se3_eps=coarse_se3_eps,
            q0=q0_coarse,
            q0_noise=0.0,
            eps_joint_lim=torch.pi / 64,
            print_freq=-1,
            debug=debug,
        )
        _synchronize_tensor_device(planning_task.tensor_args)
        coarse_ik_time = time.perf_counter() - coarse_t0

        H_coarse = diff_panda.compute_forward_kinematics_all_links(
            q_solutions_coarse,
            link_list=[link_name],
        ).squeeze(1)
        ee_pose_goal_expanded = ee_pose_goal.unsqueeze(0).expand(q_solutions_coarse.shape[0], -1, -1)
        coarse_se3_error = SE3_distance(H_coarse, ee_pose_goal_expanded, w_pos=1.0, w_rot=1.0)
        coarse_ranked_indices = torch.argsort(coarse_se3_error)

        idx_valid_coarse = torch.atleast_1d(idx_valid_coarse)
        idx_valid_coarse = idx_valid_coarse.reshape(-1)
        valid_coarse_mask = torch.zeros(batch_size, dtype=torch.bool, device=planning_task.tensor_args["device"])
        if idx_valid_coarse.numel() > 0:
            valid_coarse_mask[idx_valid_coarse] = True
            coarse_ranked_indices = torch.cat(
                [
                    coarse_ranked_indices[valid_coarse_mask[coarse_ranked_indices]],
                    coarse_ranked_indices[~valid_coarse_mask[coarse_ranked_indices]],
                ]
            )
        coarse_seed_indices = coarse_ranked_indices[:coarse_topk]
        q0_fine = q_solutions_coarse[coarse_seed_indices]

        _synchronize_tensor_device(planning_task.tensor_args)
        fine_t0 = time.perf_counter()
        q_solutions, idx_valid = diff_panda.inverse_kinematics(
            ee_pose_goal,
            link_name=link_name,
            batch_size=q0_fine.shape[0],
            max_iters=fine_iterations,
            lr=lr,
            se3_eps=se3_eps,
            q0=q0_fine,
            q0_noise=0.0,
            eps_joint_lim=torch.pi / 64,
            print_freq=-1,
            debug=debug,
        )
        _synchronize_tensor_device(planning_task.tensor_args)
        fine_ik_time = time.perf_counter() - fine_t0
    else:
        _synchronize_tensor_device(planning_task.tensor_args)
        coarse_t0 = time.perf_counter()
        q_solutions, idx_valid = diff_panda.inverse_kinematics(
            ee_pose_goal,
            link_name=link_name,
            batch_size=batch_size,
            max_iters=max_iterations,
            lr=lr,
            se3_eps=se3_eps,
            q0=None,
            q0_noise=q0_noise,
            eps_joint_lim=torch.pi / 64,
            print_freq=-1,
            debug=debug,
        )
        _synchronize_tensor_device(planning_task.tensor_args)
        coarse_ik_time = time.perf_counter() - coarse_t0

    if idx_valid.ndim == 0:
        idx_valid = idx_valid.unsqueeze(0)
    idx_valid = torch.atleast_1d(idx_valid)

    debug_data = dict(
        q_start=q_start,
        ee_pose_goal=ee_pose_goal,
        q_solutions_all=q_solutions.detach().clone(),
        idx_valid_ik=idx_valid.detach().clone(),
        t_ik_coarse_wall=coarse_ik_time,
        t_ik_refined_wall=fine_ik_time,
        t_ik_collision_refine_wall=collision_refine_time,
        t_ik_total_wall=coarse_ik_time + fine_ik_time + collision_refine_time,
        q_candidates_ik_valid=torch.empty((0, planning_task.robot.q_dim), **planning_task.tensor_args),
        q_candidates_ik_refined=torch.empty((0, planning_task.robot.q_dim), **planning_task.tensor_args),
        refined_se3_error=torch.empty((0,), **planning_task.tensor_args),
        idx_valid_refined_se3=torch.empty((0,), dtype=torch.long, device=planning_task.tensor_args["device"]),
        in_collision_ik_valid=torch.empty((0,), dtype=torch.bool, device=planning_task.tensor_args["device"]),
        in_collision_ik_refined=torch.empty((0,), dtype=torch.bool, device=planning_task.tensor_args["device"]),
        collision_cost_ik_refined=torch.empty((0,), **planning_task.tensor_args),
        collision_cost_collision_free=torch.empty((0,), **planning_task.tensor_args),
        refined_se3_error_collision_free=torch.empty((0,), **planning_task.tensor_args),
        q_start_distance_collision_free=torch.empty((0,), **planning_task.tensor_args),
        ranking_score_collision_free=torch.empty((0,), **planning_task.tensor_args),
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
    _synchronize_tensor_device(planning_task.tensor_args)
    collision_refine_t0 = time.perf_counter()
    q_candidates_refined = _refine_panda_goal_ik_with_collision(
        planning_task,
        q_candidates=q_candidates,
        q_start=q_start,
        ee_pose_goal=ee_pose_goal,
        max_iterations=collision_refine_max_iterations,
        lr=collision_refine_lr,
        se3_weight=collision_refine_se3_weight,
        collision_weight=collision_refine_weight,
        q_start_weight=collision_refine_q_start_weight,
        collision_margin=collision_refine_margin,
        debug=debug,
    )
    _synchronize_tensor_device(planning_task.tensor_args)
    collision_refine_time = time.perf_counter() - collision_refine_t0
    debug_data["t_ik_collision_refine_wall"] = collision_refine_time
    debug_data["t_ik_refined_wall"] = fine_ik_time + collision_refine_time
    debug_data["t_ik_total_wall"] = coarse_ik_time + fine_ik_time + collision_refine_time
    debug_data["q_candidates_ik_refined"] = q_candidates_refined.detach().clone()

    H_refined = planning_task.robot.diff_panda.compute_forward_kinematics_all_links(
        q_candidates_refined,
        link_list=[planning_task.robot.link_name_ee],
    ).squeeze(1)
    ee_pose_goal_expanded = ee_pose_goal.unsqueeze(0).expand(q_candidates_refined.shape[0], -1, -1)
    refined_se3_error = SE3_distance(H_refined, ee_pose_goal_expanded, w_pos=1.0, w_rot=1.0)
    debug_data["refined_se3_error"] = refined_se3_error.detach().clone()
    idx_valid_refined_se3 = torch.argwhere(refined_se3_error < se3_eps).reshape(-1)
    debug_data["idx_valid_refined_se3"] = idx_valid_refined_se3.detach().clone()
    debug_data["collision_cost_ik_refined"] = planning_task.compute_collision_cost(
        q_candidates_refined,
        margin=collision_refine_margin,
    ).reshape(-1).detach().clone()

    if idx_valid_refined_se3.numel() == 0:
        empty = torch.empty((0, planning_task.robot.q_dim), **planning_task.tensor_args)
        if return_debug_data:
            return empty, debug_data
        return empty

    q_candidates = q_candidates_refined[idx_valid_refined_se3]
    refined_se3_error = refined_se3_error[idx_valid_refined_se3]
    in_collision = planning_task.compute_collision(q_candidates, margin=0.0).bool().reshape(-1)
    collision_cost = planning_task.compute_collision_cost(q_candidates, margin=collision_refine_margin).reshape(-1)
    debug_data["in_collision_ik_valid"] = planning_task.compute_collision(
        debug_data["q_candidates_ik_valid"], margin=0.0
    ).bool().reshape(-1).detach().clone()
    debug_data["in_collision_ik_refined"] = in_collision.detach().clone()
    debug_data["q_candidates_colliding"] = q_candidates[in_collision].detach().clone()
    q_candidates = q_candidates[~in_collision]
    collision_cost = collision_cost[~in_collision]
    refined_se3_error = refined_se3_error[~in_collision]
    debug_data["q_candidates_collision_free"] = q_candidates.detach().clone()
    debug_data["collision_cost_collision_free"] = collision_cost.detach().clone()
    debug_data["refined_se3_error_collision_free"] = refined_se3_error.detach().clone()
    if q_candidates.numel() == 0:
        empty = torch.empty((0, planning_task.robot.q_dim), **planning_task.tensor_args)
        if return_debug_data:
            return empty, debug_data
        return empty

    distances = planning_task.robot.distance_q(q_candidates, q_start)
    debug_data["q_start_distance_collision_free"] = distances.detach().clone()

    def _normalize_for_ranking(values):
        values = values.reshape(-1)
        if values.numel() <= 1:
            return torch.zeros_like(values)
        min_val = values.min()
        max_val = values.max()
        denom = torch.clamp(max_val - min_val, min=1e-8)
        return (values - min_val) / denom

    ranking_score = (
        1e6 * _normalize_for_ranking(collision_cost)
        + 1e3 * _normalize_for_ranking(refined_se3_error)
        + _normalize_for_ranking(distances)
    )
    debug_data["ranking_score_collision_free"] = ranking_score.detach().clone()
    sorted_indices = torch.argsort(ranking_score)
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
