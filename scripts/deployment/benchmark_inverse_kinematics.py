import os
import sys
import types

REPO_PATH_BOOTSTRAP = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_PATH_BOOTSTRAP not in sys.path:
    sys.path.insert(0, REPO_PATH_BOOTSTRAP)
TORCH_ROBOTICS_PATH = os.path.join(REPO_PATH_BOOTSTRAP, "mpd", "torch_robotics")
if TORCH_ROBOTICS_PATH not in sys.path:
    sys.path.insert(0, TORCH_ROBOTICS_PATH)

try:
    import wandb  # noqa: F401
except ModuleNotFoundError:
    sys.modules["wandb"] = types.ModuleType("wandb")

from mpd.utils.patches import numpy_monkey_patch

numpy_monkey_patch()

import argparse
import csv
import json
import math
import time
from dataclasses import dataclass

import numpy as np
import torch
import yaml
from dotmap import DotMap

from mpd.deployment.goal_ik import (
    _ensure_homogeneous_pose_matrix,
    _refine_panda_goal_ik_with_collision,
    _synchronize_tensor_device,
)
from mpd.deployment.scene_primitives import build_object_fields_from_boxes
from mpd.paths import REPO_PATH
from mpd.utils.loaders import get_planning_task_and_dataset, load_params_from_yaml, save_to_yaml
from torch_robotics.torch_kinematics_tree.geometrics.utils import SE3_distance
from torch_robotics.torch_utils.seed import fix_random_seed
from torch_robotics.torch_utils.torch_utils import get_torch_device, to_torch


ORIGINAL_TORCH_ROBOTICS_COMMIT = "4cefb9f5cabf2940ba50b7258df4d045fd700012"


@dataclass(frozen=True)
class MethodSpec:
    name: str
    ik_backend: str
    refine_backend: str | None
    two_stage: bool


METHODS = [
    MethodSpec("current_lm_lbfgs", "lm", "lbfgs", True),
    MethodSpec("original_adam", "adam", None, False),
    # MethodSpec("two_stage_adam_lbfgs", "adam", "lbfgs", True),
    MethodSpec("lm_adam", "lm", "adam", True),
    # MethodSpec("adam_adam", "adam", "adam", True),
]


def _resolve_repo_path(path):
    expanded = os.path.expandvars(path)
    if os.path.isabs(expanded):
        return expanded
    return os.path.join(REPO_PATH, expanded)


def _resolve_model_dir(model_dir):
    expanded = os.path.expandvars(model_dir)
    if os.path.exists(expanded):
        return expanded

    marker = "data_trained_models"
    if marker in expanded:
        suffix = expanded.split(f"{marker}/", 1)[1]
        candidate = os.path.join(REPO_PATH, marker, suffix)
        if os.path.exists(candidate):
            return candidate

    raise FileNotFoundError(f"Could not resolve model directory: {model_dir}")


def _select_model_dir(args_inference):
    if "cvae" in args_inference.planner_alg:
        if args_inference.model_selection == "bspline":
            return args_inference.model_dir_cvae_bspline
        if args_inference.model_selection == "waypoints":
            return args_inference.model_dir_cvae_waypoints
    else:
        if args_inference.model_selection == "bspline":
            return args_inference.model_dir_ddpm_bspline
        if args_inference.model_selection == "waypoints":
            return args_inference.model_dir_ddpm_waypoints
    raise NotImplementedError("Unsupported planner/model selection combination")


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


def _original_torch_robotics_adam_ik(
    diff_panda,
    H_target,
    link_name="ee_link",
    batch_size=1,
    max_iters=1000,
    lr=1e-2,
    se3_eps=1e-1,
    q0=None,
    q0_noise=torch.pi / 8,
    eps_joint_lim=torch.pi / 100,
    print_freq=-1,
    debug=False,
):
    """Adam IK from torch_robotics commit 4cefb9f5.../robot_tree.py."""
    if H_target.ndim == 2:
        H_target = H_target.unsqueeze(0)

    lower, upper, _, _ = diff_panda.get_joint_limit_array()
    lower += eps_joint_lim
    upper -= eps_joint_lim
    lower = to_torch(lower, device=diff_panda._device)
    upper = to_torch(upper, device=diff_panda._device)
    if H_target.shape[0] == 1 and batch_size != 1:
        H_target = H_target.expand(batch_size, -1, -1)
    elif H_target.shape[0] != batch_size:
        raise ValueError(f"H_target batch size ({H_target.shape[0]}) must be 1 or match batch_size ({batch_size})")

    if q0 is None:
        q0 = torch.rand(batch_size, diff_panda._n_dofs, device=diff_panda._device, dtype=lower.dtype)
        q0 = lower + q0 * (upper - lower)
    else:
        q0 = q0.to(device=diff_panda._device, dtype=lower.dtype).clone()
        q0 += torch.randn(batch_size, diff_panda._n_dofs, device=diff_panda._device, dtype=lower.dtype) * q0_noise
        q0 = torch.clamp(q0, lower, upper)
        assert q0.shape[0] == batch_size
        assert q0.shape[1] == diff_panda._n_dofs

    q = q0.clone().requires_grad_(True)
    optimizer = torch.optim.Adam([q], lr=lr)
    idx_valid = torch.empty((0,), dtype=torch.long, device=diff_panda._device)
    err_per_q = torch.full((batch_size,), float("inf"), device=diff_panda._device, dtype=q.dtype)
    for i in range(max_iters):
        optimizer.zero_grad()
        idx_valid = diff_panda.ik_termination(q, H_target, link_name, lower, upper, se3_eps=se3_eps, debug=debug)
        if idx_valid.nelement() == batch_size:
            if print_freq != -1:
                print(f"\nIK converged for all joint configurations in {i} iterations")
            break

        err_per_q = diff_panda.loss_fn_ik_per_q(
            q,
            H_target,
            link_name,
            w_se3=1.0,
            w_joint_limits=300.0,
            lower=lower,
            upper=upper,
            w_q_rest=1.0,
            q_rest=None,
            debug=debug,
        )
        if (i == 0 or (i % print_freq) == 0) and print_freq != -1:
            print(f"\n---> Iter {i}/{max_iters}")
            print(f"Error mean, std: {err_per_q.mean():.3f}, {err_per_q.std():.3f}")
            print(f"idx_valid: {len(idx_valid)}/{batch_size}")

        err_per_q.sum().backward()
        optimizer.step()

    if print_freq != -1 and i == max_iters - 1 and idx_valid.nelement() != batch_size:
        print("\nIK did not converge for all joint configurations!")
        print(f"Error mean, std: {err_per_q.mean():.3f}, {err_per_q.std():.3f}")
        print(f"idx_valid: {len(idx_valid)}/{batch_size}")

    return q.detach(), idx_valid


def _run_pose_ik_backend(
    planning_task,
    ee_pose_goal,
    q0,
    batch_size,
    max_iterations,
    lr,
    se3_eps,
    backend,
    q0_noise,
    debug=False,
):
    diff_panda = planning_task.robot.diff_panda
    kwargs = dict(
        H_target=ee_pose_goal,
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
    if backend == "lm":
        return diff_panda.inverse_kinematics(**kwargs)
    if backend == "adam":
        return _original_torch_robotics_adam_ik(diff_panda, **kwargs)
    raise ValueError(f"Unknown IK backend: {backend}")


def _sample_uniform_joint_configs(planning_task, batch_size, q_start):
    diff_panda = planning_task.robot.diff_panda
    lower, upper, _, _ = diff_panda.get_joint_limit_array()
    lower = to_torch(lower, **planning_task.tensor_args)
    upper = to_torch(upper, **planning_task.tensor_args)
    q0 = lower + torch.rand(batch_size, planning_task.robot.q_dim, **planning_task.tensor_args) * (upper - lower)
    q0[0] = torch.clamp(q_start, lower, upper)
    return q0


def _select_fine_seeds(planning_task, ee_pose_goal, q_solutions, idx_valid, n_seeds):
    link_name = planning_task.robot.link_name_ee
    H = planning_task.robot.diff_panda.compute_forward_kinematics_all_links(q_solutions, link_list=[link_name]).squeeze(
        1
    )
    target = ee_pose_goal.unsqueeze(0).expand(q_solutions.shape[0], -1, -1)
    se3_error = SE3_distance(H, target, w_pos=1.0, w_rot=1.0)
    ranked_indices = torch.argsort(se3_error)

    idx_valid = torch.atleast_1d(idx_valid).reshape(-1)
    if idx_valid.numel() > 0:
        valid_mask = torch.zeros(q_solutions.shape[0], dtype=torch.bool, device=planning_task.tensor_args["device"])
        valid_mask[idx_valid] = True
        ranked_indices = torch.cat(
            [ranked_indices[valid_mask[ranked_indices]], ranked_indices[~valid_mask[ranked_indices]]]
        )

    return q_solutions[ranked_indices[:n_seeds]]


def _run_two_stage_pose_ik(
    planning_task,
    q_start,
    ee_pose_goal,
    backend,
    batch_size,
    max_iterations,
    lr,
    se3_eps,
    max_candidates,
    debug=False,
):
    use_coarse_to_fine = batch_size > max_candidates and max_iterations > 40
    if not use_coarse_to_fine:
        return _run_pose_ik_backend(
            planning_task,
            ee_pose_goal,
            q0=None,
            batch_size=batch_size,
            max_iterations=max_iterations,
            lr=lr,
            se3_eps=se3_eps,
            backend=backend,
            q0_noise=torch.pi / 8,
            debug=debug,
        )

    coarse_iterations = min(16, max_iterations - 4)
    coarse_iterations = max(coarse_iterations, 8)
    fine_iterations = max(max_iterations - coarse_iterations, 1)
    coarse_se3_eps = max(se3_eps * 1.5, 0.075)
    coarse_topk = min(batch_size, max(max_candidates * 4, 24))
    q0_coarse = _sample_uniform_joint_configs(planning_task, batch_size, q_start)

    q_solutions_coarse, idx_valid_coarse = _run_pose_ik_backend(
        planning_task,
        ee_pose_goal,
        q0=q0_coarse,
        batch_size=batch_size,
        max_iterations=coarse_iterations,
        lr=lr,
        se3_eps=coarse_se3_eps,
        backend=backend,
        q0_noise=0.0,
        debug=debug,
    )
    q0_fine = _select_fine_seeds(planning_task, ee_pose_goal, q_solutions_coarse, idx_valid_coarse, coarse_topk)
    return _run_pose_ik_backend(
        planning_task,
        ee_pose_goal,
        q0=q0_fine,
        batch_size=q0_fine.shape[0],
        max_iterations=fine_iterations,
        lr=lr,
        se3_eps=se3_eps,
        backend=backend,
        q0_noise=0.0,
        debug=debug,
    )


def _refine_panda_goal_ik_with_collision_adam(
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
):
    if q_candidates.numel() == 0 or max_iterations <= 0:
        return q_candidates

    diff_panda = planning_task.robot.diff_panda
    link_name = planning_task.robot.link_name_ee
    lower, upper, _, _ = diff_panda.get_joint_limit_array()
    lower = to_torch(lower, **planning_task.tensor_args)
    upper = to_torch(upper, **planning_task.tensor_args)
    q_refined = q_candidates.detach().clone().requires_grad_(True)
    optimizer = torch.optim.Adam([q_refined], lr=lr)
    q_start_expanded = q_start.unsqueeze(0).expand_as(q_refined)
    ee_pose_goal_expanded = ee_pose_goal.unsqueeze(0).expand(q_refined.shape[0], -1, -1)

    for _ in range(max_iterations):
        optimizer.zero_grad()
        q_eval = torch.clamp(q_refined, lower, upper)
        H = diff_panda.compute_forward_kinematics_all_links(q_eval, link_list=[link_name]).squeeze(1)
        err_se3 = SE3_distance(H, ee_pose_goal_expanded, w_pos=1.0, w_rot=1.0)
        collision_cost = planning_task.compute_collision_cost(q_eval, margin=collision_margin).reshape(-1)
        err_q_start = torch.linalg.norm(q_eval - q_start_expanded, dim=-1)
        loss = (se3_weight * err_se3 + collision_weight * collision_cost + q_start_weight * err_q_start).sum()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            q_refined.clamp_(lower, upper)

    return q_refined.detach()


def _validate_and_rank_candidates(
    planning_task, q_candidates, q_start, ee_pose_goal, se3_eps, max_candidates, collision_margin
):
    if q_candidates.numel() == 0:
        return torch.empty((0, planning_task.robot.q_dim), **planning_task.tensor_args)

    link_name = planning_task.robot.link_name_ee
    H = planning_task.robot.diff_panda.compute_forward_kinematics_all_links(
        q_candidates, link_list=[link_name]
    ).squeeze(1)
    target = ee_pose_goal.unsqueeze(0).expand(q_candidates.shape[0], -1, -1)
    se3_error = SE3_distance(H, target, w_pos=1.0, w_rot=1.0)
    valid_se3 = se3_error < se3_eps
    if not valid_se3.any():
        return torch.empty((0, planning_task.robot.q_dim), **planning_task.tensor_args)

    q_candidates = q_candidates[valid_se3]
    se3_error = se3_error[valid_se3]
    in_collision = planning_task.compute_collision(q_candidates, margin=0.0).bool().reshape(-1)
    if (~in_collision).sum() == 0:
        return torch.empty((0, planning_task.robot.q_dim), **planning_task.tensor_args)

    q_candidates = q_candidates[~in_collision]
    se3_error = se3_error[~in_collision]
    collision_cost = planning_task.compute_collision_cost(q_candidates, margin=collision_margin).reshape(-1)
    distances = planning_task.robot.distance_q(q_candidates, q_start)

    def normalize(values):
        values = values.reshape(-1)
        if values.numel() <= 1:
            return torch.zeros_like(values)
        return (values - values.min()) / torch.clamp(values.max() - values.min(), min=1e-8)

    ranking_score = 1e6 * normalize(collision_cost) + 1e3 * normalize(se3_error) + normalize(distances)
    return q_candidates[torch.argsort(ranking_score)[:max_candidates]]


def solve_variant(
    planning_task,
    q_start,
    ee_pose_goal,
    method,
    batch_size,
    max_iterations,
    lr,
    se3_eps,
    max_candidates,
    collision_refine_max_iterations,
    collision_refine_lr,
    collision_refine_margin,
    debug=False,
):
    q_start = to_torch(q_start, **planning_task.tensor_args)
    ee_pose_goal = _ensure_homogeneous_pose_matrix(ee_pose_goal, tensor_args=planning_task.tensor_args)

    _synchronize_tensor_device(planning_task.tensor_args)
    t0 = time.perf_counter()
    if method.two_stage:
        q_solutions, idx_valid = _run_two_stage_pose_ik(
            planning_task,
            q_start=q_start,
            ee_pose_goal=ee_pose_goal,
            backend=method.ik_backend,
            batch_size=batch_size,
            max_iterations=max_iterations,
            lr=lr,
            se3_eps=se3_eps,
            max_candidates=max_candidates,
            debug=debug,
        )
    else:
        q_solutions, idx_valid = _run_pose_ik_backend(
            planning_task,
            ee_pose_goal,
            q0=None,
            batch_size=batch_size,
            max_iterations=max_iterations,
            lr=lr,
            se3_eps=se3_eps,
            backend=method.ik_backend,
            q0_noise=torch.pi / 8,
            debug=debug,
        )

    idx_valid = torch.atleast_1d(idx_valid).reshape(-1)
    q_candidates = (
        q_solutions[idx_valid]
        if idx_valid.numel() > 0
        else torch.empty((0, planning_task.robot.q_dim), **planning_task.tensor_args)
    )

    if method.refine_backend == "lbfgs":
        q_candidates = _refine_panda_goal_ik_with_collision(
            planning_task,
            q_candidates=q_candidates,
            q_start=q_start,
            ee_pose_goal=ee_pose_goal,
            max_iterations=collision_refine_max_iterations,
            lr=collision_refine_lr,
            collision_margin=collision_refine_margin,
            debug=debug,
        )
    elif method.refine_backend == "adam":
        q_candidates = _refine_panda_goal_ik_with_collision_adam(
            planning_task,
            q_candidates=q_candidates,
            q_start=q_start,
            ee_pose_goal=ee_pose_goal,
            max_iterations=collision_refine_max_iterations,
            lr=collision_refine_lr,
            collision_margin=collision_refine_margin,
        )

    q_candidates = _validate_and_rank_candidates(
        planning_task,
        q_candidates=q_candidates,
        q_start=q_start,
        ee_pose_goal=ee_pose_goal,
        se3_eps=se3_eps,
        max_candidates=max_candidates,
        collision_margin=collision_refine_margin,
    )
    _synchronize_tensor_device(planning_task.tensor_args)
    elapsed = time.perf_counter() - t0
    return q_candidates, elapsed


def _build_planning_task(cfg, args):
    device = get_torch_device(args.device or cfg.get("device", "cuda:0"))
    tensor_args = {"device": device, "dtype": torch.float32}

    args_inference = DotMap(load_params_from_yaml(_resolve_repo_path(cfg["cfg_inference_path"])))
    args_inference.model_dir = _resolve_model_dir(_select_model_dir(args_inference))
    args_train = DotMap(load_params_from_yaml(os.path.join(args_inference.model_dir, "args.yaml")))

    extra_boxes = _load_extra_boxes(cfg)
    extra_object_fields = build_object_fields_from_boxes(extra_boxes, tensor_args=tensor_args)
    update_kwargs = dict(
        **args_inference,
        gripper=True,
        reload_data=False,
        results_dir=_resolve_repo_path(args.results_dir),
        load_indices=True,
        tensor_args=tensor_args,
        obj_extra_list=extra_object_fields,
    )
    update_kwargs["env_id_replace"] = args.env_id
    if args.env_sdf_cell_size is not None:
        update_kwargs["sdf_cell_size"] = float(args.env_sdf_cell_size)
    elif cfg.get("env_sdf_cell_size") is not None:
        update_kwargs["sdf_cell_size"] = float(cfg["env_sdf_cell_size"])
    args_train.update(**update_kwargs)

    planning_task, _, _, val_subset, _ = get_planning_task_and_dataset(**args_train)
    return planning_task, val_subset, tensor_args, extra_boxes, args_inference.model_dir


def _sample_validation_indices(val_subset, n_samples, seed):
    rng = np.random.default_rng(seed)
    n_available = len(val_subset)
    replace = n_available < n_samples
    return rng.choice(np.arange(n_available), size=n_samples, replace=replace).tolist()


def _format_time(seconds):
    if math.isinf(seconds):
        return "inf"
    return f"{seconds:.6f}"


def main():
    parser = argparse.ArgumentParser(description="Benchmark Panda goal inverse kinematics variants in EnvWarehouse.")
    parser.add_argument("--cfg", default="scripts/deployment/cfgs/interactive_viser_demo_warehouse.yaml")
    parser.add_argument("--results-dir", default="logs/ik_benchmark")
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--num-goals", type=int, default=100)
    parser.add_argument("--env-id", default="EnvWarehouse")
    parser.add_argument("--env-sdf-cell-size", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-candidates", type=int, default=None)
    parser.add_argument("--max-iterations", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--se3-eps", type=float, default=None)
    parser.add_argument("--collision-refine-max-iterations", type=int, default=80)
    parser.add_argument("--collision-refine-lr", type=float, default=5e-2)
    parser.add_argument("--collision-refine-margin", type=float, default=0.03)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    cfg_path = _resolve_repo_path(args.cfg)
    cfg = load_params_from_yaml(cfg_path)
    seed = int(cfg.get("seed", 0) if args.seed is None else args.seed)
    fix_random_seed(seed)

    os.makedirs(_resolve_repo_path(args.results_dir), exist_ok=True)
    planning_task, val_subset, tensor_args, extra_boxes, model_dir = _build_planning_task(cfg, args)
    dataset = val_subset.dataset

    batch_size = int(args.batch_size or cfg.get("n_ik_candidates", 128))
    max_candidates = int(args.max_candidates or cfg.get("max_goal_candidates", 4))
    max_iterations = int(args.max_iterations or cfg.get("ik_max_iterations", 500))
    lr = float(args.lr or cfg.get("ik_lr", 2e-1))
    se3_eps = float(args.se3_eps or cfg.get("ik_se3_eps", 5e-2))

    sample_indices = _sample_validation_indices(val_subset, args.num_goals, seed)
    rows = []
    print("\n----------------IK BENCHMARK----------------")
    print(f"cfg: {cfg_path}")
    print(f"model_dir: {model_dir}")
    print(f"env: {args.env_id}")
    print(f"extra_boxes: {len(extra_boxes)}")
    print(f"device: {tensor_args['device']}")
    print(f"validation_goals: {args.num_goals}")
    print(f"batch_size: {batch_size}, max_iterations: {max_iterations}, se3_eps: {se3_eps}")
    print(f"original torch_robotics commit: {ORIGINAL_TORCH_ROBOTICS_COMMIT}")

    for problem_idx, subset_idx in enumerate(sample_indices):
        sample = val_subset[subset_idx]
        q_start = sample[dataset.field_key_q_start]
        ee_pose_goal = sample[dataset.field_key_context_ee_goal_pose]
        print(f"\n[{problem_idx + 1:03d}/{len(sample_indices):03d}] validation_idx={subset_idx}")
        for method in METHODS:
            try:
                q_candidates, elapsed = solve_variant(
                    planning_task,
                    q_start=q_start,
                    ee_pose_goal=ee_pose_goal,
                    method=method,
                    batch_size=batch_size,
                    max_iterations=max_iterations,
                    lr=lr,
                    se3_eps=se3_eps,
                    max_candidates=max_candidates,
                    collision_refine_max_iterations=args.collision_refine_max_iterations,
                    collision_refine_lr=args.collision_refine_lr,
                    collision_refine_margin=args.collision_refine_margin,
                    debug=args.debug,
                )
                success = bool(q_candidates.shape[0] > 0)
                n_solutions = int(q_candidates.shape[0])
            except Exception as exc:
                elapsed = float("inf")
                success = False
                n_solutions = 0
                print(f"  {method.name:24s} error: {exc}")

            rows.append(
                {
                    "problem_idx": problem_idx,
                    "validation_subset_idx": int(subset_idx),
                    "method": method.name,
                    "success": success,
                    "n_solutions": n_solutions,
                    "time_sec": elapsed if success else float("inf"),
                    "measured_time_sec": elapsed,
                }
            )
            status = "success" if success else "fail"
            print(f"  {method.name:24s} {status:7s} time={_format_time(elapsed)} n={n_solutions}")

    successful_problem_indices = {row["problem_idx"] for row in rows if row["success"]}
    summary = {}
    for method in METHODS:
        method_rows = [
            row for row in rows if row["method"] == method.name and row["problem_idx"] in successful_problem_indices
        ]
        successes = [row["success"] for row in method_rows]
        times = np.asarray([row["time_sec"] for row in method_rows], dtype=float)
        summary[method.name] = {
            "success_rate": float(np.mean(successes)) if successes else 0.0,
            "num_success": int(np.sum(successes)) if successes else 0,
            "num_problems": len(method_rows),
            "median_solution_time_sec": float(np.median(times)) if len(times) else float("inf"),
        }

    results_dir = _resolve_repo_path(args.results_dir)
    csv_path = os.path.join(results_dir, "ik_benchmark_results.csv")
    json_path = os.path.join(results_dir, "ik_benchmark_summary.json")
    config_path = os.path.join(results_dir, "ik_benchmark_config.yaml")
    with open(csv_path, "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with open(json_path, "w") as stream:
        json.dump(
            {
                "summary": summary,
                "eligible_problem_count": len(successful_problem_indices),
                "original_torch_robotics_commit": ORIGINAL_TORCH_ROBOTICS_COMMIT,
            },
            stream,
            indent=2,
        )
    save_to_yaml(vars(args), config_path)

    print("\n----------------SUMMARY----------------")
    print(
        f"Eligible problems where at least one method succeeded: {len(successful_problem_indices)}/{len(sample_indices)}"
    )
    print(f"{'method':24s} {'success':>12s} {'median_time_sec':>18s}")
    for method in METHODS:
        item = summary[method.name]
        median_time = item["median_solution_time_sec"]
        print(
            f"{method.name:24s} "
            f"{item['num_success']:4d}/{item['num_problems']:<7d} "
            f"{_format_time(median_time):>18s}"
        )
    print(f"\nWrote per-problem results to: {csv_path}")
    print(f"Wrote summary to: {json_path}")


if __name__ == "__main__":
    main()
