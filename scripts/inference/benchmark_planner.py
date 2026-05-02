import argparse
import csv
import gc
import json
import math
import os
import sys
from dataclasses import dataclass
from functools import partial
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mpd.utils.patches import numpy_monkey_patch

numpy_monkey_patch()

import numpy as np
import torch
from dotmap import DotMap
from einops._torch_specific import allow_ops_in_compiled_graph

from mpd.deployment.sapien_trajectory_executor import (
    PersistentSapienTrajectoryExecutor,
    build_scene_spec_from_planning_env,
)
from mpd.inference.inference import EvaluationSamplesGenerator, GenerativeOptimizationPlanner
from mpd.utils.loaders import get_planning_task_and_dataset, load_params_from_yaml, save_to_yaml
from torch_robotics.torch_utils.seed import fix_random_seed
from torch_robotics.torch_utils.torch_utils import get_torch_device

allow_ops_in_compiled_graph()
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"


@dataclass(frozen=True)
class PlannerConfig:
    name: str
    planner_alg: str
    n_trajectory_samples: int


def _resolve_repo_path(path):
    path = os.path.expanduser(os.path.expandvars(path))
    if os.path.isabs(path):
        return path
    return os.path.abspath(path)


def _select_model_dir(args_inference):
    if "cvae" in args_inference.planner_alg:
        if args_inference.model_selection == "bspline":
            model_dir = args_inference.model_dir_cvae_bspline
        elif args_inference.model_selection == "waypoints":
            model_dir = args_inference.model_dir_cvae_waypoints
        else:
            raise ValueError(f"Unknown model_selection: {args_inference.model_selection}")
    else:
        if args_inference.model_selection == "bspline":
            model_dir = args_inference.model_dir_ddpm_bspline
        elif args_inference.model_selection == "waypoints":
            model_dir = args_inference.model_dir_ddpm_waypoints
        else:
            raise ValueError(f"Unknown model_selection: {args_inference.model_selection}")
    return _resolve_model_dir(model_dir)


def _resolve_model_dir(model_dir):
    expanded = os.path.expandvars(model_dir)
    if os.path.exists(expanded):
        return expanded

    marker = "data_trained_models"
    if marker in expanded:
        suffix = expanded.split(f"{marker}/", 1)[1]
        candidates = [
            REPO_ROOT / marker / suffix,
            REPO_ROOT / "data_public" / marker / suffix,
        ]
        for candidate in candidates:
            if candidate.exists():
                return str(candidate)

    raise FileNotFoundError(
        f"Could not resolve model directory: {model_dir}. "
        f"Tried the expanded path and repo-local {marker} fallbacks."
    )


def _make_args_inference(base_cfg, planner_config, env_id):
    args_inference = DotMap(dict(base_cfg))
    args_inference.planner_alg = planner_config.planner_alg
    args_inference.n_trajectory_samples = planner_config.n_trajectory_samples
    args_inference.env_id_replace = env_id
    args_inference.model_dir = _select_model_dir(args_inference)
    return args_inference


def _make_planning_stack(args_inference, results_dir, tensor_args):
    args_train = DotMap(load_params_from_yaml(os.path.join(args_inference.model_dir, "args.yaml")))
    args_train.update(
        **args_inference,
        gripper=True,
        reload_data=False,
        results_dir=results_dir,
        load_indices=True,
        tensor_args=tensor_args,
    )
    planning_task, train_subset, _, val_subset, _ = get_planning_task_and_dataset(**args_train)
    return planning_task, train_subset, val_subset, args_train


def _sample_validation_problems(sample_generator, num_problems):
    problems = []
    for problem_idx in range(num_problems):
        q_pos_start, q_pos_goal, ee_pose_goal = sample_generator.get_data_sample(problem_idx)
        problems.append(
            {
                "problem_idx": problem_idx,
                "q_pos_start": q_pos_start.detach().cpu(),
                "q_pos_goal": q_pos_goal.detach().cpu(),
                "ee_pose_goal": ee_pose_goal.detach().cpu(),
            }
        )
    return problems


def _format_time(seconds):
    if math.isinf(seconds):
        return "inf"
    return f"{seconds:.6f}"


def _to_csv_value(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float, str)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    return value


def _write_csv(rows, path):
    if not rows:
        return
    with open(path, "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows([{key: _to_csv_value(value) for key, value in row.items()} for row in rows])


def _drop_tensors_inplace(value):
    if value is None:
        return
    if isinstance(value, torch.Tensor):
        return
    if isinstance(value, dict) or hasattr(value, "items"):
        for key in list(value.keys()):
            if isinstance(value[key], torch.Tensor):
                value[key] = None
            else:
                _drop_tensors_inplace(value[key])
    elif isinstance(value, list):
        for idx, item in enumerate(value):
            if isinstance(item, torch.Tensor):
                value[idx] = None
            else:
                _drop_tensors_inplace(item)


def _release_result_tensors(results):
    if results is not None:
        _drop_tensors_inplace(results)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _cuda_memory_text():
    if not torch.cuda.is_available():
        return "cuda unavailable"
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    max_allocated = torch.cuda.max_memory_allocated() / 1024**3
    return f"allocated={allocated:.2f}GiB reserved={reserved:.2f}GiB peak={max_allocated:.2f}GiB"


def _cuda_tensor_census_text(max_entries=8):
    if not torch.cuda.is_available():
        return "cuda unavailable"

    stats = {}
    for obj in gc.get_objects():
        try:
            if isinstance(obj, torch.Tensor) and obj.is_cuda:
                key = (tuple(obj.shape), str(obj.dtype), bool(obj.requires_grad))
                nbytes = obj.numel() * obj.element_size()
                count, total = stats.get(key, (0, 0))
                stats[key] = (count + 1, total + nbytes)
        except Exception:
            pass

    entries = sorted(stats.items(), key=lambda item: item[1][1], reverse=True)[:max_entries]
    parts = []
    for (shape, dtype, requires_grad), (count, nbytes) in entries:
        parts.append(f"shape={shape} dtype={dtype} grad={requires_grad} count={count} bytes={nbytes / 1024**3:.2f}GiB")
    return " | ".join(parts) if parts else "no live cuda tensors found by gc"


def _infinite_sapien_metrics():
    return {
        "sapien_success": False,
        "sapien_median_tracking_error_l2": float("inf"),
        "sapien_mean_tracking_error_l2": float("inf"),
        "sapien_max_tracking_error_l2": float("inf"),
        "sapien_final_tracking_error_l2": float("inf"),
        "sapien_num_sim_steps": 0,
        "sapien_error": "",
    }


def _run_sapien_execution(executor, results, scene_spec, sapien_options):
    stats = executor.execute_trajectory(
        q_traj=results["q_traj"],
        timesteps=results["timesteps"],
        scene_spec=scene_spec,
        robot_cfg=sapien_options["robot_cfg"],
        render_viewer=sapien_options["render_viewer"],
        add_ground=sapien_options["add_ground"],
        scene_timestep=sapien_options["scene_timestep"],
        render_every_n_steps=sapien_options["render_every_n_steps"],
        stiffness=sapien_options["stiffness"],
        damping=sapien_options["damping"],
        force_limit=sapien_options["force_limit"],
        drive_mode=sapien_options["drive_mode"],
        balance_passive_force=sapien_options["balance_passive_force"],
        compensate_gravity=sapien_options["compensate_gravity"],
        compensate_coriolis_and_centrifugal=sapien_options["compensate_coriolis_and_centrifugal"],
        n_pre_steps=sapien_options["n_pre_steps"],
        n_post_steps=sapien_options["n_post_steps"],
        viewer_preset=sapien_options["viewer_preset"],
    )
    return {
        "sapien_success": True,
        "sapien_median_tracking_error_l2": float(stats.get("median_tracking_error_l2", float("inf"))),
        "sapien_mean_tracking_error_l2": float(stats.get("mean_tracking_error_l2", float("inf"))),
        "sapien_max_tracking_error_l2": float(stats.get("max_tracking_error_l2", float("inf"))),
        "sapien_final_tracking_error_l2": float(stats.get("final_tracking_error_l2", float("inf"))),
        "sapien_num_sim_steps": int(stats.get("num_sim_steps", 0)),
        "sapien_error": "",
    }


def _make_cpu_execution_payload(results):
    if results.q_trajs_pos_best is None:
        return None
    return {
        "q_traj": results.q_trajs_pos_best.detach().cpu(),
        "timesteps": results.timesteps.detach().cpu() if hasattr(results.timesteps, "detach") else results.timesteps,
    }


def _compute_best_path_length(q_traj_cpu, robot):
    if q_traj_cpu is None:
        return float("inf")
    q_pos_cpu = robot.get_position(q_traj_cpu)
    path_length = torch.linalg.norm(torch.diff(q_pos_cpu, dim=-2), dim=-1).sum()
    return float(path_length.item())


def _planner_configs(baseline_samples):
    return [
        PlannerConfig("mpd_100", "mpd", 100),
        PlannerConfig("mpd_10", "mpd", 10),
        PlannerConfig("mpd_1", "mpd", 1),
        PlannerConfig("gp_prior_then_guide", "gp_prior_then_guide", baseline_samples),
    ]


def _run_planner_config(
    planner_config,
    base_cfg,
    env_id,
    problems,
    results_dir,
    tensor_args,
    planner_allowed_time,
    sapien_options,
    sapien_executor,
    log_cuda_memory,
    debug,
):
    args_inference = _make_args_inference(base_cfg, planner_config, env_id)
    config_results_dir = os.path.join(results_dir, planner_config.name)
    os.makedirs(config_results_dir, exist_ok=True)
    save_to_yaml(args_inference.toDict(), os.path.join(config_results_dir, "args_inference.yaml"))

    planning_task, train_subset, val_subset, args_train = _make_planning_stack(
        args_inference, config_results_dir, tensor_args
    )
    sample_generator = EvaluationSamplesGenerator(
        planning_task,
        train_subset,
        val_subset,
        selection_start_goal="validation",
        planner="RRTConnect",
        tensor_args=tensor_args,
        debug=debug,
        render_pybullet=False,
        **args_inference,
    )
    planner = GenerativeOptimizationPlanner(
        planning_task,
        train_subset.dataset,
        args_train,
        args_inference,
        tensor_args,
        sampling_based_planner_fn=partial(
            sample_generator.generate_data_ompl_worker.run,
            planner_allowed_time=planner_allowed_time,
            interpolate_num=args_inference.num_T_pts,
            simplify_path=True,
        ),
        debug=debug,
    )
    scene_spec = build_scene_spec_from_planning_env(planning_task.env)

    rows = []
    try:
        for problem in problems:
            problem_idx = problem["problem_idx"]
            q_pos_start = problem["q_pos_start"].to(**tensor_args)
            q_pos_goal = problem["q_pos_goal"].to(**tensor_args)
            ee_pose_goal = problem["ee_pose_goal"].to(**tensor_args)
            results = None
            execution_payload = None

            print(f"  [{problem_idx + 1:03d}/{len(problems):03d}] {planner_config.name}", flush=True)
            if log_cuda_memory:
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
                print(f"      cuda before: {_cuda_memory_text()}", flush=True)
            try:
                results = planner.plan_trajectory(
                    q_pos_start,
                    q_pos_goal,
                    ee_pose_goal,
                    results_ns=DotMap(t_generator=0.0, t_guide=0.0),
                    debug=debug,
                )
                success = bool(results.q_trajs_pos_best is not None)
                elapsed = float(results.t_inference_total)
                n_valid = int(results.q_trajs_pos_valid.shape[0]) if results.q_trajs_pos_valid is not None else 0
                execution_payload = _make_cpu_execution_payload(results)
                path_length = _compute_best_path_length(
                    execution_payload["q_traj"] if execution_payload is not None else None,
                    planning_task.robot,
                )
                _release_result_tensors(results)
                sapien_metrics = _infinite_sapien_metrics()
                if execution_payload is not None and sapien_executor is not None:
                    try:
                        sapien_metrics = _run_sapien_execution(
                            sapien_executor,
                            execution_payload,
                            scene_spec,
                            sapien_options,
                        )
                    except Exception as exc:
                        sapien_metrics["sapien_error"] = repr(exc)
                        print(f"      sapien error: {exc}", flush=True)
                row = {
                    "problem_idx": problem_idx,
                    "method": planner_config.name,
                    "planner_alg": planner_config.planner_alg,
                    "n_trajectory_samples": planner_config.n_trajectory_samples,
                    "success": success,
                    "n_valid_trajectories": n_valid,
                    "time_sec": elapsed if success else float("inf"),
                    "path_length": path_length,
                    "measured_time_sec": elapsed,
                    "t_generator_sec": float(results.get("t_generator", 0.0)),
                    "t_guide_sec": float(results.get("t_guide", 0.0)),
                    "error": "",
                }
                row.update(sapien_metrics)
                status = "success" if success else "fail"
                print(
                    f"      {status:7s} time={_format_time(elapsed)} n_valid={n_valid} "
                    f"tracking={_format_time(row['sapien_median_tracking_error_l2'])} "
                    f"path_length={_format_time(row['path_length'])}",
                    flush=True,
                )
            except Exception as exc:
                row = {
                    "problem_idx": problem_idx,
                    "method": planner_config.name,
                    "planner_alg": planner_config.planner_alg,
                    "n_trajectory_samples": planner_config.n_trajectory_samples,
                    "success": False,
                    "n_valid_trajectories": 0,
                    "time_sec": float("inf"),
                    "path_length": float("inf"),
                    "measured_time_sec": float("inf"),
                    "t_generator_sec": 0.0,
                    "t_guide_sec": 0.0,
                    "error": repr(exc),
                }
                row.update(_infinite_sapien_metrics())
                print(f"      error: {exc}", flush=True)
            rows.append(row)

            del q_pos_start
            del q_pos_goal
            del ee_pose_goal
            del execution_payload
            del results
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if log_cuda_memory:
                print(f"      cuda after:  {_cuda_memory_text()}", flush=True)
                print(f"      cuda tensors: {_cuda_tensor_census_text()}", flush=True)
    finally:
        sample_generator.generate_data_ompl_worker.terminate()
        del planner
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return rows


def _summarize(rows, planner_configs, num_problems):
    successful_problem_indices = {row["problem_idx"] for row in rows if row["success"]}
    summary = {}
    for planner_config in planner_configs:
        method_rows = [
            row
            for row in rows
            if row["method"] == planner_config.name and row["problem_idx"] in successful_problem_indices
        ]
        successes = np.asarray([row["success"] for row in method_rows], dtype=bool)
        times = np.asarray([row["time_sec"] for row in method_rows], dtype=float)
        summary[planner_config.name] = {
            "planner_alg": planner_config.planner_alg,
            "n_trajectory_samples": planner_config.n_trajectory_samples,
            "success_rate": float(np.mean(successes)) if len(successes) else 0.0,
            "num_success": int(np.sum(successes)) if len(successes) else 0,
            "num_eligible_problems": len(method_rows),
            "median_solution_time_sec": float(np.median(times)) if len(times) else float("inf"),
            "median_tracking_error_l2": (
                float(np.median([row["sapien_median_tracking_error_l2"] for row in method_rows]))
                if len(method_rows)
                else float("inf")
            ),
            "median_path_length": (
                float(np.median([row["path_length"] for row in method_rows])) if len(method_rows) else float("inf")
            ),
        }
    return {
        "summary": summary,
        "eligible_problem_count": len(successful_problem_indices),
        "num_sampled_problems": num_problems,
    }


def main():
    parser = argparse.ArgumentParser(
        description=("Benchmark MPD planner variants on EnvWarehouse extra obstacles using validation-set problems.")
    )
    parser.add_argument(
        "--cfg", default="scripts/inference/cfgs/config_EnvWarehouse-RobotPanda-config_file_v01_00.yaml"
    )
    parser.add_argument("--results-dir", default="logs/planner_benchmark")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--num-problems", type=int, default=100)
    parser.add_argument("--env-id", default="EnvWarehouseExtraObjectsV00")
    parser.add_argument("--baseline-samples", type=int, default=100)
    parser.add_argument("--planner-allowed-time", type=float, default=10.0)
    parser.add_argument("--skip-sapien", action="store_true")
    parser.add_argument("--render-sapien-viewer", action="store_true")
    parser.add_argument("--sapien-viewer-preset", default="isaac_gym_default")
    parser.add_argument("--sapien-add-ground", action="store_true")
    parser.add_argument("--sapien-scene-timestep", type=float, default=1.0 / 240.0)
    parser.add_argument("--sapien-render-every-n-steps", type=int, default=4)
    parser.add_argument("--sapien-drive-stiffness", type=float, default=200.0)
    parser.add_argument("--sapien-drive-damping", type=float, default=40.0)
    parser.add_argument("--sapien-force-limit", type=float, default=1000.0)
    parser.add_argument("--sapien-drive-mode", default="force")
    parser.add_argument("--sapien-balance-passive-force", dest="sapien_balance_passive_force", action="store_true")
    parser.add_argument("--no-sapien-balance-passive-force", dest="sapien_balance_passive_force", action="store_false")
    parser.set_defaults(sapien_balance_passive_force=True)
    parser.add_argument("--sapien-compensate-gravity", dest="sapien_compensate_gravity", action="store_true")
    parser.add_argument("--no-sapien-compensate-gravity", dest="sapien_compensate_gravity", action="store_false")
    parser.set_defaults(sapien_compensate_gravity=True)
    parser.add_argument(
        "--sapien-compensate-coriolis-and-centrifugal",
        dest="sapien_compensate_coriolis_and_centrifugal",
        action="store_true",
    )
    parser.add_argument(
        "--no-sapien-compensate-coriolis-and-centrifugal",
        dest="sapien_compensate_coriolis_and_centrifugal",
        action="store_false",
    )
    parser.set_defaults(sapien_compensate_coriolis_and_centrifugal=True)
    parser.add_argument("--sapien-n-pre-steps", type=int, default=5)
    parser.add_argument("--sapien-n-post-steps", type=int, default=10)
    parser.add_argument("--log-cuda-memory", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    fix_random_seed(args.seed)
    device = get_torch_device(args.device)
    tensor_args = {"device": device, "dtype": torch.float32}
    cfg_path = _resolve_repo_path(args.cfg)
    results_dir = _resolve_repo_path(args.results_dir)
    os.makedirs(results_dir, exist_ok=True)

    base_cfg = load_params_from_yaml(cfg_path)
    planner_configs = _planner_configs(args.baseline_samples)
    sapien_options = {
        "enabled": not args.skip_sapien,
        "render_viewer": args.render_sapien_viewer,
        "viewer_preset": args.sapien_viewer_preset,
        "add_ground": args.sapien_add_ground,
        "scene_timestep": args.sapien_scene_timestep,
        "render_every_n_steps": args.sapien_render_every_n_steps,
        "stiffness": args.sapien_drive_stiffness,
        "damping": args.sapien_drive_damping,
        "force_limit": args.sapien_force_limit,
        "drive_mode": args.sapien_drive_mode,
        "balance_passive_force": args.sapien_balance_passive_force,
        "compensate_gravity": args.sapien_compensate_gravity,
        "compensate_coriolis_and_centrifugal": args.sapien_compensate_coriolis_and_centrifugal,
        "n_pre_steps": args.sapien_n_pre_steps,
        "n_post_steps": args.sapien_n_post_steps,
        "robot_cfg": {"enabled": True, "fix_root_link": True},
    }

    print("\n----------------PLANNER BENCHMARK----------------")
    print(f"cfg: {cfg_path}")
    print(f"env_id: {args.env_id}")
    print(f"device: {device}")
    print(f"num_problems: {args.num_problems}")
    print(f"results_dir: {results_dir}")

    sample_args_inference = _make_args_inference(base_cfg, planner_configs[0], args.env_id)
    planning_task, train_subset, val_subset, _ = _make_planning_stack(sample_args_inference, results_dir, tensor_args)
    sample_generator = EvaluationSamplesGenerator(
        planning_task,
        train_subset,
        val_subset,
        selection_start_goal="validation",
        planner="RRTConnect",
        tensor_args=tensor_args,
        debug=args.debug,
        render_pybullet=False,
        **sample_args_inference,
    )
    try:
        problems = _sample_validation_problems(sample_generator, args.num_problems)
    finally:
        sample_generator.generate_data_ompl_worker.terminate()

    problems_path = os.path.join(results_dir, "sampled_validation_problems.pt")
    torch.save(problems, problems_path, _use_new_zipfile_serialization=True)
    del sample_generator
    del planning_task
    del train_subset
    del val_subset
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    rows = []
    sapien_executor = None
    if sapien_options["enabled"]:
        sapien_executor = PersistentSapienTrajectoryExecutor(viewer_preset=sapien_options["viewer_preset"])
    try:
        for planner_config in planner_configs:
            print(
                f"\n----------------{planner_config.name} "
                f"({planner_config.n_trajectory_samples} samples)----------------",
                flush=True,
            )
            rows.extend(
                _run_planner_config(
                    planner_config,
                    base_cfg,
                    args.env_id,
                    problems,
                    results_dir,
                    tensor_args,
                    args.planner_allowed_time,
                    sapien_options,
                    sapien_executor,
                    args.log_cuda_memory,
                    args.debug,
                )
            )
    finally:
        if sapien_executor is not None:
            sapien_executor.close()

    report = _summarize(rows, planner_configs, args.num_problems)
    report.update(
        {
            "cfg": cfg_path,
            "env_id": args.env_id,
            "seed": args.seed,
            "device": str(device),
            "sampled_problems": problems_path,
            "sapien_enabled": sapien_options["enabled"],
        }
    )

    csv_path = os.path.join(results_dir, "planner_benchmark_results.csv")
    json_path = os.path.join(results_dir, "planner_benchmark_summary.json")
    config_path = os.path.join(results_dir, "planner_benchmark_config.yaml")
    _write_csv(rows, csv_path)
    with open(json_path, "w") as stream:
        json.dump(report, stream, indent=2)
    save_to_yaml(vars(args), config_path)

    print("\n----------------SUMMARY----------------")
    print(
        "Eligible problems where at least one method succeeded: "
        f"{report['eligible_problem_count']}/{report['num_sampled_problems']}"
    )
    print(
        f"{'method':24s} {'success':>12s} {'median_time_sec':>18s} "
        f"{'median_tracking_l2':>20s} {'median_path_length':>20s}"
    )
    for planner_config in planner_configs:
        item = report["summary"][planner_config.name]
        print(
            f"{planner_config.name:24s} "
            f"{item['num_success']:4d}/{item['num_eligible_problems']:<7d} "
            f"{_format_time(item['median_solution_time_sec']):>18s} "
            f"{_format_time(item['median_tracking_error_l2']):>20s} "
            f"{_format_time(item['median_path_length']):>20s}"
        )
    print(f"\nWrote per-problem results to: {csv_path}")
    print(f"Wrote summary to: {json_path}")
    print(f"Wrote sampled problems to: {problems_path}")


if __name__ == "__main__":
    main()
