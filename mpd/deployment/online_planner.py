import gc
import os

import torch
from dotmap import DotMap

from mpd.deployment.goal_ik import render_panda_goal_ik_debug, solve_panda_goal_ik
from mpd.deployment.sapien_trajectory_executor import (
    build_scene_spec_from_planning_env,
    execute_trajectory_in_sapien,
)
from mpd.deployment.scene_primitives import build_object_fields_from_boxes
from mpd.inference.inference import GenerativeOptimizationPlanner
from mpd.metrics.metrics import PlanningMetricsCalculator
from mpd.paths import REPO_PATH
from mpd.utils.loaders import get_planning_task_and_dataset, load_params_from_yaml
from torch_robotics.torch_utils.torch_utils import get_torch_device, to_torch


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


class OnlineMPDPlanner:
    def __init__(
        self,
        cfg_inference_path,
        extra_boxes=None,
        device="cuda:0",
        debug=False,
        results_dir="logs/phase1_online_planner",
        env_id_override="EnvWarehouse",
    ):
        self.debug = debug
        self.results_dir = _resolve_repo_path(results_dir)
        os.makedirs(self.results_dir, exist_ok=True)

        device = get_torch_device(device)
        self.tensor_args = {"device": device, "dtype": torch.float32}

        self.cfg_inference_path = _resolve_repo_path(cfg_inference_path)
        self.args_inference = DotMap(load_params_from_yaml(self.cfg_inference_path))
        self.args_inference.model_dir = _resolve_model_dir(_select_model_dir(self.args_inference))
        self.args_inference.env_id_replace = env_id_override

        self.args_train = DotMap(load_params_from_yaml(os.path.join(self.args_inference.model_dir, "args.yaml")))

        self.extra_object_fields = build_object_fields_from_boxes(extra_boxes, tensor_args=self.tensor_args)

        self.args_train.update(
            **self.args_inference,
            gripper=True,
            reload_data=False,
            results_dir=self.results_dir,
            load_indices=True,
            tensor_args=self.tensor_args,
            obj_extra_list=self.extra_object_fields,
        )

        self.planning_task, self.train_subset, _, self.val_subset, _ = get_planning_task_and_dataset(**self.args_train)
        self.dataset = self.train_subset.dataset
        self.generative_optimization_planner = GenerativeOptimizationPlanner(
            self.planning_task,
            self.dataset,
            self.args_train,
            self.args_inference,
            tensor_args=self.tensor_args,
            debug=self.debug,
        )
        self.planning_metrics_calculator = PlanningMetricsCalculator(self.planning_task)
        self.last_ik_debug_data = None

    def get_reference_sample(self, index=0, selection="validation"):
        subset = self.val_subset if selection == "validation" else self.train_subset
        sample = subset[index % len(subset)]
        return DotMap(
            q_start=sample[self.dataset.field_key_q_start],
            q_goal=sample[self.dataset.field_key_q_goal],
            ee_goal_pose=sample[self.dataset.field_key_context_ee_goal_pose],
        )

    def solve_goal_ik(
        self,
        q_start,
        ee_pose_goal,
        batch_size=32,
        max_iterations=500,
        lr=2e-1,
        se3_eps=5e-2,
        max_candidates=8,
        debug=False,
    ):
        q_goal_candidates, ik_debug_data = solve_panda_goal_ik(
            self.planning_task,
            q_start=q_start,
            ee_pose_goal=ee_pose_goal,
            batch_size=batch_size,
            max_iterations=max_iterations,
            lr=lr,
            se3_eps=se3_eps,
            max_candidates=max_candidates,
            debug=debug,
            return_debug_data=True,
        )
        self.last_ik_debug_data = ik_debug_data
        return q_goal_candidates

    def save_last_ik_debug_visualization(
        self,
        save_path,
        show=False,
        max_collision_free_to_render=4,
        max_colliding_to_render=4,
        draw_collision_spheres=False,
    ):
        if self.last_ik_debug_data is None:
            raise RuntimeError("No IK debug data is available to visualize")

        render_panda_goal_ik_debug(
            self.planning_task,
            self.last_ik_debug_data,
            save_path=save_path,
            show=show,
            max_collision_free_to_render=max_collision_free_to_render,
            max_colliding_to_render=max_colliding_to_render,
            draw_collision_spheres=draw_collision_spheres,
        )

    def plan_to_ee_goal(
        self,
        q_start,
        ee_pose_goal,
        q_goal_candidates=None,
        n_ik_candidates=32,
        max_goal_candidates=4,
        ik_max_iterations=500,
        ik_lr=2e-1,
        ik_se3_eps=5e-2,
        debug=False,
    ):
        q_start = to_torch(q_start, **self.tensor_args)
        ee_pose_goal = to_torch(ee_pose_goal, **self.tensor_args)
        self.last_ik_debug_data = None

        if q_goal_candidates is None:
            q_goal_candidates = self.solve_goal_ik(
                q_start=q_start,
                ee_pose_goal=ee_pose_goal,
                batch_size=n_ik_candidates,
                max_iterations=ik_max_iterations,
                lr=ik_lr,
                se3_eps=ik_se3_eps,
                max_candidates=max_goal_candidates,
                debug=debug,
            )
        else:
            q_goal_candidates = to_torch(q_goal_candidates, **self.tensor_args)
            if q_goal_candidates.ndim == 1:
                q_goal_candidates = q_goal_candidates.unsqueeze(0)

        if q_goal_candidates.numel() == 0:
            raise RuntimeError("No collision-free IK candidate was found for the requested EE goal pose")

        if self.planning_task.compute_collision(q_start).item():
            raise RuntimeError("q_start is in collision with the current environment")

        attempts = []
        best_failure = None
        for idx, q_goal in enumerate(q_goal_candidates[:max_goal_candidates]):
            results_ns = DotMap(t_generator=0.0, t_guide=0.0, ik_candidate_index=idx, ik_q_goal=q_goal)
            results_single_plan = self.generative_optimization_planner.plan_trajectory(
                q_start,
                q_goal,
                ee_pose_goal,
                results_ns=results_ns,
                debug=debug,
            )
            results_single_plan.metrics = self.planning_metrics_calculator.compute_metrics(results_single_plan)

            n_valid = 0
            if results_single_plan.q_trajs_pos_valid is not None:
                n_valid = int(results_single_plan.q_trajs_pos_valid.shape[0])

            attempts.append(
                DotMap(
                    ik_candidate_index=idx,
                    ik_q_goal=q_goal,
                    n_valid_trajectories=n_valid,
                    success=results_single_plan.q_trajs_pos_best is not None,
                    metrics=results_single_plan.metrics,
                    t_inference_total=results_single_plan.t_inference_total,
                )
            )

            if results_single_plan.q_trajs_pos_best is not None:
                results_single_plan.ik_attempt_summaries = attempts
                return results_single_plan

            best_failure = results_single_plan

        best_failure.ik_attempt_summaries = attempts
        return best_failure

    def execute_best_trajectory_in_isaacgym(
        self,
        results_single_plan,
        render_viewer=True,
        render_movie=False,
        draw_collision_spheres=False,
        video_path=None,
    ):
        import isaacgym  # noqa: F401

        from torch_robotics.isaac_gym_envs.motion_planning_envs import (
            MotionPlanningControllerIsaacGym,
            MotionPlanningIsaacGymEnv,
        )
        from torch_robotics.torch_kinematics_tree.utils.files import get_robot_path

        if results_single_plan.q_trajs_pos_valid is None or results_single_plan.q_trajs_pos_valid.shape[0] == 0:
            raise RuntimeError("There is no valid trajectory to execute in Isaac Gym")

        robot_asset_file = self.planning_task.robot.robot_urdf_file
        if draw_collision_spheres:
            robot_asset_file = self.planning_task.robot.robot_urdf_collision_spheres_file

        motion_planning_isaac_env = MotionPlanningIsaacGymEnv(
            self.planning_task.env,
            self.planning_task.robot,
            asset_root=get_robot_path().as_posix(),
            robot_asset_file=robot_asset_file.replace(get_robot_path().as_posix() + "/", ""),
            num_envs=results_single_plan.q_trajs_pos_valid.shape[0],
            all_robots_in_one_env=True,
            render_isaacgym_viewer=render_viewer,
            render_camera_global=render_movie,
            render_camera_global_append_to_recorder=render_movie,
            sync_viewer_with_real_time=False,
            show_viewer=render_viewer,
            camera_global_from_top=True if self.planning_task.env.dim == 2 else False,
            add_ground_plane=False,
            viewer_time_between_steps=torch.diff(self.planning_task.parametric_trajectory.get_timesteps()[:2]).item(),
            draw_goal_configuration=False,
            draw_ee_pose_goal=True,
            color_robots=False,
            draw_contact_forces=False,
            draw_end_effector_frame=False,
            draw_end_effector_path=True,
        )

        motion_planning_controller_isaac_gym = MotionPlanningControllerIsaacGym(motion_planning_isaac_env)
        motion_planning_isaac_env.ee_pose_goal = self.planning_task.robot.get_EE_pose(
            to_torch(results_single_plan.q_pos_goal.unsqueeze(0), **self.tensor_args),
            flatten_pos_quat=True,
            quat_xyzw=True,
        ).squeeze(0)

        q_trajs_pos = results_single_plan.q_trajs_pos_valid.movedim(1, 0)
        statistics = motion_planning_controller_isaac_gym.execute_trajectories(
            q_trajs_pos,
            q_pos_starts=q_trajs_pos[0],
            q_pos_goal=q_trajs_pos[-1][0],
            n_pre_steps=5 if render_viewer or render_movie else 0,
            n_post_steps=5 if render_viewer or render_movie else 0,
            stop_robot_if_in_contact=False,
            make_video=render_movie,
            video_duration=self.args_inference.trajectory_duration,
            video_path=video_path if video_path is not None else os.path.join(self.results_dir, "phase1_isaacgym.mp4"),
            make_gif=False,
        )

        motion_planning_isaac_env.clean_up()
        del motion_planning_controller_isaac_gym
        del motion_planning_isaac_env
        gc.collect()
        torch.cuda.empty_cache()
        return statistics

    def cleanup(self):
        gc.collect()
        torch.cuda.empty_cache()

    def execute_best_trajectory_in_sapien(
        self,
        results_single_plan,
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
        robot_cfg=None,
        viewer_preset="isaac_gym_default",
    ):
        if results_single_plan.q_trajs_pos_best is None:
            raise RuntimeError("There is no valid trajectory to execute in SAPIEN")

        scene_spec = build_scene_spec_from_planning_env(self.planning_task.env)

        return execute_trajectory_in_sapien(
            q_traj=results_single_plan.q_trajs_pos_best,
            timesteps=results_single_plan.timesteps,
            scene_spec=scene_spec,
            robot_cfg=robot_cfg,
            render_viewer=render_viewer,
            add_ground=add_ground,
            scene_timestep=scene_timestep,
            render_every_n_steps=render_every_n_steps,
            stiffness=stiffness,
            damping=damping,
            force_limit=force_limit,
            drive_mode=drive_mode,
            balance_passive_force=balance_passive_force,
            compensate_gravity=compensate_gravity,
            compensate_coriolis_and_centrifugal=compensate_coriolis_and_centrifugal,
            n_pre_steps=n_pre_steps,
            n_post_steps=n_post_steps,
            viewer_preset=viewer_preset,
        )


def evaluate_q_start_collision_with_boxes(
    cfg_inference_path,
    q_start,
    extra_boxes=None,
    device="cuda:0",
    debug=False,
    results_dir="logs/phase1_online_planner",
    env_id_override="EnvWarehouse",
):
    planner = OnlineMPDPlanner(
        cfg_inference_path=cfg_inference_path,
        extra_boxes=extra_boxes,
        device=device,
        debug=debug,
        results_dir=results_dir,
        env_id_override=env_id_override,
    )
    try:
        q_start_t = to_torch(q_start, **planner.tensor_args)
        colliding = bool(planner.planning_task.compute_collision(q_start_t).reshape(-1)[0].item())
        collision_cost = planner.planning_task.compute_collision_cost(q_start_t)
        collision_cost = float(torch.as_tensor(collision_cost).reshape(-1).max().item())
        return {"colliding": colliding, "collision_cost": collision_cost}
    finally:
        planner.cleanup()


def prune_extra_boxes_for_collision_free_q_start(
    cfg_inference_path,
    q_start,
    extra_boxes,
    device="cuda:0",
    debug=False,
    results_dir="logs/phase1_online_planner",
    env_id_override="EnvWarehouse",
    max_removals=None,
    mode="individual",
):
    extra_boxes = list(extra_boxes or [])
    diagnostics = []

    base_metrics = evaluate_q_start_collision_with_boxes(
        cfg_inference_path=cfg_inference_path,
        q_start=q_start,
        extra_boxes=[],
        device=device,
        debug=debug,
        results_dir=results_dir,
        env_id_override=env_id_override,
    )
    if base_metrics["colliding"]:
        return extra_boxes, [], {"base_metrics": base_metrics, "iterations": diagnostics, "mode": mode}

    if mode == "individual":
        kept_boxes = []
        removed_boxes = []
        for box_idx, box_spec in enumerate(extra_boxes):
            candidate_metrics = evaluate_q_start_collision_with_boxes(
                cfg_inference_path=cfg_inference_path,
                q_start=q_start,
                extra_boxes=[box_spec],
                device=device,
                debug=debug,
                results_dir=results_dir,
                env_id_override=env_id_override,
            )
            if candidate_metrics["colliding"]:
                removed_boxes.append(box_spec)
                diagnostics.append(
                    {
                        "removed_box_name": box_spec.get("name", f"runtime_box_{box_idx}"),
                        "reason": "box_alone_causes_q_start_collision",
                        "collision_cost_with_box_only": candidate_metrics["collision_cost"],
                    }
                )
            else:
                kept_boxes.append(box_spec)

        final_metrics = evaluate_q_start_collision_with_boxes(
            cfg_inference_path=cfg_inference_path,
            q_start=q_start,
            extra_boxes=kept_boxes,
            device=device,
            debug=debug,
            results_dir=results_dir,
            env_id_override=env_id_override,
        )
        return kept_boxes, removed_boxes, {
            "base_metrics": base_metrics,
            "final_metrics": final_metrics,
            "iterations": diagnostics,
            "mode": mode,
        }

    current_boxes = list(extra_boxes)
    removed_boxes = []
    current_metrics = evaluate_q_start_collision_with_boxes(
        cfg_inference_path=cfg_inference_path,
        q_start=q_start,
        extra_boxes=current_boxes,
        device=device,
        debug=debug,
        results_dir=results_dir,
        env_id_override=env_id_override,
    )

    max_removals = len(current_boxes) if max_removals is None else int(max_removals)
    while current_metrics["colliding"] and current_boxes and len(removed_boxes) < max_removals:
        best_idx = None
        best_metrics = None
        best_box = None

        for box_idx, box_spec in enumerate(current_boxes):
            candidate_boxes = current_boxes[:box_idx] + current_boxes[box_idx + 1 :]
            candidate_metrics = evaluate_q_start_collision_with_boxes(
                cfg_inference_path=cfg_inference_path,
                q_start=q_start,
                extra_boxes=candidate_boxes,
                device=device,
                debug=debug,
                results_dir=results_dir,
                env_id_override=env_id_override,
            )

            is_better = False
            if best_metrics is None:
                is_better = True
            elif best_metrics["colliding"] and not candidate_metrics["colliding"]:
                is_better = True
            elif best_metrics["colliding"] == candidate_metrics["colliding"]:
                is_better = candidate_metrics["collision_cost"] < best_metrics["collision_cost"]

            if is_better:
                best_idx = box_idx
                best_metrics = candidate_metrics
                best_box = box_spec

        if best_idx is None or best_metrics is None:
            break

        diagnostics.append(
            {
                "removed_box_name": best_box.get("name", f"runtime_box_{best_idx}"),
                "collision_cost_before": current_metrics["collision_cost"],
                "collision_cost_after": best_metrics["collision_cost"],
                "colliding_before": current_metrics["colliding"],
                "colliding_after": best_metrics["colliding"],
            }
        )
        removed_boxes.append(current_boxes.pop(best_idx))
        current_metrics = best_metrics

    return current_boxes, removed_boxes, {
        "base_metrics": base_metrics,
        "final_metrics": current_metrics,
        "iterations": diagnostics,
        "mode": mode,
    }
