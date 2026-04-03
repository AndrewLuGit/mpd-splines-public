from mpd.utils.patches import numpy_monkey_patch

numpy_monkey_patch()

# import isaacgym  # noqa: F401

import argparse
import os
from pprint import pprint

import torch
import yaml

from mpd.deployment.goal_ik import build_ee_pose_goal_from_dict
from mpd.deployment.online_planner import OnlineMPDPlanner, prune_extra_boxes_for_collision_free_q_start
from mpd.deployment.scene_primitives import filter_box_specs_for_panda_q
from mpd.paths import REPO_PATH
from mpd.utils.loaders import load_params_from_yaml, save_to_yaml
from torch_robotics.torch_utils.seed import fix_random_seed
from torch_robotics.torch_utils.torch_utils import get_torch_device, to_torch


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


def _save_box_specs(path, box_specs):
    with open(path, "w") as stream:
        yaml.dump({"extra_boxes": list(box_specs or [])}, stream, Dumper=yaml.Dumper, allow_unicode=True)


def main():
    parser = argparse.ArgumentParser(description="Phase-1 online MPD planning with runtime box obstacles")
    parser.add_argument(
        "--cfg",
        default="scripts/deployment/cfgs/phase1_online_planner_warehouse.yaml",
        help="Path to the phase-1 deployment YAML config",
    )
    args = parser.parse_args()

    cfg_path = _resolve_repo_path(args.cfg)
    cfg = load_params_from_yaml(cfg_path)

    fix_random_seed(cfg.get("seed", 0))

    device = get_torch_device(cfg.get("device", "cuda:0"))
    tensor_args = {"device": device, "dtype": torch.float32}

    results_dir = _resolve_repo_path(cfg.get("results_dir", "logs/phase1_online_planner"))
    os.makedirs(results_dir, exist_ok=True)
    save_to_yaml(cfg, os.path.join(results_dir, "config.yaml"))

    raw_extra_boxes = _load_extra_boxes(cfg)
    filter_boxes_at_q_start = bool(cfg.get("filter_extra_boxes_at_q_start", bool(raw_extra_boxes)))

    bootstrap_planner = OnlineMPDPlanner(
        cfg_inference_path=cfg["cfg_inference_path"],
        extra_boxes=[] if filter_boxes_at_q_start else raw_extra_boxes,
        device=cfg.get("device", "cuda:0"),
        debug=cfg.get("debug", False),
        results_dir=results_dir,
        env_id_override=cfg.get("env_id_override", "EnvWarehouse"),
        env_sdf_cell_size=cfg.get("env_sdf_cell_size"),
    )

    reference_sample = bootstrap_planner.get_reference_sample(
        index=cfg.get("reference_index", 0),
        selection=cfg.get("reference_split", "validation"),
    )

    q_start = reference_sample.q_start
    if cfg.get("q_start") is not None:
        q_start = to_torch(cfg["q_start"], **tensor_args)

    ee_pose_goal = reference_sample.ee_goal_pose
    if cfg.get("ee_goal_pose") is not None:
        ee_pose_goal = build_ee_pose_goal_from_dict(cfg["ee_goal_pose"], tensor_args=tensor_args)

    extra_boxes = raw_extra_boxes
    removed_extra_boxes = []
    if filter_boxes_at_q_start and raw_extra_boxes:
        extra_boxes, removed_extra_boxes = filter_box_specs_for_panda_q(
            raw_extra_boxes,
            q=q_start,
            gripper=bool(cfg.get("start_state_filter_gripper", True)),
            sphere_margin=cfg.get("start_state_filter_sphere_margin", 0.0),
            box_margin=cfg.get("start_state_filter_box_margin", 0.0),
        )
        planner_pruned_boxes = []
        planner_filter_summary = None
        if cfg.get("planner_filter_extra_boxes_at_q_start", True):
            extra_boxes, planner_pruned_boxes, planner_filter_summary = prune_extra_boxes_for_collision_free_q_start(
                planner=bootstrap_planner,
                q_start=q_start,
                extra_boxes=extra_boxes,
                max_removals=cfg.get("planner_filter_max_removals"),
                mode=cfg.get("planner_filter_mode", "individual"),
            )
            removed_extra_boxes.extend(planner_pruned_boxes)
        if cfg.get("save_start_state_filtered_boxes", True):
            _save_box_specs(
                os.path.join(results_dir, cfg.get("start_state_filtered_boxes_filename", "phase1_filtered_boxes.yaml")),
                extra_boxes,
            )
            _save_box_specs(
                os.path.join(results_dir, cfg.get("start_state_removed_boxes_filename", "phase1_removed_boxes.yaml")),
                removed_extra_boxes,
            )
            if planner_filter_summary is not None:
                save_to_yaml(
                    planner_filter_summary,
                    os.path.join(
                        results_dir,
                        cfg.get("planner_filter_summary_filename", "phase1_planner_filter_summary.yaml"),
                    ),
                )

    bootstrap_planner.cleanup()

    planner = OnlineMPDPlanner(
        cfg_inference_path=cfg["cfg_inference_path"],
        extra_boxes=extra_boxes,
        device=cfg.get("device", "cuda:0"),
        debug=cfg.get("debug", False),
        results_dir=results_dir,
        env_id_override=cfg.get("env_id_override", "EnvWarehouse"),
        env_sdf_cell_size=cfg.get("env_sdf_cell_size"),
    )

    print("\n----------------PHASE 1 ONLINE MPD----------------")
    print(f"cfg: {cfg_path}")
    print(f"results_dir: {results_dir}")
    print(f"model_dir: {planner.args_inference.model_dir}")
    print(f"q_start: {q_start}")
    print(f"ee_pose_goal: {ee_pose_goal}")
    print(f"extra_boxes: {len(extra_boxes)}")
    if cfg.get("env_sdf_cell_size") is not None:
        print(f"env_sdf_cell_size: {cfg['env_sdf_cell_size']}")
    if cfg.get("extra_boxes_path"):
        print(f"extra_boxes_path: {_resolve_repo_path(cfg['extra_boxes_path'])}")
    if filter_boxes_at_q_start:
        print(f"raw_extra_boxes: {len(raw_extra_boxes)}")
        print(f"removed_boxes_at_q_start: {len(removed_extra_boxes)}")

    try:
        results_single_plan = planner.plan_to_ee_goal(
            q_start=q_start,
            ee_pose_goal=ee_pose_goal,
            n_ik_candidates=cfg.get("n_ik_candidates", 32),
            max_goal_candidates=cfg.get("max_goal_candidates", 4),
            ik_max_iterations=cfg.get("ik_max_iterations", 500),
            ik_lr=cfg.get("ik_lr", 2e-1),
            ik_se3_eps=cfg.get("ik_se3_eps", 5e-2),
            debug=cfg.get("debug", False),
        )
    except RuntimeError as exc:
        if cfg.get("save_ik_debug_plot_on_failure", True) and planner.last_ik_debug_data is not None:
            ik_debug_plot_path = os.path.join(results_dir, cfg.get("ik_debug_plot_filename", "ik_debug.png"))
            planner.save_last_ik_debug_visualization(
                save_path=ik_debug_plot_path,
                show=cfg.get("show_ik_debug_plot", False),
                max_collision_free_to_render=cfg.get("max_collision_free_to_render", 4),
                max_colliding_to_render=cfg.get("max_colliding_to_render", 4),
                draw_collision_spheres=cfg.get("draw_collision_spheres", False),
            )
            print("\n----------------IK DEBUG----------------")
            print(f"Saved IK debug visualization to: {ik_debug_plot_path}")
        raise

    torch.save(results_single_plan, os.path.join(results_dir, "phase1_plan.pt"), _use_new_zipfile_serialization=True)

    print("\n----------------IK ATTEMPTS----------------")
    pprint(results_single_plan.ik_attempt_summaries)

    print("\n----------------METRICS----------------")
    pprint(results_single_plan.metrics)

    if results_single_plan.q_trajs_pos_best is None:
        print("\nNo valid trajectory found for the available IK candidates.")
    else:
        print("\nA valid trajectory was found.")
        print(f"Selected q_goal: {results_single_plan.q_pos_goal}")
        print(f"t_inference_total: {results_single_plan.t_inference_total:.3f} sec")

    run_sapien = cfg.get("run_sapien", cfg.get("run_isaacgym", False))
    if run_sapien and results_single_plan.q_trajs_pos_best is not None:
        sapien_statistics = planner.execute_best_trajectory_in_sapien(
            results_single_plan,
            render_viewer=cfg.get("render_sapien_viewer", cfg.get("render_isaacgym_viewer", True)),
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
        print("\n----------------SAPIEN----------------")
        pprint(sapien_statistics)

    planner.cleanup()


if __name__ == "__main__":
    main()
