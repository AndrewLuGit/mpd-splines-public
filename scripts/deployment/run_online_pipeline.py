from mpd.utils.patches import numpy_monkey_patch

numpy_monkey_patch()

import argparse
import os
from collections.abc import Mapping
from copy import deepcopy
from pprint import pprint

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")

import torch
import yaml

from mpd.deployment.goal_ik import build_ee_pose_goal_from_dict
from mpd.deployment.nvblox_bridge import (
    build_robot_ignore_spheres,
    reconstruct_occupancy_from_bundle,
    save_reconstruction_outputs,
)
from mpd.deployment.online_planner import (
    OnlineMPDPlanner,
    prune_extra_boxes_for_collision_free_q_start,
)
from mpd.deployment.scene_box_visualizer import render_scene_boxes_debug
from mpd.deployment.scene_primitives import build_object_fields_from_boxes, filter_box_specs_for_panda_q
from mpd.deployment.sapien_depth_adapter import (
    build_capture_request,
    build_warehouse_scene_spec,
    capture_depth_with_sapien,
    parse_camera_specs,
    save_depth_capture_bundle,
    save_depth_preview_images,
    save_yaml,
)
from mpd.deployment.scene_voxelizer import save_occupancy_projections
from mpd.paths import REPO_PATH
from mpd.utils.loaders import load_params_from_yaml, save_to_yaml
from torch_robotics.torch_utils.seed import fix_random_seed
from torch_robotics.torch_utils.torch_utils import get_torch_device, to_torch


def _resolve_repo_path(path):
    if path is None:
        return None
    if os.path.isabs(path):
        return path
    return os.path.join(REPO_PATH, path)


def _load_yaml(path):
    with open(path, "r") as stream:
        return yaml.safe_load(stream) or {}


def _deep_update(base, updates):
    merged = deepcopy(base)
    for key, value in (updates or {}).items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _deep_update(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _save_box_specs(path, box_specs):
    with open(path, "w") as stream:
        yaml.dump({"extra_boxes": list(box_specs or [])}, stream, Dumper=yaml.Dumper, allow_unicode=True)


def _prepare_start_and_goal(cfg_pipeline, cfg_phase1, tensor_args, results_dir):
    bootstrap_planner = OnlineMPDPlanner(
        cfg_inference_path=cfg_phase1["cfg_inference_path"],
        extra_boxes=[],
        device=cfg_phase1.get("device", "cuda:0"),
        debug=cfg_phase1.get("debug", False),
        results_dir=results_dir,
        env_id_override=cfg_phase1.get("env_id_override", "EnvWarehouse"),
    )
    try:
        reference_sample = bootstrap_planner.get_reference_sample(
            index=cfg_pipeline.get("reference_index", cfg_phase1.get("reference_index", 0)),
            selection=cfg_pipeline.get("reference_split", cfg_phase1.get("reference_split", "validation")),
        )

        q_start = reference_sample.q_start
        if cfg_pipeline.get("q_start") is not None:
            q_start = to_torch(cfg_pipeline["q_start"], **tensor_args)

        ee_pose_goal = reference_sample.ee_goal_pose
        if cfg_pipeline.get("ee_goal_pose") is not None:
            ee_pose_goal = build_ee_pose_goal_from_dict(cfg_pipeline["ee_goal_pose"], tensor_args=tensor_args)

        return bootstrap_planner, q_start, ee_pose_goal
    except Exception:
        bootstrap_planner.cleanup()
        raise


def _run_capture_phase(cfg_phase2, q_start, runtime_extra_boxes):
    results_dir = _resolve_repo_path(cfg_phase2["results_dir"])
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, "config.yaml"), "w") as stream:
        yaml.dump(cfg_phase2, stream, Dumper=yaml.Dumper, allow_unicode=True)

    extra_object_fields = build_object_fields_from_boxes(runtime_extra_boxes)
    scene_spec = build_warehouse_scene_spec(
        extra_boxes=extra_object_fields,
        rotation_z_axis_deg=cfg_phase2.get("rotation_z_axis_deg", 0.0),
    )
    camera_specs = parse_camera_specs(cfg_phase2["camera_specs"])
    capture_request = build_capture_request(
        scene_spec=scene_spec,
        camera_specs=camera_specs,
        metadata={"config_path": os.path.join(results_dir, "config.yaml")},
    )

    scene_spec_path = os.path.join(results_dir, cfg_phase2.get("scene_spec_filename", "scene_spec.yaml"))
    capture_request_path = os.path.join(results_dir, cfg_phase2.get("capture_request_filename", "capture_request.yaml"))
    save_yaml(scene_spec, scene_spec_path)
    save_yaml(capture_request, capture_request_path)

    robot_cfg = deepcopy(cfg_phase2.get("robot", {}))
    robot_cfg["enabled"] = True
    robot_cfg["qpos"] = to_torch(q_start, device=torch.device("cpu"), dtype=torch.float32).tolist()

    bundle = capture_depth_with_sapien(
        scene_spec=scene_spec,
        camera_specs=camera_specs,
        metadata={"config_path": os.path.join(results_dir, "config.yaml")},
        add_ground=cfg_phase2.get("add_ground", False),
        capture_backend=cfg_phase2.get("capture_backend", "render_camera"),
        stereo_sensor_config_overrides=cfg_phase2.get("stereo_sensor_config"),
        robot_cfg=robot_cfg,
    )
    bundle_path = os.path.join(results_dir, cfg_phase2.get("bundle_filename", "depth_capture_bundle.npz"))
    save_depth_capture_bundle(bundle, bundle_path)
    preview_dir = os.path.join(results_dir, cfg_phase2.get("preview_dirname", "depth_previews"))
    preview_paths = save_depth_preview_images(
        bundle,
        output_dir=preview_dir,
        cmap=cfg_phase2.get("preview_cmap", "viridis"),
    )

    return {
        "results_dir": results_dir,
        "scene_spec": scene_spec,
        "scene_spec_path": scene_spec_path,
        "capture_request_path": capture_request_path,
        "bundle": bundle,
        "bundle_path": bundle_path,
        "preview_paths": preview_paths,
        "robot_cfg": robot_cfg,
    }


def _run_reconstruction_phase(cfg_phase3, bundle, scene_spec, q_start):
    results_dir = _resolve_repo_path(cfg_phase3["results_dir"])
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, "config.yaml"), "w") as stream:
        yaml.dump(cfg_phase3, stream, Dumper=yaml.Dumper, allow_unicode=True)

    robot_cfg = deepcopy(cfg_phase3.get("robot", {}))
    if not robot_cfg:
        robot_cfg = {"enabled": True}
    robot_cfg["enabled"] = True
    robot_cfg["qpos"] = to_torch(q_start, device=torch.device("cpu"), dtype=torch.float32).tolist()

    robot_ignore_spheres = []
    if cfg_phase3.get("subtract_robot", True):
        robot_ignore_spheres = build_robot_ignore_spheres(
            robot_cfg=robot_cfg,
            robot_sphere_margin=cfg_phase3.get("robot_sphere_margin", 0.0),
        )

    result = reconstruct_occupancy_from_bundle(
        bundle=bundle,
        scene_spec=scene_spec,
        workspace_limits=cfg_phase3.get("workspace_limits", scene_spec["workspace_limits"]),
        voxel_size=cfg_phase3.get("voxel_size", 0.05),
        integrator_type=cfg_phase3.get("integrator_type", "occupancy"),
        extraction_method=cfg_phase3.get("extraction_method"),
        mapper_params_cfg=cfg_phase3.get("mapper_params"),
        device=cfg_phase3.get("device", "cuda"),
        pose_convention=cfg_phase3.get("pose_convention", "sapien"),
        occupancy_threshold_log_odds=cfg_phase3.get("occupancy_threshold_log_odds", 0.0),
        tsdf_weight_threshold=cfg_phase3.get("tsdf_weight_threshold", 1e-4),
        tsdf_distance_threshold=cfg_phase3.get("tsdf_distance_threshold", 0.0),
        query_chunk_size=cfg_phase3.get("query_chunk_size", 250000),
        workspace_padding=cfg_phase3.get("workspace_padding", 0.0),
        ignore_boxes=cfg_phase3.get("ignore_boxes"),
        ignore_box_margin=cfg_phase3.get("ignore_box_margin", 0.0),
        ignore_scene_box_sources=cfg_phase3.get("ignore_scene_box_sources"),
        ignore_scene_box_margin=cfg_phase3.get("ignore_scene_box_margin", 0.0),
        robot_ignore_spheres=robot_ignore_spheres,
        robot_sphere_margin=0.0,
        min_component_voxels=cfg_phase3.get("min_component_voxels", 1),
        max_boxes=cfg_phase3.get("max_boxes"),
        merge_strategy=cfg_phase3.get("merge_strategy", "greedy_cuboids"),
    )

    saved_paths = save_reconstruction_outputs(
        result,
        results_dir=results_dir,
        boxes_filename=cfg_phase3.get("boxes_filename", "phase3_boxes.yaml"),
        summary_filename=cfg_phase3.get("summary_filename", "phase3_summary.yaml"),
        voxels_filename=cfg_phase3.get("voxels_filename", "phase3_occupied_voxels.npz"),
    )

    projection_paths = []
    if cfg_phase3.get("save_occupancy_projections", True):
        projection_dir = os.path.join(results_dir, cfg_phase3.get("projection_dirname", "occupancy_projections"))
        projection_paths = save_occupancy_projections(
            result.occupancy_mask,
            output_dir=projection_dir,
            prefix=cfg_phase3.get("projection_prefix", "phase3"),
        )

    scene_debug_path = None
    if cfg_phase3.get("save_scene_debug_plot", True):
        scene_debug_path = os.path.join(results_dir, cfg_phase3.get("scene_debug_plot_filename", "phase3_scene_debug.png"))
        render_scene_boxes_debug(
            scene_spec=scene_spec,
            reconstructed_boxes=result.merged_boxes,
            occupied_voxel_centers=result.occupied_voxel_centers if cfg_phase3.get("draw_occupied_voxels", True) else None,
            robot_ignore_spheres=result.robot_ignore_spheres if cfg_phase3.get("draw_robot_ignore_spheres", True) else None,
            save_path=scene_debug_path,
            show=cfg_phase3.get("show_scene_debug_plot", False),
            title=cfg_phase3.get("scene_debug_title", "Phase 3 Scene Debug"),
        )

    return {
        "results_dir": results_dir,
        "result": result,
        "saved_paths": saved_paths,
        "projection_paths": projection_paths,
        "scene_debug_path": scene_debug_path,
    }


def _run_planning_phase(cfg_phase1, q_start, ee_pose_goal, filtered_boxes, results_dir):
    planner = OnlineMPDPlanner(
        cfg_inference_path=cfg_phase1["cfg_inference_path"],
        extra_boxes=filtered_boxes,
        device=cfg_phase1.get("device", "cuda:0"),
        debug=cfg_phase1.get("debug", False),
        results_dir=results_dir,
        env_id_override=cfg_phase1.get("env_id_override", "EnvWarehouse"),
    )

    try:
        results_single_plan = planner.plan_to_ee_goal(
            q_start=q_start,
            ee_pose_goal=ee_pose_goal,
            n_ik_candidates=cfg_phase1.get("n_ik_candidates", 32),
            max_goal_candidates=cfg_phase1.get("max_goal_candidates", 4),
            ik_max_iterations=cfg_phase1.get("ik_max_iterations", 500),
            ik_lr=cfg_phase1.get("ik_lr", 2e-1),
            ik_se3_eps=cfg_phase1.get("ik_se3_eps", 5e-2),
            debug=cfg_phase1.get("debug", False),
        )

        torch.save(
            results_single_plan,
            os.path.join(results_dir, cfg_phase1.get("plan_filename", "phase1_plan.pt")),
            _use_new_zipfile_serialization=True,
        )

        sapien_statistics = None
        if cfg_phase1.get("run_sapien", False) and results_single_plan.q_trajs_pos_best is not None:
            sapien_statistics = planner.execute_best_trajectory_in_sapien(
                results_single_plan,
                render_viewer=cfg_phase1.get("render_sapien_viewer", True),
                add_ground=cfg_phase1.get("sapien_add_ground", False),
                scene_timestep=cfg_phase1.get("sapien_scene_timestep", 1.0 / 240.0),
                render_every_n_steps=cfg_phase1.get("sapien_render_every_n_steps", 4),
                stiffness=cfg_phase1.get("sapien_drive_stiffness", 200.0),
                damping=cfg_phase1.get("sapien_drive_damping", 40.0),
                force_limit=cfg_phase1.get("sapien_force_limit", 1000.0),
                drive_mode=cfg_phase1.get("sapien_drive_mode", "force"),
                balance_passive_force=cfg_phase1.get("sapien_balance_passive_force", True),
                compensate_gravity=cfg_phase1.get("sapien_compensate_gravity", True),
                compensate_coriolis_and_centrifugal=cfg_phase1.get("sapien_compensate_coriolis_and_centrifugal", True),
                n_pre_steps=cfg_phase1.get("sapien_n_pre_steps", 5),
                n_post_steps=cfg_phase1.get("sapien_n_post_steps", 10),
                robot_cfg=cfg_phase1.get("sapien_robot"),
                viewer_preset=cfg_phase1.get("sapien_viewer_preset", "isaac_gym_default"),
            )

        return planner, results_single_plan, sapien_statistics
    except Exception:
        planner.cleanup()
        raise


def main():
    parser = argparse.ArgumentParser(description="Combined deployment pipeline: SAPIEN capture -> nvblox -> online planning")
    parser.add_argument(
        "--cfg",
        default="scripts/deployment/cfgs/online_pipeline_warehouse.yaml",
        help="Path to the combined deployment YAML config",
    )
    args = parser.parse_args()

    cfg_path = _resolve_repo_path(args.cfg)
    cfg_pipeline = _load_yaml(cfg_path)

    cfg_phase1 = _deep_update(
        _load_yaml(_resolve_repo_path(cfg_pipeline["phase1_cfg_path"])),
        cfg_pipeline.get("phase1_overrides"),
    )
    cfg_phase2 = _deep_update(
        _load_yaml(_resolve_repo_path(cfg_pipeline["phase2_cfg_path"])),
        cfg_pipeline.get("phase2_overrides"),
    )
    cfg_phase3 = _deep_update(
        _load_yaml(_resolve_repo_path(cfg_pipeline["phase3_cfg_path"])),
        cfg_pipeline.get("phase3_overrides"),
    )

    fix_random_seed(cfg_pipeline.get("seed", cfg_phase1.get("seed", 0)))
    device = get_torch_device(cfg_phase1.get("device", "cuda:0"))
    tensor_args = {"device": device, "dtype": torch.float32}

    base_results_dir = _resolve_repo_path(cfg_pipeline.get("results_dir", "logs/online_pipeline"))
    os.makedirs(base_results_dir, exist_ok=True)
    save_to_yaml(cfg_pipeline, os.path.join(base_results_dir, "config.yaml"))

    cfg_phase2["results_dir"] = os.path.join(base_results_dir, cfg_pipeline.get("phase2_results_subdir", "phase2_capture"))
    cfg_phase3["results_dir"] = os.path.join(base_results_dir, cfg_pipeline.get("phase3_results_subdir", "phase3_nvblox"))
    cfg_phase1["results_dir"] = os.path.join(base_results_dir, cfg_pipeline.get("phase1_results_subdir", "phase1_planning"))

    bootstrap_planner, q_start, ee_pose_goal = _prepare_start_and_goal(
        cfg_pipeline=cfg_pipeline,
        cfg_phase1=cfg_phase1,
        tensor_args=tensor_args,
        results_dir=cfg_phase1["results_dir"],
    )

    runtime_extra_boxes = deepcopy(cfg_pipeline.get("runtime_extra_boxes"))
    if runtime_extra_boxes is None:
        runtime_extra_boxes = deepcopy(cfg_phase2.get("extra_boxes", cfg_phase1.get("extra_boxes", [])))

    capture_info = None
    reconstruction_info = None
    planning_results = None
    try:
        capture_info = _run_capture_phase(
            cfg_phase2=cfg_phase2,
            q_start=q_start,
            runtime_extra_boxes=runtime_extra_boxes,
        )

        cfg_phase3["depth_bundle_path"] = capture_info["bundle_path"]
        cfg_phase3["scene_spec_path"] = capture_info["scene_spec_path"]
        cfg_phase3["robot"] = {"enabled": True, "qpos": to_torch(q_start, device=torch.device("cpu"), dtype=torch.float32).tolist()}
        reconstruction_info = _run_reconstruction_phase(
            cfg_phase3=cfg_phase3,
            bundle=capture_info["bundle"],
            scene_spec=capture_info["scene_spec"],
            q_start=q_start,
        )

        filtered_boxes, removed_boxes = filter_box_specs_for_panda_q(
            reconstruction_info["result"].merged_boxes,
            q=q_start,
            gripper=bool(cfg_phase1.get("start_state_filter_gripper", False)),
            sphere_margin=cfg_phase1.get("start_state_filter_sphere_margin", 0.0),
            box_margin=cfg_phase1.get("start_state_filter_box_margin", 0.0),
        )
        planner_pruned_boxes = []
        planner_filter_summary = None
        if cfg_phase1.get("planner_filter_extra_boxes_at_q_start", True):
            filtered_boxes, planner_pruned_boxes, planner_filter_summary = prune_extra_boxes_for_collision_free_q_start(
                cfg_inference_path=cfg_phase1["cfg_inference_path"],
                q_start=q_start,
                extra_boxes=filtered_boxes,
                device=cfg_phase1.get("device", "cuda:0"),
                debug=cfg_phase1.get("debug", False),
                results_dir=cfg_phase1["results_dir"],
                env_id_override=cfg_phase1.get("env_id_override", "EnvWarehouse"),
                max_removals=cfg_phase1.get("planner_filter_max_removals"),
                mode=cfg_phase1.get("planner_filter_mode", "individual"),
            )
            removed_boxes.extend(planner_pruned_boxes)

        phase1_results_dir = _resolve_repo_path(cfg_phase1["results_dir"])
        os.makedirs(phase1_results_dir, exist_ok=True)
        _save_box_specs(
            os.path.join(phase1_results_dir, cfg_phase1.get("start_state_filtered_boxes_filename", "phase1_filtered_boxes.yaml")),
            filtered_boxes,
        )
        _save_box_specs(
            os.path.join(phase1_results_dir, cfg_phase1.get("start_state_removed_boxes_filename", "phase1_removed_boxes.yaml")),
            removed_boxes,
        )
        cfg_phase1["sapien_robot"] = _deep_update(
            cfg_phase1.get("sapien_robot", {}),
            {"enabled": True, "qpos": to_torch(q_start, device=torch.device("cpu"), dtype=torch.float32).tolist()},
        )
        save_to_yaml(cfg_phase1, os.path.join(phase1_results_dir, "config.yaml"))
        if planner_filter_summary is not None:
            save_to_yaml(
                planner_filter_summary,
                os.path.join(
                    phase1_results_dir,
                    cfg_phase1.get("planner_filter_summary_filename", "phase1_planner_filter_summary.yaml"),
                ),
            )

        planner, results_single_plan, sapien_statistics = _run_planning_phase(
            cfg_phase1=cfg_phase1,
            q_start=q_start,
            ee_pose_goal=ee_pose_goal,
            filtered_boxes=filtered_boxes,
            results_dir=phase1_results_dir,
        )
        planning_results = {
            "planner": planner,
            "results_single_plan": results_single_plan,
            "sapien_statistics": sapien_statistics,
            "filtered_boxes": filtered_boxes,
            "removed_boxes": removed_boxes,
        }

        print("\n----------------COMBINED PIPELINE----------------")
        print(f"cfg: {cfg_path}")
        print(f"results_dir: {base_results_dir}")
        print(f"q_start: {q_start}")
        print(f"ee_pose_goal: {ee_pose_goal}")
        print(f"runtime_extra_boxes_for_capture: {len(runtime_extra_boxes)}")
        print(f"reconstructed_boxes: {len(reconstruction_info['result'].merged_boxes)}")
        print(f"filtered_boxes_for_planning: {len(filtered_boxes)}")
        print(f"removed_boxes_at_q_start: {len(removed_boxes)}")
        if planner_pruned_boxes:
            print(f"planner_pruned_boxes_at_q_start: {len(planner_pruned_boxes)}")

        print("\n----------------CAPTURE----------------")
        print(f"scene_spec_path: {capture_info['scene_spec_path']}")
        print(f"capture_request_path: {capture_info['capture_request_path']}")
        print(f"bundle_path: {capture_info['bundle_path']}")
        for frame, preview_path in zip(capture_info["bundle"].frames, capture_info["preview_paths"]):
            valid_pixels = int((frame.depth_meters > 0.0).sum())
            print(f"  camera={frame.camera_name} valid_pixels={valid_pixels} preview={preview_path}")

        print("\n----------------NVBLOX----------------")
        print(f"boxes_path: {reconstruction_info['saved_paths']['boxes_path']}")
        print(f"summary_path: {reconstruction_info['saved_paths']['summary_path']}")
        print(f"voxels_path: {reconstruction_info['saved_paths']['voxels_path']}")
        if reconstruction_info["scene_debug_path"] is not None:
            print(f"scene_debug_plot: {reconstruction_info['scene_debug_path']}")

        print("\n----------------PLANNING----------------")
        pprint(results_single_plan.ik_attempt_summaries)
        pprint(results_single_plan.metrics)
        if results_single_plan.q_trajs_pos_best is None:
            print("\nNo valid trajectory found for the available IK candidates.")
        else:
            print("\nA valid trajectory was found.")
            print(f"Selected q_goal: {results_single_plan.q_pos_goal}")
            print(f"t_inference_total: {results_single_plan.t_inference_total:.3f} sec")
        if sapien_statistics is not None:
            print("\n----------------SAPIEN----------------")
            pprint(sapien_statistics)
    finally:
        bootstrap_planner.cleanup()
        if planning_results is not None and planning_results.get("planner") is not None:
            planning_results["planner"].cleanup()


if __name__ == "__main__":
    main()
