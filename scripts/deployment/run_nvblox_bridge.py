import argparse
import os
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")

import yaml

REPO_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_PATH not in sys.path:
    sys.path.insert(0, REPO_PATH)
TORCH_ROBOTICS_PATH = os.path.join(REPO_PATH, "mpd", "torch_robotics")
if TORCH_ROBOTICS_PATH not in sys.path:
    sys.path.insert(0, TORCH_ROBOTICS_PATH)

from mpd.deployment.nvblox_bridge import (
    build_robot_ignore_spheres,
    load_yaml,
    reconstruct_occupancy_from_bundle,
    save_reconstruction_outputs,
)
from mpd.deployment.scene_box_visualizer import render_scene_boxes_debug
from mpd.deployment.sapien_depth_adapter import load_depth_capture_bundle
from mpd.deployment.scene_voxelizer import save_occupancy_projections
from mpd.paths import REPO_PATH


def _resolve_repo_path(path):
    if os.path.isabs(path):
        return path
    return os.path.join(REPO_PATH, path)


def main():
    parser = argparse.ArgumentParser(description="Phase-3 nvblox reconstruction bridge for the warehouse scene")
    parser.add_argument(
        "--cfg",
        default="scripts/deployment/cfgs/phase3_nvblox_bridge_warehouse.yaml",
        help="Path to the phase-3 nvblox bridge YAML config",
    )
    args = parser.parse_args()

    cfg_path = _resolve_repo_path(args.cfg)
    with open(cfg_path, "r") as stream:
        cfg = yaml.safe_load(stream)

    results_dir = _resolve_repo_path(cfg.get("results_dir", "logs/phase3_nvblox_bridge"))
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, "config.yaml"), "w") as stream:
        yaml.dump(cfg, stream, Dumper=yaml.Dumper, allow_unicode=True)

    bundle_path = _resolve_repo_path(cfg["depth_bundle_path"])
    scene_spec_path = _resolve_repo_path(cfg["scene_spec_path"])

    bundle = load_depth_capture_bundle(bundle_path)
    scene_spec = load_yaml(scene_spec_path)
    phase2_cfg_path = cfg.get("phase2_config_path", bundle.metadata.get("config_path"))
    phase2_cfg = load_yaml(_resolve_repo_path(phase2_cfg_path)) if phase2_cfg_path else {}
    robot_cfg = phase2_cfg.get("robot")
    robot_ignore_spheres = []
    if cfg.get("subtract_robot", True):
        robot_ignore_spheres = build_robot_ignore_spheres(
            robot_cfg=robot_cfg,
            robot_sphere_margin=cfg.get("robot_sphere_margin", 0.0),
        )

    result = reconstruct_occupancy_from_bundle(
        bundle=bundle,
        scene_spec=scene_spec,
        workspace_limits=cfg.get("workspace_limits", scene_spec["workspace_limits"]),
        voxel_size=cfg.get("voxel_size", 0.05),
        integrator_type=cfg.get("integrator_type", "occupancy"),
        extraction_method=cfg.get("extraction_method"),
        mapper_params_cfg=cfg.get("mapper_params"),
        device=cfg.get("device", "cuda"),
        pose_convention=cfg.get("pose_convention", "sapien"),
        occupancy_threshold_log_odds=cfg.get("occupancy_threshold_log_odds", 0.0),
        tsdf_weight_threshold=cfg.get("tsdf_weight_threshold", 1e-4),
        tsdf_distance_threshold=cfg.get("tsdf_distance_threshold", 0.0),
        query_chunk_size=cfg.get("query_chunk_size", 250000),
        workspace_padding=cfg.get("workspace_padding", 0.0),
        ignore_boxes=cfg.get("ignore_boxes"),
        ignore_box_margin=cfg.get("ignore_box_margin", 0.0),
        ignore_scene_box_sources=cfg.get("ignore_scene_box_sources"),
        ignore_scene_box_margin=cfg.get("ignore_scene_box_margin", 0.0),
        robot_ignore_spheres=robot_ignore_spheres,
        robot_sphere_margin=0.0,
        min_component_voxels=cfg.get("min_component_voxels", 1),
        max_boxes=cfg.get("max_boxes"),
        merge_strategy=cfg.get("merge_strategy", "greedy_cuboids"),
    )

    saved_paths = save_reconstruction_outputs(
        result,
        results_dir=results_dir,
        boxes_filename=cfg.get("boxes_filename", "phase3_boxes.yaml"),
        summary_filename=cfg.get("summary_filename", "phase3_summary.yaml"),
        voxels_filename=cfg.get("voxels_filename", "phase3_occupied_voxels.npz"),
    )

    projection_paths = []
    if cfg.get("save_occupancy_projections", True):
        projection_dir = os.path.join(results_dir, cfg.get("projection_dirname", "occupancy_projections"))
        projection_paths = save_occupancy_projections(
            result.occupancy_mask,
            output_dir=projection_dir,
            prefix=cfg.get("projection_prefix", "phase3"),
        )

    scene_debug_path = None
    if cfg.get("save_scene_debug_plot", True):
        scene_debug_path = os.path.join(results_dir, cfg.get("scene_debug_plot_filename", "phase3_scene_debug.png"))
        render_scene_boxes_debug(
            scene_spec=scene_spec,
            reconstructed_boxes=result.merged_boxes,
            occupied_voxel_centers=result.occupied_voxel_centers if cfg.get("draw_occupied_voxels", True) else None,
            robot_ignore_spheres=result.robot_ignore_spheres if cfg.get("draw_robot_ignore_spheres", True) else None,
            save_path=scene_debug_path,
            show=cfg.get("show_scene_debug_plot", False),
            title=cfg.get("scene_debug_title", "Phase 3 Scene Debug"),
        )

    print("\n----------------PHASE 3 NVBLOX----------------")
    print(f"cfg: {cfg_path}")
    print(f"bundle_path: {bundle_path}")
    print(f"scene_spec_path: {scene_spec_path}")
    print(f"results_dir: {results_dir}")
    print(f"integrator_type: {cfg.get('integrator_type', 'occupancy')}")
    print(f"extraction_method: {result.metadata.get('extraction_method')}")
    print(f"pose_convention: {result.metadata.get('pose_convention')}")
    print(f"voxel_size: {cfg.get('voxel_size', 0.05)}")
    print(f"n_frames: {result.metadata['n_frames']}")
    print(f"n_query_voxels: {result.metadata['n_query_voxels']}")
    print(f"n_occupied_voxels: {result.metadata['n_occupied_voxels']}")
    print(f"n_merged_boxes: {len(result.merged_boxes)}")
    print(f"n_robot_ignore_spheres: {len(result.robot_ignore_spheres)}")
    print(f"boxes_path: {saved_paths['boxes_path']}")
    print(f"summary_path: {saved_paths['summary_path']}")
    print(f"voxels_path: {saved_paths['voxels_path']}")
    if scene_debug_path is not None:
        print(f"scene_debug_plot: {scene_debug_path}")
    if projection_paths:
        print("occupancy_projections:")
        for path in projection_paths:
            print(f"  {path}")


if __name__ == "__main__":
    main()
