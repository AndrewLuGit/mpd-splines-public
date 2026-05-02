import argparse
import csv
import json
import os
import sys
import time
from copy import deepcopy

import numpy as np

REPO_PATH_BOOTSTRAP = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_PATH_BOOTSTRAP not in sys.path:
    sys.path.insert(0, REPO_PATH_BOOTSTRAP)
TORCH_ROBOTICS_PATH = os.path.join(REPO_PATH_BOOTSTRAP, "mpd", "torch_robotics")
if TORCH_ROBOTICS_PATH not in sys.path:
    sys.path.insert(0, TORCH_ROBOTICS_PATH)

from mpd.utils.patches import numpy_monkey_patch

numpy_monkey_patch()

import torch
import yaml

from mpd.deployment.nvblox_bridge import (
    build_robot_ignore_spheres,
    reconstruct_occupancy_from_bundle,
    require_nvblox_torch,
)
from mpd.deployment.sapien_depth_adapter import (
    build_warehouse_scene_spec,
    capture_depth_with_sapien,
    parse_camera_specs,
    save_yaml,
)
from mpd.deployment.scene_primitives import build_object_fields_from_boxes
from mpd.deployment.scene_voxelizer import make_voxel_grid, mask_points_in_boxes
from mpd.paths import REPO_PATH
from torch_robotics.torch_utils.seed import fix_random_seed


def _resolve_repo_path(path):
    expanded = os.path.expandvars(path)
    if os.path.isabs(expanded):
        return expanded
    return os.path.join(REPO_PATH, expanded)


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


def _sample_center_from_bounds(center, size, rng, randomization_cfg):
    center = np.asarray(center, dtype=np.float32).reshape(3)
    size = np.asarray(size, dtype=np.float32).reshape(3)
    sampled = center.copy()

    bounds = randomization_cfg.get("center_bounds")
    if bounds is None:
        bounds = {"x": [0.35, 1.05], "y": [-0.35, 0.35]}

    keep_inside_bounds = bool(randomization_cfg.get("keep_box_inside_bounds", True))
    if isinstance(bounds, dict):
        axis_to_index = {"x": 0, "y": 1, "z": 2}
        for axis_name, axis_bounds in bounds.items():
            axis_idx = axis_to_index[str(axis_name).lower()]
            low, high = [float(v) for v in axis_bounds]
            if keep_inside_bounds:
                half = 0.5 * float(size[axis_idx])
                low += half
                high -= half
            if high < low:
                raise ValueError(f"Randomization bounds for axis '{axis_name}' are smaller than box size")
            sampled[axis_idx] = rng.uniform(low, high)
        return sampled

    bounds_array = np.asarray(bounds, dtype=np.float32)
    if bounds_array.shape != (2, 3):
        raise ValueError("randomization.center_bounds must be a dict of axes or a [[low_xyz], [high_xyz]] array")

    low = bounds_array[0].copy()
    high = bounds_array[1].copy()
    if keep_inside_bounds:
        half_size = 0.5 * size
        low += half_size
        high -= half_size
    if np.any(high < low):
        raise ValueError("randomization.center_bounds are smaller than at least one box size")
    sampled = rng.uniform(low, high).astype(np.float32)
    if randomization_cfg.get("preserve_z", True):
        sampled[2] = center[2]
    return sampled


def _randomize_extra_boxes(base_boxes, rng, randomization_cfg):
    randomized_boxes = []
    for idx, box in enumerate(base_boxes):
        randomized_box = deepcopy(box)
        randomized_box.setdefault("name", f"runtime_box_{idx}")
        randomized_box["center"] = (
            _sample_center_from_bounds(
                randomized_box["center"],
                randomized_box["size"],
                rng,
                randomization_cfg,
            )
            .astype(np.float32)
            .tolist()
        )
        randomized_boxes.append(randomized_box)
    return randomized_boxes


def _sync_device(device):
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def _time_nvblox_reconstruction(bundle, scene_spec, cfg, robot_ignore_spheres):
    device = cfg.get("device", "cuda")
    _sync_device(device)
    start = time.perf_counter()
    result = reconstruct_occupancy_from_bundle(
        bundle=bundle,
        workspace_limits=cfg.get("workspace_limits", scene_spec["workspace_limits"]),
        voxel_size=cfg.get("voxel_size", 0.05),
        integrator_type=cfg.get("integrator_type", "occupancy"),
        mapper_params_cfg=cfg.get("mapper_params"),
        device=device,
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
        scene_spec=scene_spec,
        robot_ignore_spheres=robot_ignore_spheres,
        robot_sphere_margin=cfg.get("robot_sphere_margin", 0.0),
        inflate_robot_mask_by_voxel_extent=cfg.get("inflate_robot_mask_by_voxel_extent", True),
        update_esdf=cfg.get("update_esdf", False),
        keep_mapper=False,
        min_component_voxels=cfg.get("min_component_voxels", 1),
        max_boxes=cfg.get("max_boxes"),
        extraction_method=cfg.get("extraction_method"),
        merge_strategy=cfg.get("merge_strategy", "greedy_cuboids"),
    )
    _sync_device(device)
    return result, time.perf_counter() - start


def _voxelized_box_iou(reconstructed_boxes, ground_truth_boxes, workspace_limits, voxel_size, workspace_padding=0.0):
    voxel_grid = make_voxel_grid(
        workspace_limits=workspace_limits,
        voxel_size=voxel_size,
        padding=workspace_padding,
    )
    reconstructed_mask = mask_points_in_boxes(voxel_grid.centers, reconstructed_boxes)
    ground_truth_mask = mask_points_in_boxes(voxel_grid.centers, ground_truth_boxes)

    union = int(np.logical_or(reconstructed_mask, ground_truth_mask).sum())
    if union == 0:
        return 1.0

    intersection = int(np.logical_and(reconstructed_mask, ground_truth_mask).sum())
    return float(intersection / union)


def _extra_scene_boxes(scene_spec):
    return [box for box in scene_spec.get("boxes", []) if box.get("source") == "extra"]


def _write_csv(rows, path):
    if not rows:
        return

    with open(path, "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Benchmark nvblox reconstruction on randomized EnvWarehouse boxes.")
    parser.add_argument(
        "--cfg",
        default="scripts/deployment/cfgs/benchmark_nvblox_reconstruction_warehouse.yaml",
        help="Path to the nvblox reconstruction benchmark YAML config",
    )
    parser.add_argument("--iterations", type=int, default=None, help="Override the config iteration count")
    parser.add_argument("--seed", type=int, default=None, help="Override the config random seed")
    args = parser.parse_args()

    cfg_path = _resolve_repo_path(args.cfg)
    with open(cfg_path, "r") as stream:
        cfg = yaml.safe_load(stream) or {}

    iterations = int(args.iterations if args.iterations is not None else cfg.get("iterations", 100))
    seed = int(args.seed if args.seed is not None else cfg.get("seed", 0))
    if iterations <= 0:
        raise ValueError(f"iterations must be positive, got {iterations}")

    fix_random_seed(seed)
    rng = np.random.default_rng(seed)

    require_nvblox_torch()

    results_dir = _resolve_repo_path(cfg.get("results_dir", "logs/nvblox_reconstruction_benchmark"))
    os.makedirs(results_dir, exist_ok=True)
    save_yaml(cfg, os.path.join(results_dir, "benchmark_config.yaml"))

    base_extra_boxes = _load_extra_boxes(cfg)
    camera_specs = parse_camera_specs(cfg["camera_specs"])
    robot_ignore_spheres = []
    if cfg.get("subtract_robot", True):
        robot_ignore_spheres = build_robot_ignore_spheres(
            robot_cfg=deepcopy(cfg.get("robot", {})),
            robot_sphere_margin=0.0,
        )

    rows = []
    capture_times = []
    reconstruction_times = []
    box_ious = []

    print("\n----------------NVBLOX RECONSTRUCTION BENCHMARK----------------")
    print(f"cfg: {cfg_path}")
    print(f"results_dir: {results_dir}")
    print(f"iterations: {iterations}")
    print(f"seed: {seed}")
    print(f"extra_boxes: {len(base_extra_boxes)}")
    print(f"capture_backend: {cfg.get('capture_backend', 'render_camera')}")
    print(f"timed section: nvblox reconstruction only; depth capture is recorded separately")

    for iteration in range(iterations):
        runtime_extra_boxes = _randomize_extra_boxes(
            base_extra_boxes,
            rng=rng,
            randomization_cfg=cfg.get("randomization", {}) or {},
        )
        extra_object_fields = build_object_fields_from_boxes(runtime_extra_boxes)
        scene_spec = build_warehouse_scene_spec(
            extra_boxes=extra_object_fields,
            rotation_z_axis_deg=cfg.get("rotation_z_axis_deg", 0.0),
        )

        capture_start = time.perf_counter()
        bundle = capture_depth_with_sapien(
            scene_spec=scene_spec,
            camera_specs=camera_specs,
            metadata={"config_path": cfg_path, "iteration": iteration},
            add_ground=cfg.get("add_ground", False),
            capture_backend=cfg.get("capture_backend", "render_camera"),
            stereo_sensor_config_overrides=cfg.get("stereo_sensor_config"),
            robot_cfg=cfg.get("robot"),
            mask_robot_from_depth=cfg.get("mask_robot_from_depth", False),
            robot_depth_mask_epsilon_m=cfg.get("robot_depth_mask_epsilon_m", 0.02),
            robot_depth_mask_dilation_px=cfg.get("robot_depth_mask_dilation_px", 0),
        )
        capture_time_s = time.perf_counter() - capture_start

        result, reconstruction_time_s = _time_nvblox_reconstruction(
            bundle=bundle,
            scene_spec=scene_spec,
            cfg=cfg,
            robot_ignore_spheres=robot_ignore_spheres,
        )
        ground_truth_boxes = _extra_scene_boxes(scene_spec)
        box_iou = _voxelized_box_iou(
            reconstructed_boxes=result.merged_boxes,
            ground_truth_boxes=ground_truth_boxes,
            workspace_limits=cfg.get("workspace_limits", scene_spec["workspace_limits"]),
            voxel_size=cfg.get("voxel_size", 0.05),
            workspace_padding=cfg.get("workspace_padding", 0.0),
        )

        capture_times.append(capture_time_s)
        reconstruction_times.append(reconstruction_time_s)
        box_ious.append(box_iou)
        rows.append(
            {
                "iteration": iteration,
                "capture_time_s": capture_time_s,
                "reconstruction_time_s": reconstruction_time_s,
                "box_iou": box_iou,
                "box_iou_voxelized": box_iou,
                "n_depth_frames": len(bundle.frames),
                "n_ground_truth_boxes": len(ground_truth_boxes),
                "n_reconstructed_boxes": len(result.merged_boxes),
                "n_occupied_voxels": int(result.metadata["n_occupied_voxels"]),
                "runtime_extra_boxes_json": json.dumps(runtime_extra_boxes),
                "reconstructed_boxes_json": json.dumps(result.merged_boxes),
            }
        )

        print(
            f"iter {iteration + 1:03d}/{iterations}: "
            f"reconstruct={reconstruction_time_s:.4f}s "
            f"capture={capture_time_s:.4f}s "
            f"box_iou={box_iou:.4f} "
            f"boxes={len(result.merged_boxes)}"
        )

    csv_path = os.path.join(results_dir, "nvblox_reconstruction_benchmark_results.csv")
    summary_path = os.path.join(results_dir, "nvblox_reconstruction_benchmark_summary.json")
    _write_csv(rows, csv_path)

    summary = {
        "cfg_path": cfg_path,
        "iterations": iterations,
        "seed": seed,
        "median_reconstruction_time_s": float(np.median(reconstruction_times)),
        "mean_reconstruction_time_s": float(np.mean(reconstruction_times)),
        "std_reconstruction_time_s": float(np.std(reconstruction_times)),
        "median_capture_time_s": float(np.median(capture_times)),
        "median_box_iou": float(np.median(box_ious)),
        "median_box_iou_voxelized": float(np.median(box_ious)),
        "mean_box_iou": float(np.mean(box_ious)),
        "mean_box_iou_voxelized": float(np.mean(box_ious)),
        "min_box_iou_voxelized": float(np.min(box_ious)),
        "max_box_iou_voxelized": float(np.max(box_ious)),
        "voxel_size": float(cfg.get("voxel_size", 0.05)),
        "results_csv": csv_path,
    }
    with open(summary_path, "w") as stream:
        json.dump(summary, stream, indent=2)

    print("\nsummary:")
    print(f"  median_nvblox_reconstruction_time_s: {summary['median_reconstruction_time_s']:.6f}")
    print(f"  median_box_iou_voxelized: {summary['median_box_iou_voxelized']:.6f}")
    print(f"  results_csv: {csv_path}")
    print(f"  summary_json: {summary_path}")


if __name__ == "__main__":
    main()
