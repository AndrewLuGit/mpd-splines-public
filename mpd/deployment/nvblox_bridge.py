from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import torch
import yaml

from mpd.deployment.sapien_depth_adapter import DepthCaptureBundle
from mpd.deployment.scene_voxelizer import (
    inflate_box_specs,
    inflate_sphere_specs,
    make_voxel_grid,
    mask_points_in_boxes,
    mask_points_in_spheres,
    occupancy_mask_to_boxes,
    voxel_centers_to_occupancy_mask,
)


@dataclass
class NvbloxReconstructionResult:
    occupied_voxel_centers: np.ndarray
    occupancy_mask: np.ndarray
    occupancy_values: np.ndarray
    voxel_grid_origin: np.ndarray
    voxel_grid_shape: tuple[int, int, int]
    voxel_size: float
    merged_boxes: list[dict]
    robot_ignore_spheres: list[dict]
    metadata: dict
    mapper: object | None = None


def require_nvblox_torch():
    try:
        from nvblox_torch.mapper import Mapper, MapperParams, ProjectiveIntegratorType, QueryType
        from nvblox_torch.sensor import Sensor

        return Mapper, MapperParams, ProjectiveIntegratorType, QueryType, Sensor
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "nvblox_torch is not installed in this environment. Phase 3 requires the nvblox_torch package."
        ) from exc


def load_yaml(path):
    with open(path, "r") as stream:
        return yaml.safe_load(stream)


def _set_nested_mapper_params(mapper_params, mapper_params_cfg):
    if not mapper_params_cfg:
        return mapper_params

    for section_name, section_overrides in mapper_params_cfg.items():
        getter = getattr(mapper_params, f"get_{section_name}", None)
        setter = getattr(mapper_params, f"set_{section_name}", None)
        if getter is None or setter is None:
            raise ValueError(f"Unknown MapperParams section: {section_name}")

        section = getter()
        for key, value in section_overrides.items():
            setattr(section, key, value)
        setter(section)

    return mapper_params


def _integrator_type_from_string(integrator_type_str, projective_integrator_type_cls):
    integrator_type_str = str(integrator_type_str).lower()
    if integrator_type_str == "occupancy":
        return projective_integrator_type_cls.OCCUPANCY
    if integrator_type_str == "tsdf":
        return projective_integrator_type_cls.TSDF
    raise ValueError(f"Unsupported integrator_type: {integrator_type_str}")


def build_mapper(
    voxel_size,
    integrator_type="occupancy",
    mapper_params_cfg=None,
):
    Mapper, MapperParams, ProjectiveIntegratorType, _, _ = require_nvblox_torch()
    mapper_params = _set_nested_mapper_params(MapperParams(), mapper_params_cfg)
    return Mapper(
        float(voxel_size),
        _integrator_type_from_string(integrator_type, ProjectiveIntegratorType),
        mapper_parameters=mapper_params,
    )


def convert_pose_world_to_nvblox_camera(pose_world, pose_convention="sapien"):
    pose_world = np.asarray(pose_world, dtype=np.float32)

    if pose_convention in {"nvblox", "optical", "opencv_optical"}:
        return pose_world

    if pose_convention in {"sapien", "sapien_x_forward_y_left_z_up"}:
        # Convert SAPIEN camera body frame (x forward, y left, z up) to the optical camera
        # frame expected by nvblox/OpenCV-style pinhole projection (x right, y down, z forward).
        body_T_optical = np.eye(4, dtype=np.float32)
        body_T_optical[:3, :3] = np.asarray(
            [
                [0.0, 0.0, 1.0],
                [-1.0, 0.0, 0.0],
                [0.0, -1.0, 0.0],
            ],
            dtype=np.float32,
        )
        return pose_world @ body_T_optical

    raise ValueError(
        f"Unsupported pose_convention '{pose_convention}'. "
        "Expected one of: sapien, sapien_x_forward_y_left_z_up, optical, opencv_optical, nvblox"
    )


def integrate_depth_bundle(
    mapper,
    bundle: DepthCaptureBundle,
    device="cuda",
    pose_convention="sapien",
):
    _, _, _, _, Sensor = require_nvblox_torch()

    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "nvblox_torch depth integration requires CUDA tensors, but torch.cuda.is_available() is False "
            "in this runtime."
        )

    for frame in bundle.frames:
        depth_meters = np.asarray(frame.depth_meters, dtype=np.float32)
        depth_meters = np.where(np.isfinite(depth_meters) & (depth_meters > 0.0), depth_meters, 0.0)

        depth_tensor = torch.from_numpy(depth_meters).to(device=device, dtype=torch.float32)
        pose_world_nvblox = convert_pose_world_to_nvblox_camera(
            frame.pose_world,
            pose_convention=pose_convention,
        )
        pose_world = torch.from_numpy(np.asarray(pose_world_nvblox, dtype=np.float32)).to(dtype=torch.float32)
        intrinsic_matrix = torch.from_numpy(np.asarray(frame.intrinsic_matrix, dtype=np.float32)).to(
            dtype=torch.float32
        )

        sensor = Sensor.from_camera_matrix(
            intrinsic_matrix,
            width=depth_meters.shape[1],
            height=depth_meters.shape[0],
        )
        mapper.add_depth_frame(depth_tensor, pose_world, sensor, mapper_id=0)

    return mapper


def _extract_tsdf_voxels_sparse(mapper):
    tsdf_layer = mapper.tsdf_layer_view(mapper_id=0)
    values, points = tsdf_layer.get_tsdfs_below_zero()
    points_np = points.detach().cpu().numpy().astype(np.float32)
    values_np = values.detach().cpu().numpy().astype(np.float32)
    return points_np, values_np


def _query_mapper_in_chunks(mapper, query_points, query_type, mapper_id, query_chunk_size, device):
    _, _, _, QueryType, _ = require_nvblox_torch()

    outputs = []
    for start_idx in range(0, query_points.shape[0], int(query_chunk_size)):
        stop_idx = min(start_idx + int(query_chunk_size), query_points.shape[0])
        chunk = torch.from_numpy(query_points[start_idx:stop_idx]).to(device=device, dtype=torch.float32)
        outputs.append(mapper.query_layer(query_type, chunk, mapper_id=mapper_id).detach().cpu().numpy())
    return np.concatenate(outputs, axis=0) if outputs else np.zeros((0, 1), dtype=np.float32)


def build_robot_ignore_spheres(robot_cfg, robot_sphere_margin=0.0):
    if not robot_cfg or not robot_cfg.get("enabled", False):
        return []

    from torch_robotics.robots import RobotPanda

    tensor_args = {"device": torch.device("cpu"), "dtype": torch.float32}
    robot = RobotPanda(gripper=bool(robot_cfg.get("gripper", True)), tensor_args=tensor_args)
    try:
        qpos = robot_cfg.get("qpos")
        if qpos is None:
            return []

        qpos = torch.tensor(qpos, dtype=torch.float32, device="cpu")
        centers = robot.fk_map_collision(qpos).squeeze(0).detach().cpu().numpy().astype(np.float32)
        radii = robot.link_collision_spheres_radii.detach().cpu().numpy().astype(np.float32).reshape(-1)
        sphere_specs = [
            {
                "name": f"robot_sphere_{idx}",
                "center": centers[idx].tolist(),
                "radius": float(radii[idx]),
            }
            for idx in range(centers.shape[0])
        ]
        return inflate_sphere_specs(sphere_specs, robot_sphere_margin)
    finally:
        robot.cleanup()


def reconstruct_occupancy_from_bundle(
    bundle,
    workspace_limits,
    voxel_size,
    integrator_type="occupancy",
    mapper_params_cfg=None,
    device="cuda",
    pose_convention="sapien",
    occupancy_threshold_log_odds=0.0,
    tsdf_weight_threshold=1e-4,
    tsdf_distance_threshold=0.0,
    query_chunk_size=250000,
    workspace_padding=0.0,
    ignore_boxes=None,
    ignore_box_margin=0.0,
    ignore_scene_box_sources=None,
    ignore_scene_box_margin=0.0,
    scene_spec=None,
    robot_ignore_spheres=None,
    robot_sphere_margin=0.0,
    inflate_robot_mask_by_voxel_extent=True,
    update_esdf=False,
    keep_mapper=False,
    min_component_voxels=1,
    max_boxes=None,
    extraction_method=None,
    merge_strategy="greedy_cuboids",
):
    _, _, _, QueryType, _ = require_nvblox_torch()

    integrator_type = str(integrator_type).lower()
    if extraction_method is None:
        extraction_method = "tsdf_sparse" if integrator_type == "tsdf" else "occupancy_query"

    voxel_grid = make_voxel_grid(workspace_limits, voxel_size=voxel_size, padding=workspace_padding)
    mapper = build_mapper(
        voxel_size=voxel_size,
        integrator_type=integrator_type,
        mapper_params_cfg=mapper_params_cfg,
    )
    mapper = integrate_depth_bundle(
        mapper,
        bundle=bundle,
        device=device,
        pose_convention=pose_convention,
    )
    if update_esdf:
        mapper.update_esdf(mapper_id=0)

    if extraction_method == "tsdf_sparse":
        if integrator_type != "tsdf":
            raise ValueError("extraction_method='tsdf_sparse' requires integrator_type='tsdf'")
        occupied_centers, occupancy_values = _extract_tsdf_voxels_sparse(mapper)
        occupied_flat = np.ones((occupied_centers.shape[0],), dtype=bool)
    elif extraction_method == "occupancy_query":
        occupied_centers = voxel_grid.centers
        occupancy_values = _query_mapper_in_chunks(
            mapper=mapper,
            query_points=voxel_grid.centers,
            query_type=QueryType.OCCUPANCY,
            mapper_id=-1,
            query_chunk_size=query_chunk_size,
            device=device,
        )
        occupied_flat = occupancy_values[:, 0] > float(occupancy_threshold_log_odds)
    elif extraction_method == "tsdf_query":
        occupied_centers = voxel_grid.centers
        occupancy_values = _query_mapper_in_chunks(
            mapper=mapper,
            query_points=voxel_grid.centers,
            query_type=QueryType.TSDF,
            mapper_id=0,
            query_chunk_size=query_chunk_size,
            device=device,
        )
        occupied_flat = (occupancy_values[:, 1] > float(tsdf_weight_threshold)) & (
            occupancy_values[:, 0] <= float(tsdf_distance_threshold)
        )
    else:
        raise ValueError(
            f"Unsupported extraction_method '{extraction_method}'. "
            "Expected one of: tsdf_sparse, tsdf_query, occupancy_query"
        )

    ignore_box_list = inflate_box_specs(ignore_boxes or [], ignore_box_margin)
    if scene_spec is not None and ignore_scene_box_sources:
        scene_ignore_boxes = [
            box_spec for box_spec in scene_spec.get("boxes", []) if box_spec.get("source") in ignore_scene_box_sources
        ]
        ignore_box_list.extend(
            inflate_box_specs(
                scene_ignore_boxes,
                ignore_scene_box_margin,
            )
        )

    if ignore_box_list:
        ignore_mask = mask_points_in_boxes(occupied_centers, ignore_box_list)
        occupied_flat &= ~ignore_mask

    robot_ignore_spheres = inflate_sphere_specs(robot_ignore_spheres or [], robot_sphere_margin)
    robot_mask_voxel_extent_margin = 0.0
    robot_mask_spheres = robot_ignore_spheres
    if robot_ignore_spheres and inflate_robot_mask_by_voxel_extent:
        # Remove any voxel whose occupied cube could intersect a robot sphere, not just
        # voxels whose centers fall inside it.
        robot_mask_voxel_extent_margin = 0.5 * np.sqrt(3.0) * float(voxel_size)
        robot_mask_spheres = inflate_sphere_specs(robot_ignore_spheres, robot_mask_voxel_extent_margin)

    if robot_mask_spheres:
        robot_ignore_mask = mask_points_in_spheres(occupied_centers, robot_mask_spheres)
        occupied_flat &= ~robot_ignore_mask

    occupied_voxel_centers = occupied_centers[occupied_flat]
    occupancy_mask = voxel_centers_to_occupancy_mask(occupied_voxel_centers, voxel_grid)
    merged_boxes = occupancy_mask_to_boxes(
        occupancy_mask,
        origin=voxel_grid.origin,
        voxel_size=voxel_grid.voxel_size,
        min_component_voxels=min_component_voxels,
        max_boxes=max_boxes,
        merge_strategy=merge_strategy,
    )

    return NvbloxReconstructionResult(
        occupied_voxel_centers=occupied_voxel_centers,
        occupancy_mask=occupancy_mask,
        occupancy_values=occupancy_values,
        voxel_grid_origin=voxel_grid.origin,
        voxel_grid_shape=voxel_grid.grid_shape,
        voxel_size=voxel_grid.voxel_size,
        merged_boxes=merged_boxes,
        robot_ignore_spheres=robot_ignore_spheres,
        metadata={
            "integrator_type": integrator_type,
            "extraction_method": extraction_method,
            "n_frames": len(bundle.frames),
            "n_query_voxels": int(voxel_grid.centers.shape[0]) if extraction_method != "tsdf_sparse" else None,
            "n_occupied_voxels": int(occupied_voxel_centers.shape[0]),
            "query_chunk_size": int(query_chunk_size),
            "device": device,
            "pose_convention": pose_convention,
            "ignore_box_margin": float(ignore_box_margin),
            "ignore_scene_box_margin": float(ignore_scene_box_margin),
            "robot_sphere_margin": float(robot_sphere_margin),
            "inflate_robot_mask_by_voxel_extent": bool(inflate_robot_mask_by_voxel_extent),
            "robot_mask_voxel_extent_margin": float(robot_mask_voxel_extent_margin),
            "update_esdf": bool(update_esdf),
            "merge_strategy": merge_strategy,
        },
        mapper=mapper if keep_mapper else None,
    )


def save_reconstruction_outputs(
    result,
    results_dir,
    boxes_filename="phase3_boxes.yaml",
    summary_filename="phase3_summary.yaml",
    voxels_filename="phase3_occupied_voxels.npz",
):
    os.makedirs(results_dir, exist_ok=True)

    boxes_path = os.path.join(results_dir, boxes_filename)
    summary_path = os.path.join(results_dir, summary_filename)
    voxels_path = os.path.join(results_dir, voxels_filename)

    with open(boxes_path, "w") as stream:
        yaml.dump({"extra_boxes": result.merged_boxes}, stream, Dumper=yaml.Dumper, allow_unicode=True)

    summary = {
        **result.metadata,
        "voxel_size": float(result.voxel_size),
        "voxel_grid_origin": np.asarray(result.voxel_grid_origin, dtype=np.float32).tolist(),
        "voxel_grid_shape": list(result.voxel_grid_shape),
        "n_merged_boxes": len(result.merged_boxes),
    }
    with open(summary_path, "w") as stream:
        yaml.dump(summary, stream, Dumper=yaml.Dumper, allow_unicode=True)

    np.savez_compressed(
        voxels_path,
        occupied_voxel_centers=np.asarray(result.occupied_voxel_centers, dtype=np.float32),
        occupancy_mask=np.asarray(result.occupancy_mask, dtype=np.uint8),
        occupancy_values=np.asarray(result.occupancy_values, dtype=np.float32),
        voxel_grid_origin=np.asarray(result.voxel_grid_origin, dtype=np.float32),
        voxel_grid_shape=np.asarray(result.voxel_grid_shape, dtype=np.int32),
        voxel_size=np.asarray(result.voxel_size, dtype=np.float32),
    )

    return {
        "boxes_path": boxes_path,
        "summary_path": summary_path,
        "voxels_path": voxels_path,
    }
