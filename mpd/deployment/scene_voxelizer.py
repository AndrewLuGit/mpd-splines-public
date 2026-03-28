from __future__ import annotations

import os
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage


@dataclass
class VoxelGrid:
    centers: np.ndarray
    grid_shape: tuple[int, int, int]
    origin: np.ndarray
    voxel_size: float
    axis_coordinates: tuple[np.ndarray, np.ndarray, np.ndarray]


def make_voxel_grid(workspace_limits, voxel_size, padding=0.0):
    workspace_limits = np.asarray(workspace_limits, dtype=np.float32)
    lower = workspace_limits[0] - float(padding)
    upper = workspace_limits[1] + float(padding)
    half = 0.5 * float(voxel_size)

    axes = []
    for axis_idx in range(3):
        start = lower[axis_idx] + half
        stop = upper[axis_idx]
        axis = np.arange(start, stop, float(voxel_size), dtype=np.float32)
        if axis.size == 0:
            axis = np.asarray([start], dtype=np.float32)
        axes.append(axis)

    xs, ys, zs = axes
    grid = np.meshgrid(xs, ys, zs, indexing="ij")
    centers = np.stack(grid, axis=-1).reshape(-1, 3)
    origin = np.asarray([xs[0], ys[0], zs[0]], dtype=np.float32)
    return VoxelGrid(
        centers=centers,
        grid_shape=(len(xs), len(ys), len(zs)),
        origin=origin,
        voxel_size=float(voxel_size),
        axis_coordinates=(xs, ys, zs),
    )


def mask_points_in_boxes(points, box_specs):
    points = np.asarray(points, dtype=np.float32)
    if points.size == 0 or not box_specs:
        return np.zeros((points.shape[0],), dtype=bool)

    inside_any = np.zeros((points.shape[0],), dtype=bool)
    for box_spec in box_specs:
        center = np.asarray(box_spec["center"], dtype=np.float32)
        half_size = 0.5 * np.asarray(box_spec["size"], dtype=np.float32)
        pose_world = box_spec.get("pose_world")
        if pose_world is None:
            local_points = points - center[None, :]
        else:
            pose_world = np.asarray(pose_world, dtype=np.float32)
            rotation = pose_world[:3, :3]
            translation = pose_world[:3, 3]
            local_points = (points - translation[None, :]) @ rotation

        inside_box = np.all(np.abs(local_points) <= (half_size[None, :] + 1e-6), axis=1)
        inside_any |= inside_box

    return inside_any


def mask_points_in_spheres(points, sphere_specs):
    points = np.asarray(points, dtype=np.float32)
    if points.size == 0 or not sphere_specs:
        return np.zeros((points.shape[0],), dtype=bool)

    inside_any = np.zeros((points.shape[0],), dtype=bool)
    for sphere_spec in sphere_specs:
        center = np.asarray(sphere_spec["center"], dtype=np.float32)
        radius = float(sphere_spec["radius"])
        distances = np.linalg.norm(points - center[None, :], axis=1)
        inside_any |= distances <= (radius + 1e-6)

    return inside_any


def inflate_box_specs(box_specs, margin):
    if not box_specs or float(margin) == 0.0:
        return list(box_specs or [])

    margin = float(margin)
    inflated_boxes = []
    for box_spec in box_specs:
        inflated_box = dict(box_spec)
        size = np.asarray(box_spec["size"], dtype=np.float32) + 2.0 * margin
        size = np.maximum(size, 1e-6)
        inflated_box["size"] = size.astype(np.float32).tolist()
        inflated_boxes.append(inflated_box)
    return inflated_boxes


def inflate_sphere_specs(sphere_specs, margin):
    if not sphere_specs or float(margin) == 0.0:
        return list(sphere_specs or [])

    margin = float(margin)
    inflated_spheres = []
    for sphere_spec in sphere_specs:
        inflated_sphere = dict(sphere_spec)
        inflated_sphere["radius"] = max(float(sphere_spec["radius"]) + margin, 1e-6)
        inflated_spheres.append(inflated_sphere)
    return inflated_spheres


def occupancy_mask_to_boxes(
    occupancy_mask,
    origin,
    voxel_size,
    min_component_voxels=1,
    max_boxes=None,
    name_prefix="nvblox_box",
    merge_strategy="greedy_cuboids",
):
    occupancy_mask = np.asarray(occupancy_mask, dtype=bool)
    if occupancy_mask.ndim != 3:
        raise ValueError(f"occupancy_mask must be 3D, got shape {occupancy_mask.shape}")

    if merge_strategy == "component_aabb":
        structure = ndimage.generate_binary_structure(rank=3, connectivity=1)
        labels, n_components = ndimage.label(occupancy_mask, structure=structure)
        slices_per_component = ndimage.find_objects(labels)

        boxes = []
        for component_idx, component_slices in enumerate(slices_per_component, start=1):
            if component_slices is None:
                continue

            component_mask = labels[component_slices] == component_idx
            component_size = int(component_mask.sum())
            if component_size < int(min_component_voxels):
                continue

            min_indices = np.asarray([axis_slice.start for axis_slice in component_slices], dtype=np.int32)
            max_indices = np.asarray([axis_slice.stop - 1 for axis_slice in component_slices], dtype=np.int32)

            center = origin + ((min_indices + max_indices) * 0.5) * float(voxel_size)
            size = (max_indices - min_indices + 1).astype(np.float32) * float(voxel_size)
            boxes.append(
                {
                    "name": f"{name_prefix}_{len(boxes)}",
                    "center": center.astype(np.float32).tolist(),
                    "size": size.astype(np.float32).tolist(),
                    "n_component_voxels": component_size,
                }
            )

            if max_boxes is not None and len(boxes) >= int(max_boxes):
                break

        return boxes

    if merge_strategy != "greedy_cuboids":
        raise ValueError(f"Unsupported merge_strategy: {merge_strategy}")

    work_mask = occupancy_mask.copy()
    boxes = []
    while np.any(work_mask):
        start = np.argwhere(work_mask)[0]
        x0, y0, z0 = [int(v) for v in start]

        x1 = x0
        while x1 + 1 < work_mask.shape[0] and work_mask[x1 + 1, y0, z0]:
            x1 += 1

        y1 = y0
        while y1 + 1 < work_mask.shape[1] and np.all(work_mask[x0 : x1 + 1, y1 + 1, z0]):
            y1 += 1

        z1 = z0
        while z1 + 1 < work_mask.shape[2] and np.all(work_mask[x0 : x1 + 1, y0 : y1 + 1, z1 + 1]):
            z1 += 1

        component_size = int(work_mask[x0 : x1 + 1, y0 : y1 + 1, z0 : z1 + 1].sum())
        work_mask[x0 : x1 + 1, y0 : y1 + 1, z0 : z1 + 1] = False
        if component_size < int(min_component_voxels):
            continue

        min_indices = np.asarray([x0, y0, z0], dtype=np.int32)
        max_indices = np.asarray([x1, y1, z1], dtype=np.int32)
        center = origin + ((min_indices + max_indices) * 0.5) * float(voxel_size)
        size = (max_indices - min_indices + 1).astype(np.float32) * float(voxel_size)
        boxes.append(
            {
                "name": f"{name_prefix}_{len(boxes)}",
                "center": center.astype(np.float32).tolist(),
                "size": size.astype(np.float32).tolist(),
                "n_component_voxels": component_size,
            }
        )

        if max_boxes is not None and len(boxes) >= int(max_boxes):
            break

    return boxes


def voxel_centers_to_occupancy_mask(voxel_centers, voxel_grid: VoxelGrid):
    voxel_centers = np.asarray(voxel_centers, dtype=np.float32)
    occupancy_mask = np.zeros(voxel_grid.grid_shape, dtype=bool)
    if voxel_centers.size == 0:
        return occupancy_mask

    indices = np.rint((voxel_centers - voxel_grid.origin[None, :]) / float(voxel_grid.voxel_size)).astype(np.int32)
    valid = np.all(indices >= 0, axis=1)
    valid &= indices[:, 0] < voxel_grid.grid_shape[0]
    valid &= indices[:, 1] < voxel_grid.grid_shape[1]
    valid &= indices[:, 2] < voxel_grid.grid_shape[2]
    indices = indices[valid]
    if indices.size == 0:
        return occupancy_mask

    occupancy_mask[indices[:, 0], indices[:, 1], indices[:, 2]] = True
    return occupancy_mask


def save_occupancy_projections(occupancy_mask, output_dir, prefix="occupancy"):
    occupancy_mask = np.asarray(occupancy_mask, dtype=bool)
    os.makedirs(output_dir, exist_ok=True)

    projections = {
        "xy_max": occupancy_mask.max(axis=2),
        "xz_max": occupancy_mask.max(axis=1),
        "yz_max": occupancy_mask.max(axis=0),
    }

    saved_paths = []
    for name, image in projections.items():
        path = os.path.join(output_dir, f"{prefix}_{name}.png")
        plt.imsave(path, image.astype(np.float32), cmap="gray", vmin=0.0, vmax=1.0)
        saved_paths.append(path)
    return saved_paths
