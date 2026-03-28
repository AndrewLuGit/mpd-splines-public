from __future__ import annotations

import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def _box_vertices(center, size):
    center = np.asarray(center, dtype=np.float32)
    half = 0.5 * np.asarray(size, dtype=np.float32)
    corners = np.asarray(
        [
            [-1, -1, -1],
            [1, -1, -1],
            [1, 1, -1],
            [-1, 1, -1],
            [-1, -1, 1],
            [1, -1, 1],
            [1, 1, 1],
            [-1, 1, 1],
        ],
        dtype=np.float32,
    )
    return center[None, :] + corners * half[None, :]


def _box_faces(vertices):
    return [
        [vertices[0], vertices[1], vertices[2], vertices[3]],
        [vertices[4], vertices[5], vertices[6], vertices[7]],
        [vertices[0], vertices[1], vertices[5], vertices[4]],
        [vertices[2], vertices[3], vertices[7], vertices[6]],
        [vertices[1], vertices[2], vertices[6], vertices[5]],
        [vertices[4], vertices[7], vertices[3], vertices[0]],
    ]


def _add_box_collection(ax, box_specs, facecolor, edgecolor, alpha, linewidth=0.7, label=None):
    if not box_specs:
        return

    all_faces = []
    for box_spec in box_specs:
        vertices = _box_vertices(box_spec["center"], box_spec["size"])
        all_faces.extend(_box_faces(vertices))

    collection = Poly3DCollection(
        all_faces,
        facecolors=facecolor,
        edgecolors=edgecolor,
        linewidths=linewidth,
        alpha=alpha,
        label=label,
    )
    ax.add_collection3d(collection)


def _sphere_surface(center, radius, n_u=16, n_v=10):
    center = np.asarray(center, dtype=np.float32)
    radius = float(radius)
    u = np.linspace(0.0, 2.0 * np.pi, n_u)
    v = np.linspace(0.0, np.pi, n_v)
    x = radius * np.outer(np.cos(u), np.sin(v)) + center[0]
    y = radius * np.outer(np.sin(u), np.sin(v)) + center[1]
    z = radius * np.outer(np.ones_like(u), np.cos(v)) + center[2]
    return x, y, z


def _add_spheres(ax, sphere_specs, color="green", alpha=0.16, linewidth=0.35):
    if not sphere_specs:
        return

    for sphere_spec in sphere_specs:
        x, y, z = _sphere_surface(sphere_spec["center"], sphere_spec["radius"])
        ax.plot_surface(
            x,
            y,
            z,
            color=color,
            alpha=alpha,
            linewidth=linewidth,
            edgecolor=color,
            shade=False,
        )


def _set_axes_equal(ax, points):
    points = np.asarray(points, dtype=np.float32)
    if points.size == 0:
        return

    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = 0.5 * (mins + maxs)
    radius = 0.5 * np.max(maxs - mins)
    radius = max(radius, 0.25)

    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def render_scene_boxes_debug(
    scene_spec,
    reconstructed_boxes,
    occupied_voxel_centers=None,
    robot_ignore_spheres=None,
    save_path=None,
    show=False,
    title="Phase 3 Scene Debug",
):
    fixed_boxes = [box for box in scene_spec.get("boxes", []) if box.get("source") == "fixed"]
    extra_boxes = [box for box in scene_spec.get("boxes", []) if box.get("source") == "extra"]

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")

    _add_box_collection(
        ax,
        fixed_boxes,
        facecolor=(0.7, 0.7, 0.7, 0.12),
        edgecolor=(0.45, 0.45, 0.45, 0.35),
        alpha=0.12,
        linewidth=0.5,
        label="fixed scene",
    )
    _add_box_collection(
        ax,
        extra_boxes,
        facecolor=(0.2, 0.45, 0.95, 0.20),
        edgecolor=(0.1, 0.25, 0.7, 0.55),
        alpha=0.20,
        linewidth=0.8,
        label="phase2 extra boxes",
    )
    _add_box_collection(
        ax,
        reconstructed_boxes,
        facecolor=(0.95, 0.3, 0.2, 0.10),
        edgecolor=(0.8, 0.1, 0.05, 0.95),
        alpha=0.10,
        linewidth=1.2,
        label="phase3 reconstructed boxes",
    )

    occupied_voxel_centers = np.asarray(occupied_voxel_centers if occupied_voxel_centers is not None else [], dtype=np.float32)
    if occupied_voxel_centers.size > 0:
        ax.scatter(
            occupied_voxel_centers[:, 0],
            occupied_voxel_centers[:, 1],
            occupied_voxel_centers[:, 2],
            s=2,
            c="darkred",
            alpha=0.18,
            depthshade=False,
            label="occupied voxels",
        )

    robot_ignore_spheres = robot_ignore_spheres or []
    if robot_ignore_spheres:
        _add_spheres(ax, robot_ignore_spheres, color="green", alpha=0.16, linewidth=0.25)
        robot_centers = np.asarray([sphere["center"] for sphere in robot_ignore_spheres], dtype=np.float32)
        ax.scatter(
            robot_centers[:, 0],
            robot_centers[:, 1],
            robot_centers[:, 2],
            s=4,
            c="green",
            alpha=0.35,
            depthshade=False,
        )

    points_for_bounds = []
    for box_group in (fixed_boxes, extra_boxes, reconstructed_boxes):
        for box_spec in box_group:
            points_for_bounds.append(_box_vertices(box_spec["center"], box_spec["size"]))
    if occupied_voxel_centers.size > 0:
        points_for_bounds.append(occupied_voxel_centers)
    if robot_ignore_spheres:
        robot_bounds = []
        for sphere_spec in robot_ignore_spheres:
            center = np.asarray(sphere_spec["center"], dtype=np.float32)
            radius = float(sphere_spec["radius"])
            robot_bounds.append(center - radius)
            robot_bounds.append(center + radius)
        points_for_bounds.append(np.asarray(robot_bounds, dtype=np.float32))
    if points_for_bounds:
        _set_axes_equal(ax, np.concatenate(points_for_bounds, axis=0))

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.view_init(elev=24, azim=-60)
    ax.set_title(title)

    if save_path is not None:
        fig.savefig(save_path, dpi=180, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)
