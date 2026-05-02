import os
import sys

import numpy as np

REPO_PATH_BOOTSTRAP = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_PATH_BOOTSTRAP not in sys.path:
    sys.path.insert(0, REPO_PATH_BOOTSTRAP)
TORCH_ROBOTICS_PATH = os.path.join(REPO_PATH_BOOTSTRAP, "mpd", "torch_robotics")
if TORCH_ROBOTICS_PATH not in sys.path:
    sys.path.insert(0, TORCH_ROBOTICS_PATH)


def _numpy_monkey_patch():
    if not hasattr(np, "int"):
        np.int = int
    if not hasattr(np, "float"):
        np.float = float
    if not hasattr(np, "double"):
        np.double = float


_numpy_monkey_patch()

import argparse
import threading
import time
import traceback
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
os.environ.setdefault("MPLBACKEND", "Agg")

import viser
import yaml
from scipy.spatial.transform import Rotation
from viser.extras import ViserUrdf

from mpd.deployment.nvblox_bridge import (
    NvbloxReconstructionResult,
    build_robot_ignore_spheres,
    build_mapper,
    integrate_depth_bundle,
    require_nvblox_torch,
    save_reconstruction_outputs,
    _extract_tsdf_voxels_sparse,
    _query_mapper_in_chunks,
)
from mpd.deployment.sapien_depth_adapter import (
    DepthCaptureBundle,
    DepthFrame,
    build_capture_request,
    build_warehouse_scene_spec,
    parse_camera_specs,
    save_depth_capture_bundle,
    save_depth_preview_images,
    save_yaml,
    _apply_robot_depth_mask,
    _build_box_actor,
    _create_scene_and_capturers,
    _default_panda_urdf_path,
    _depth_from_position_picture,
    _matrix_to_sapien_pose,
)
from mpd.deployment.scene_primitives import build_object_fields_from_boxes
from mpd.deployment.scene_voxelizer import (
    inflate_box_specs,
    inflate_sphere_specs,
    make_voxel_grid,
    mask_points_in_boxes,
    mask_points_in_spheres,
    occupancy_mask_to_boxes,
    save_occupancy_projections,
    voxel_centers_to_occupancy_mask,
)
from mpd.paths import REPO_PATH
from torch_robotics.torch_utils.seed import fix_random_seed


def _resolve_repo_path(path):
    if path is None:
        return None
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


def _matrix_to_viser_wxyz_and_position(pose_world):
    pose_world = np.asarray(pose_world, dtype=np.float32)
    quat_xyzw = Rotation.from_matrix(pose_world[:3, :3]).as_quat()
    wxyz = np.asarray([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=np.float32)
    return wxyz, pose_world[:3, 3].astype(np.float32)


def _format_seconds(seconds):
    if seconds is None:
        return "-"
    return f"{seconds:.3f} s"


def _remove_scene_node_if_present(server, path):
    try:
        handle = server.scene._handle_from_node_name.get(path)
        if handle is not None:
            handle.remove()
    except Exception:
        pass


def _clear_reconstruction_viser(server):
    _remove_scene_node_if_present(server, "/reconstruction")


def _save_runtime_state(path, extra_boxes):
    save_yaml({"extra_boxes": list(extra_boxes)}, path)


@dataclass
class RuntimeBoxWidget:
    name: str
    gizmo: object
    visual: object
    enabled: object
    size: object

    def current_box_spec(self):
        if not bool(self.enabled.value):
            return None
        center = np.asarray(self.gizmo.position, dtype=np.float32).reshape(3)
        size = np.asarray(self.size.value, dtype=np.float32).reshape(3)
        return {
            "name": self.name,
            "center": center.tolist(),
            "size": size.tolist(),
        }


def _make_runtime_box_widget(server, idx, box_spec):
    name = str(box_spec.get("name", f"runtime_box_{idx}"))
    center = np.asarray(box_spec.get("center", [0.6, 0.0, 0.2]), dtype=np.float32)
    size = np.asarray(box_spec.get("size", [0.2, 0.2, 0.2]), dtype=np.float32)

    gizmo = server.scene.add_transform_controls(
        f"/runtime_boxes/{name}",
        scale=0.18,
        position=center,
        disable_rotations=True,
    )
    visual = server.scene.add_box(
        f"/runtime_boxes/{name}/visual",
        color=(70, 150, 255),
        dimensions=size,
        opacity=0.55,
    )

    enabled = server.gui.add_checkbox(f"{name} enabled", True)
    size_handle = server.gui.add_vector3(
        f"{name} size",
        tuple(float(v) for v in size),
        min=(0.02, 0.02, 0.02),
        max=(1.5, 1.5, 1.5),
        step=0.01,
    )

    @enabled.on_update
    def _(_event):
        visible = bool(enabled.value)
        gizmo.visible = visible
        visual.visible = visible

    @size_handle.on_update
    def _(_event):
        visual.dimensions = tuple(float(v) for v in size_handle.value)

    return RuntimeBoxWidget(
        name=name,
        gizmo=gizmo,
        visual=visual,
        enabled=enabled,
        size=size_handle,
    )


def _add_fixed_scene_to_viser(server, scene_spec):
    server.scene.set_up_direction("+z")
    server.scene.add_grid(
        "/scene/grid",
        width=3.0,
        height=3.0,
        plane="xy",
        cell_size=0.1,
        section_size=0.5,
        cell_color=(220, 220, 220),
        section_color=(170, 170, 170),
        plane_opacity=0.0,
    )
    for box in scene_spec.get("boxes", []):
        if box.get("source") != "fixed":
            continue
        center = np.asarray(box["center"], dtype=np.float32)
        size = np.asarray(box["size"], dtype=np.float32)
        server.scene.add_box(
            f"/scene/fixed/{box['name']}",
            color=(170, 170, 170),
            dimensions=size,
            opacity=0.25,
            position=center,
        )


def _add_camera_frames_to_viser(server, camera_specs, axes_length=0.12):
    server.scene.add_frame("/cameras", show_axes=False)
    for camera_spec in camera_specs:
        wxyz, position = _matrix_to_viser_wxyz_and_position(camera_spec.resolved_pose_world())
        server.scene.add_frame(
            f"/cameras/{camera_spec.name}",
            position=position,
            wxyz=wxyz,
            axes_length=float(axes_length),
            axes_radius=0.006,
        )
        server.scene.add_label(
            f"/cameras/{camera_spec.name}/label",
            camera_spec.name,
            position=(0.0, 0.0, 0.06),
        )


def _add_robot_model_to_viser(server, robot_cfg, show_collision=False):
    if not robot_cfg or not robot_cfg.get("enabled", False):
        return None

    qpos = robot_cfg.get("qpos")
    if qpos is None:
        return None

    urdf_path = robot_cfg.get("urdf_path", _default_panda_urdf_path())
    if not os.path.isabs(urdf_path):
        urdf_path = os.path.abspath(urdf_path)

    root_pose_world = robot_cfg.get("root_pose_world")
    if root_pose_world is None:
        robot_root = server.scene.add_frame("/robot", show_axes=False)
    else:
        wxyz, position = _matrix_to_viser_wxyz_and_position(root_pose_world)
        robot_root = server.scene.add_frame(
            "/robot",
            show_axes=False,
            wxyz=wxyz,
            position=position,
        )

    viser_urdf = ViserUrdf(
        server,
        urdf_or_path=Path(urdf_path),
        root_node_name="/robot",
        load_meshes=True,
        load_collision_meshes=bool(show_collision),
        collision_mesh_color_override=(1.0, 0.0, 0.0, 0.25),
    )
    viser_urdf.update_cfg(np.asarray(qpos, dtype=np.float32).reshape(-1))
    return robot_root, viser_urdf


def _add_reconstruction_to_viser(server, result, show_voxels=False, max_voxels=10000, voxel_point_size=0.012):
    server.scene.add_frame("/reconstruction", show_axes=False)

    for idx, box in enumerate(result.merged_boxes):
        center = np.asarray(box["center"], dtype=np.float32)
        size = np.asarray(box["size"], dtype=np.float32)
        name = str(box.get("name", f"nvblox_box_{idx}"))
        server.scene.add_box(
            f"/reconstruction/boxes/{name}",
            color=(245, 175, 50),
            dimensions=size,
            opacity=0.42,
            position=center,
        )

    if show_voxels and result.occupied_voxel_centers.size > 0:
        points = np.asarray(result.occupied_voxel_centers, dtype=np.float32)
        if max_voxels is not None and points.shape[0] > int(max_voxels):
            stride = int(np.ceil(points.shape[0] / int(max_voxels)))
            points = points[::stride]
        colors = np.tile(np.asarray([[255, 205, 90]], dtype=np.uint8), (points.shape[0], 1))
        try:
            server.scene.add_point_cloud(
                "/reconstruction/occupied_voxels",
                points=points,
                colors=colors,
                point_size=float(voxel_point_size),
            )
        except Exception:
            pass


def _build_scene_spec(cfg, runtime_extra_boxes):
    extra_object_fields = build_object_fields_from_boxes(runtime_extra_boxes)
    return build_warehouse_scene_spec(
        extra_boxes=extra_object_fields,
        rotation_z_axis_deg=cfg.get("rotation_z_axis_deg", 0.0),
    )


def _box_spec_with_pose(box_spec, source="extra", hidden=False):
    center = np.asarray(box_spec["center"], dtype=np.float32).reshape(3)
    if hidden:
        center = np.asarray([1000.0, 1000.0, 1000.0], dtype=np.float32)

    pose_world = np.eye(4, dtype=np.float32)
    pose_world[:3, 3] = center
    return {
        "name": str(box_spec.get("name", "runtime_box")),
        "center": center.tolist(),
        "size": np.asarray(box_spec["size"], dtype=np.float32).reshape(3).tolist(),
        "pose_world": pose_world.tolist(),
        "source": source,
    }


def _capture_frames_from_reused_scene(scene, capturers, camera_specs, capture_backend):
    scene.step()
    scene.update_render()

    frames = []
    if capture_backend == "render_camera":
        for capturer, camera_spec in zip(capturers, camera_specs):
            capturer.take_picture()
            position_picture = capturer.get_picture("Position")
            frames.append(
                DepthFrame(
                    camera_name=camera_spec.name,
                    depth_meters=_depth_from_position_picture(
                        position_picture,
                        near=float(camera_spec.near),
                        far=float(camera_spec.far),
                    ),
                    intrinsic_matrix=np.asarray(capturer.get_intrinsic_matrix(), dtype=np.float32),
                    pose_world=np.asarray(capturer.entity.pose.to_transformation_matrix(), dtype=np.float32),
                )
            )
        return frames

    if capture_backend == "stereo_depth_sensor":
        for capturer, camera_spec in zip(capturers, camera_specs):
            capturer.take_picture()
            capturer.compute_depth()
            frames.append(
                DepthFrame(
                    camera_name=camera_spec.name,
                    depth_meters=np.asarray(capturer.get_depth(), dtype=np.float32),
                    intrinsic_matrix=np.asarray(capturer.get_config().rgb_intrinsic, dtype=np.float32),
                    pose_world=np.asarray(capturer.get_pose().to_transformation_matrix(), dtype=np.float32),
                )
            )
        return frames

    raise ValueError(
        f"Unsupported capture backend '{capture_backend}'. Expected one of: render_camera, stereo_depth_sensor"
    )


class ReusableSapienDepthCapturer:
    def __init__(self, cfg, camera_specs, fixed_scene_spec, cfg_path):
        self.cfg = cfg
        self.camera_specs = camera_specs
        self.fixed_scene_spec = fixed_scene_spec
        self.cfg_path = cfg_path
        self.capture_backend = cfg.get("capture_backend", "render_camera")
        self.robot_cfg = deepcopy(cfg.get("robot", {}))
        self.runtime_actor_sizes = {}
        self.scene_entries = []

        if self.capture_backend == "stereo_depth_sensor":
            # SAPIEN's stereo sensor installs textured lights internally. Multiple stereo sensors
            # in one scene can exceed its textured-light limit, so keep one reused scene per camera.
            capture_groups = [[camera_spec] for camera_spec in camera_specs]
        else:
            capture_groups = [camera_specs]

        for camera_group in capture_groups:
            sapien, scene, fixed_actors, capturers, robot = _create_scene_and_capturers(
                scene_spec=fixed_scene_spec,
                camera_specs=camera_group,
                add_ground=cfg.get("add_ground", False),
                capture_backend=self.capture_backend,
                stereo_sensor_config_overrides=cfg.get("stereo_sensor_config"),
                robot_cfg=self.robot_cfg,
                include_scene_boxes=True,
            )
            self.scene_entries.append(
                {
                    "sapien": sapien,
                    "scene": scene,
                    "fixed_actors": fixed_actors,
                    "capturers": capturers,
                    "robot": robot,
                    "camera_specs": camera_group,
                    "runtime_actors": {},
                }
            )

        self.robot_only_frames = []
        if cfg.get("mask_robot_from_depth", False) and self.robot_cfg and self.robot_cfg.get("enabled", False):
            (
                self.robot_sapien,
                self.robot_scene,
                self.robot_fixed_actors,
                self.robot_capturers,
                self.robot_only_robot,
            ) = _create_scene_and_capturers(
                scene_spec=fixed_scene_spec,
                camera_specs=camera_specs,
                add_ground=cfg.get("add_ground", False),
                capture_backend="render_camera",
                robot_cfg=self.robot_cfg,
                include_scene_boxes=False,
            )
            self.robot_only_frames = _capture_frames_from_reused_scene(
                scene=self.robot_scene,
                capturers=self.robot_capturers,
                camera_specs=camera_specs,
                capture_backend="render_camera",
            )
        else:
            self.robot_sapien = None
            self.robot_scene = None
            self.robot_fixed_actors = []
            self.robot_capturers = []
            self.robot_only_robot = None

    def _remove_runtime_actor_from_entry(self, entry, name):
        actor = entry["runtime_actors"].pop(name, None)
        if actor is None:
            return
        try:
            entry["scene"].remove_actor(actor)
        except Exception:
            try:
                actor.set_pose(
                    _matrix_to_sapien_pose(
                        entry["sapien"],
                        _box_spec_with_pose(
                            {"name": name, "center": [0.0, 0.0, 0.0], "size": [0.01, 0.01, 0.01]},
                            hidden=True,
                        )["pose_world"],
                    )
                )
            except Exception:
                pass

    def _remove_runtime_actor(self, name):
        for entry in self.scene_entries:
            self._remove_runtime_actor_from_entry(entry, name)
        self.runtime_actor_sizes.pop(name, None)

    def _upsert_runtime_actor(self, box_spec):
        name = str(box_spec.get("name", "runtime_box"))
        size = np.asarray(box_spec["size"], dtype=np.float32).reshape(3)
        old_size = self.runtime_actor_sizes.get(name)
        if old_size is not None and not np.allclose(old_size, size, atol=1e-5):
            self._remove_runtime_actor(name)

        for entry in self.scene_entries:
            actor = entry["runtime_actors"].get(name)
            sapien_box_spec = _box_spec_with_pose(box_spec, source="extra")
            if actor is None:
                actor = _build_box_actor(entry["scene"], entry["sapien"], sapien_box_spec)
                entry["runtime_actors"][name] = actor
            else:
                actor.set_pose(_matrix_to_sapien_pose(entry["sapien"], sapien_box_spec["pose_world"]))
        self.runtime_actor_sizes[name] = size.copy()

    def update_runtime_boxes(self, runtime_extra_boxes):
        enabled_names = {str(box.get("name", f"runtime_box_{idx}")) for idx, box in enumerate(runtime_extra_boxes)}
        for stale_name in list(set(self.runtime_actor_sizes) - enabled_names):
            self._remove_runtime_actor(stale_name)
        for idx, box_spec in enumerate(runtime_extra_boxes):
            box_spec = dict(box_spec)
            box_spec.setdefault("name", f"runtime_box_{idx}")
            self._upsert_runtime_actor(box_spec)

    def capture(self, runtime_extra_boxes):
        self.update_runtime_boxes(runtime_extra_boxes)
        frames = []
        for entry in self.scene_entries:
            frames.extend(
                _capture_frames_from_reused_scene(
                    scene=entry["scene"],
                    capturers=entry["capturers"],
                    camera_specs=entry["camera_specs"],
                    capture_backend=self.capture_backend,
                )
            )

        robot_mask_summary = None
        if self.robot_only_frames:
            robot_frame_map = {frame.camera_name: frame for frame in self.robot_only_frames}
            robot_mask_summary = []
            masked_frames = []
            for frame in frames:
                robot_frame = robot_frame_map.get(frame.camera_name)
                if robot_frame is None:
                    masked_frames.append(frame)
                    continue
                masked_depth, robot_mask = _apply_robot_depth_mask(
                    depth_meters=frame.depth_meters,
                    robot_depth_meters=robot_frame.depth_meters,
                    depth_epsilon_m=self.cfg.get("robot_depth_mask_epsilon_m", 0.02),
                    dilation_px=self.cfg.get("robot_depth_mask_dilation_px", 0),
                )
                robot_mask_summary.append(
                    {
                        "camera_name": frame.camera_name,
                        "n_masked_pixels": int(robot_mask.sum()),
                        "mask_fraction": float(robot_mask.mean()),
                    }
                )
                masked_frames.append(
                    DepthFrame(
                        camera_name=frame.camera_name,
                        depth_meters=masked_depth,
                        intrinsic_matrix=frame.intrinsic_matrix,
                        pose_world=frame.pose_world,
                    )
                )
            frames = masked_frames

        return DepthCaptureBundle(
            frames=frames,
            metadata={
                "config_path": self.cfg_path,
                "capture_backend": self.capture_backend,
                "robot_depth_mask_applied": bool(self.robot_only_frames),
                "robot_depth_mask_backend": "render_camera" if self.robot_only_frames else None,
                "robot_depth_mask_epsilon_m": float(self.cfg.get("robot_depth_mask_epsilon_m", 0.02)),
                "robot_depth_mask_dilation_px": int(self.cfg.get("robot_depth_mask_dilation_px", 0)),
                "robot_depth_mask_summary": robot_mask_summary,
            },
        )

    def close(self):
        for actor_name in list(self.runtime_actor_sizes):
            self._remove_runtime_actor(actor_name)
        for entry in self.scene_entries:
            for key in ["capturers", "robot", "fixed_actors", "scene", "sapien"]:
                entry[key] = None
        self.scene_entries = []
        for attr_name in [
            "robot_capturers",
            "robot_only_robot",
            "robot_fixed_actors",
            "robot_scene",
            "robot_sapien",
        ]:
            if hasattr(self, attr_name):
                try:
                    setattr(self, attr_name, None)
                except Exception:
                    pass


class ReusableNvbloxReconstructor:
    def __init__(self, cfg, scene_spec, robot_ignore_spheres):
        self.cfg = cfg
        self.scene_spec = scene_spec
        self.robot_ignore_spheres = robot_ignore_spheres
        self.integrator_type = str(cfg.get("integrator_type", "occupancy")).lower()
        self.extraction_method = cfg.get("extraction_method")
        if self.extraction_method is None:
            self.extraction_method = "tsdf_sparse" if self.integrator_type == "tsdf" else "occupancy_query"
        self.device = cfg.get("device", "cuda")
        self.voxel_grid = make_voxel_grid(
            cfg.get("workspace_limits", scene_spec["workspace_limits"]),
            voxel_size=cfg.get("voxel_size", 0.05),
            padding=cfg.get("workspace_padding", 0.0),
        )
        self.mapper = build_mapper(
            voxel_size=cfg.get("voxel_size", 0.05),
            integrator_type=self.integrator_type,
            mapper_params_cfg=cfg.get("mapper_params"),
        )
        self.mapper_recreated_count = 0
        self.last_reset_reused_mapper = True

    def _new_mapper(self):
        self.mapper_recreated_count += 1
        self.mapper = build_mapper(
            voxel_size=self.cfg.get("voxel_size", 0.05),
            integrator_type=self.integrator_type,
            mapper_params_cfg=self.cfg.get("mapper_params"),
        )

    def _reset_mapper(self):
        if self.cfg.get("accumulate_map", False):
            self.last_reset_reused_mapper = True
            return

        reset_methods = self.cfg.get(
            "mapper_reset_methods",
            ["clear", "reset", "clear_map", "reset_map", "clear_mapper", "reset_mapper"],
        )
        for method_name in reset_methods:
            method = getattr(self.mapper, method_name, None)
            if method is None:
                continue
            for args, kwargs in [
                ((), {}),
                ((0,), {}),
                ((), {"mapper_id": 0}),
            ]:
                try:
                    method(*args, **kwargs)
                    self.last_reset_reused_mapper = True
                    return
                except TypeError:
                    continue
                except Exception:
                    continue

        self.last_reset_reused_mapper = False
        self._new_mapper()

    def reconstruct(self, bundle):
        _, _, _, QueryType, _ = require_nvblox_torch()
        self._reset_mapper()
        integrate_depth_bundle(
            self.mapper,
            bundle=bundle,
            device=self.device,
            pose_convention=self.cfg.get("pose_convention", "sapien"),
        )
        if self.cfg.get("update_esdf", False):
            self.mapper.update_esdf(mapper_id=0)

        if self.extraction_method == "tsdf_sparse":
            if self.integrator_type != "tsdf":
                raise ValueError("extraction_method='tsdf_sparse' requires integrator_type='tsdf'")
            occupied_centers, occupancy_values = _extract_tsdf_voxels_sparse(self.mapper)
            occupied_flat = np.ones((occupied_centers.shape[0],), dtype=bool)
        elif self.extraction_method == "occupancy_query":
            occupied_centers = self.voxel_grid.centers
            occupancy_values = _query_mapper_in_chunks(
                mapper=self.mapper,
                query_points=self.voxel_grid.centers,
                query_type=QueryType.OCCUPANCY,
                mapper_id=-1,
                query_chunk_size=self.cfg.get("query_chunk_size", 250000),
                device=self.device,
            )
            occupied_flat = occupancy_values[:, 0] > float(self.cfg.get("occupancy_threshold_log_odds", 0.0))
        elif self.extraction_method == "tsdf_query":
            occupied_centers = self.voxel_grid.centers
            occupancy_values = _query_mapper_in_chunks(
                mapper=self.mapper,
                query_points=self.voxel_grid.centers,
                query_type=QueryType.TSDF,
                mapper_id=0,
                query_chunk_size=self.cfg.get("query_chunk_size", 250000),
                device=self.device,
            )
            occupied_flat = (occupancy_values[:, 1] > float(self.cfg.get("tsdf_weight_threshold", 1e-4))) & (
                occupancy_values[:, 0] <= float(self.cfg.get("tsdf_distance_threshold", 0.0))
            )
        else:
            raise ValueError(
                f"Unsupported extraction_method '{self.extraction_method}'. "
                "Expected one of: tsdf_sparse, tsdf_query, occupancy_query"
            )

        ignore_box_list = inflate_box_specs(self.cfg.get("ignore_boxes") or [], self.cfg.get("ignore_box_margin", 0.0))
        ignore_scene_box_sources = self.cfg.get("ignore_scene_box_sources")
        if ignore_scene_box_sources:
            scene_ignore_boxes = [
                box_spec
                for box_spec in self.scene_spec.get("boxes", [])
                if box_spec.get("source") in ignore_scene_box_sources
            ]
            ignore_box_list.extend(
                inflate_box_specs(
                    scene_ignore_boxes,
                    self.cfg.get("ignore_scene_box_margin", 0.0),
                )
            )

        if ignore_box_list:
            occupied_flat &= ~mask_points_in_boxes(occupied_centers, ignore_box_list)

        robot_ignore_spheres = inflate_sphere_specs(
            self.robot_ignore_spheres or [],
            self.cfg.get("robot_sphere_margin", 0.0),
        )
        robot_mask_voxel_extent_margin = 0.0
        robot_mask_spheres = robot_ignore_spheres
        if robot_ignore_spheres and self.cfg.get("inflate_robot_mask_by_voxel_extent", True):
            robot_mask_voxel_extent_margin = 0.5 * np.sqrt(3.0) * float(self.voxel_grid.voxel_size)
            robot_mask_spheres = inflate_sphere_specs(robot_ignore_spheres, robot_mask_voxel_extent_margin)

        if robot_mask_spheres:
            occupied_flat &= ~mask_points_in_spheres(occupied_centers, robot_mask_spheres)

        occupied_voxel_centers = occupied_centers[occupied_flat]
        occupancy_mask = voxel_centers_to_occupancy_mask(occupied_voxel_centers, self.voxel_grid)
        merged_boxes = occupancy_mask_to_boxes(
            occupancy_mask,
            origin=self.voxel_grid.origin,
            voxel_size=self.voxel_grid.voxel_size,
            min_component_voxels=self.cfg.get("min_component_voxels", 1),
            max_boxes=self.cfg.get("max_boxes"),
            merge_strategy=self.cfg.get("merge_strategy", "greedy_cuboids"),
        )

        return NvbloxReconstructionResult(
            occupied_voxel_centers=occupied_voxel_centers,
            occupancy_mask=occupancy_mask,
            occupancy_values=occupancy_values,
            voxel_grid_origin=self.voxel_grid.origin,
            voxel_grid_shape=self.voxel_grid.grid_shape,
            voxel_size=self.voxel_grid.voxel_size,
            merged_boxes=merged_boxes,
            robot_ignore_spheres=robot_ignore_spheres,
            metadata={
                "integrator_type": self.integrator_type,
                "extraction_method": self.extraction_method,
                "n_frames": len(bundle.frames),
                "n_query_voxels": (
                    int(self.voxel_grid.centers.shape[0]) if self.extraction_method != "tsdf_sparse" else None
                ),
                "n_occupied_voxels": int(occupied_voxel_centers.shape[0]),
                "query_chunk_size": int(self.cfg.get("query_chunk_size", 250000)),
                "device": self.device,
                "pose_convention": self.cfg.get("pose_convention", "sapien"),
                "ignore_box_margin": float(self.cfg.get("ignore_box_margin", 0.0)),
                "ignore_scene_box_margin": float(self.cfg.get("ignore_scene_box_margin", 0.0)),
                "robot_sphere_margin": float(self.cfg.get("robot_sphere_margin", 0.0)),
                "inflate_robot_mask_by_voxel_extent": bool(self.cfg.get("inflate_robot_mask_by_voxel_extent", True)),
                "robot_mask_voxel_extent_margin": float(robot_mask_voxel_extent_margin),
                "update_esdf": bool(self.cfg.get("update_esdf", False)),
                "merge_strategy": self.cfg.get("merge_strategy", "greedy_cuboids"),
                "mapper_reused_after_reset": bool(self.last_reset_reused_mapper),
                "mapper_recreated_count": int(self.mapper_recreated_count),
            },
            mapper=self.mapper if self.cfg.get("keep_mapper", False) else None,
        )


def save_reconstruction_outputs_for_demo(result, reconstruction_dir, cfg):
    return save_reconstruction_outputs(
        result,
        results_dir=reconstruction_dir,
        boxes_filename=cfg.get("boxes_filename", "reconstructed_boxes.yaml"),
        summary_filename=cfg.get("summary_filename", "reconstruction_summary.yaml"),
        voxels_filename=cfg.get("voxels_filename", "occupied_voxels.npz"),
    )


def _save_click_artifacts(cfg, cfg_path, results_dir, runtime_extra_boxes, camera_specs, bundle, result):
    capture_dir = os.path.join(results_dir, cfg.get("capture_results_subdir", "capture"))
    reconstruction_dir = os.path.join(results_dir, cfg.get("reconstruction_results_subdir", "nvblox"))
    os.makedirs(capture_dir, exist_ok=True)
    os.makedirs(reconstruction_dir, exist_ok=True)

    scene_spec = _build_scene_spec(cfg, runtime_extra_boxes)
    capture_request = build_capture_request(
        scene_spec=scene_spec,
        camera_specs=camera_specs,
        metadata={"config_path": cfg_path},
    )

    scene_spec_path = os.path.join(capture_dir, cfg.get("scene_spec_filename", "scene_spec.yaml"))
    capture_request_path = os.path.join(capture_dir, cfg.get("capture_request_filename", "capture_request.yaml"))
    save_yaml(scene_spec, scene_spec_path)
    save_yaml(capture_request, capture_request_path)

    bundle_path = None
    if cfg.get("save_depth_bundle", True):
        bundle_path = os.path.join(capture_dir, cfg.get("bundle_filename", "depth_capture_bundle.npz"))
        save_depth_capture_bundle(bundle, bundle_path)

    preview_paths = []
    if cfg.get("save_depth_previews", False):
        preview_dir = os.path.join(capture_dir, cfg.get("preview_dirname", "depth_previews"))
        preview_paths = save_depth_preview_images(
            bundle,
            output_dir=preview_dir,
            cmap=cfg.get("preview_cmap", "viridis"),
        )

    saved_paths = save_reconstruction_outputs_for_demo(result, reconstruction_dir, cfg)
    projection_paths = []
    if cfg.get("save_occupancy_projections", False):
        projection_dir = os.path.join(reconstruction_dir, cfg.get("projection_dirname", "occupancy_projections"))
        projection_paths = save_occupancy_projections(
            result.occupancy_mask,
            output_dir=projection_dir,
            prefix=cfg.get("projection_prefix", "interactive_nvblox"),
        )

    return {
        "scene_spec_path": scene_spec_path,
        "capture_request_path": capture_request_path,
        "bundle_path": bundle_path,
        "preview_paths": preview_paths,
        "saved_paths": saved_paths,
        "projection_paths": projection_paths,
    }


def main():
    parser = argparse.ArgumentParser(description="Interactive viser demo: SAPIEN depth capture -> nvblox boxes")
    parser.add_argument(
        "--cfg",
        default="scripts/deployment/cfgs/interactive_nvblox_reconstruction_demo_warehouse.yaml",
        help="Path to the interactive nvblox reconstruction demo YAML config",
    )
    args = parser.parse_args()

    cfg_path = _resolve_repo_path(args.cfg)
    with open(cfg_path, "r") as stream:
        cfg = yaml.safe_load(stream) or {}

    fix_random_seed(cfg.get("seed", 0))

    results_dir = _resolve_repo_path(cfg.get("results_dir", "logs/interactive_nvblox_reconstruction_demo"))
    os.makedirs(results_dir, exist_ok=True)
    save_yaml(cfg, os.path.join(results_dir, "config.yaml"))

    initial_boxes = _load_extra_boxes(cfg)
    initial_scene_spec = _build_scene_spec(cfg, [])
    camera_specs = parse_camera_specs(cfg["camera_specs"])
    robot_ignore_spheres = []
    if cfg.get("subtract_robot", True):
        robot_ignore_spheres = build_robot_ignore_spheres(
            robot_cfg=deepcopy(cfg.get("robot", {})),
            robot_sphere_margin=0.0,
        )

    sapien_capturer = ReusableSapienDepthCapturer(
        cfg=cfg,
        camera_specs=camera_specs,
        fixed_scene_spec=initial_scene_spec,
        cfg_path=cfg_path,
    )
    nvblox_reconstructor = ReusableNvbloxReconstructor(
        cfg=cfg,
        scene_spec=initial_scene_spec,
        robot_ignore_spheres=robot_ignore_spheres,
    )

    server = viser.ViserServer(
        host=cfg.get("viser_host", "0.0.0.0"),
        port=int(cfg.get("viser_port", 8081)),
        label=cfg.get("viser_label", "MPD nvblox Reconstruction Demo"),
    )

    _add_fixed_scene_to_viser(server, initial_scene_spec)
    if cfg.get("show_camera_frames", True):
        _add_camera_frames_to_viser(
            server,
            camera_specs,
            axes_length=cfg.get("camera_axes_length", 0.12),
        )

    status = server.gui.add_text("Status", "Idle", disabled=True)
    capture_time_text = server.gui.add_text("Capture time", "-", disabled=True)
    reconstruction_time_text = server.gui.add_text("Reconstruction time", "-", disabled=True)
    total_time_text = server.gui.add_text("Total time", "-", disabled=True)
    frames_text = server.gui.add_text("Depth frames", "-", disabled=True)
    voxels_text = server.gui.add_text("Occupied voxels", "-", disabled=True)
    boxes_text = server.gui.add_text("Reconstructed boxes", "-", disabled=True)
    capture_count_text = server.gui.add_text("Capture count", "0", disabled=True)
    mapper_text = server.gui.add_text("Mapper reuse", "-", disabled=True)
    show_voxels = server.gui.add_checkbox("Show occupied voxels", bool(cfg.get("show_occupied_voxels", False)))
    reconstruct_button = server.gui.add_button("Capture And Reconstruct")
    clear_button = server.gui.add_button("Clear Reconstruction")

    if cfg.get("show_robot_model", True):
        try:
            _add_robot_model_to_viser(
                server,
                robot_cfg=cfg.get("robot", {}),
                show_collision=cfg.get("show_robot_collision", False),
            )
        except Exception as exc:
            traceback.print_exc()
            status.value = f"Robot model visualization failed: {exc}"

    runtime_widgets = [_make_runtime_box_widget(server, idx, box_spec) for idx, box_spec in enumerate(initial_boxes)]

    scene_update_lock = threading.Lock()
    busy_lock = threading.Lock()
    busy = {"value": False}
    capture_count = {"value": 0}

    @clear_button.on_click
    def _(_event):
        with scene_update_lock:
            _clear_reconstruction_viser(server)
        status.value = "Cleared reconstruction"
        frames_text.value = "-"
        voxels_text.value = "-"
        boxes_text.value = "-"

    @reconstruct_button.on_click
    def _(_event):
        with busy_lock:
            if busy["value"]:
                return
            busy["value"] = True
            reconstruct_button.disabled = True
            clear_button.disabled = True

        status.value = "Capturing depth in reused SAPIEN scene..."
        capture_time_text.value = "-"
        reconstruction_time_text.value = "-"
        total_time_text.value = "-"
        frames_text.value = "-"
        voxels_text.value = "-"
        boxes_text.value = "-"

        total_t0 = time.perf_counter()
        try:
            current_extra_boxes = [box for widget in runtime_widgets if (box := widget.current_box_spec()) is not None]
            if cfg.get("save_runtime_state_on_click", True):
                _save_runtime_state(
                    os.path.join(results_dir, cfg.get("runtime_state_filename", "interactive_nvblox_state.yaml")),
                    extra_boxes=current_extra_boxes,
                )

            capture_count["value"] += 1
            capture_count_text.value = str(capture_count["value"])

            capture_t0 = time.perf_counter()
            bundle = sapien_capturer.capture(current_extra_boxes)
            capture_time = time.perf_counter() - capture_t0

            status.value = "Reconstructing occupancy with reused nvblox mapper..."
            reconstruction_t0 = time.perf_counter()
            result = nvblox_reconstructor.reconstruct(bundle)
            reconstruction_time = time.perf_counter() - reconstruction_t0
            total_time = time.perf_counter() - total_t0

            artifacts = None
            if cfg.get("save_outputs_on_click", True):
                artifacts = _save_click_artifacts(
                    cfg=cfg,
                    cfg_path=cfg_path,
                    results_dir=results_dir,
                    runtime_extra_boxes=current_extra_boxes,
                    camera_specs=camera_specs,
                    bundle=bundle,
                    result=result,
                )

            with scene_update_lock:
                _clear_reconstruction_viser(server)
                _add_reconstruction_to_viser(
                    server,
                    result,
                    show_voxels=bool(show_voxels.value),
                    max_voxels=cfg.get("max_occupied_voxels_to_render", 10000),
                    voxel_point_size=cfg.get("occupied_voxel_point_size", 0.012),
                )

            capture_time_text.value = _format_seconds(capture_time)
            reconstruction_time_text.value = _format_seconds(reconstruction_time)
            total_time_text.value = _format_seconds(total_time)
            frames_text.value = str(result.metadata.get("n_frames", len(bundle.frames)))
            voxels_text.value = str(result.metadata.get("n_occupied_voxels", result.occupied_voxel_centers.shape[0]))
            boxes_text.value = str(len(result.merged_boxes))
            mapper_text.value = (
                "reused"
                if result.metadata.get("mapper_reused_after_reset", True)
                else f"recreated ({result.metadata.get('mapper_recreated_count', 0)})"
            )

            status_suffix = ""
            if artifacts is not None:
                status_suffix = f" Saved to {artifacts['saved_paths']['boxes_path']}"
            status.value = (
                f"Capture {capture_count['value']}: {len(result.merged_boxes)} boxes, "
                f"{int(result.occupied_voxel_centers.shape[0])} occupied voxels."
                f"{status_suffix}"
            )
        except Exception as exc:
            traceback.print_exc()
            status.value = f"Error: {exc}"
        finally:
            with busy_lock:
                busy["value"] = False
                reconstruct_button.disabled = False
                clear_button.disabled = False

    print("\n----------------INTERACTIVE NVBLOX RECONSTRUCTION DEMO----------------")
    print(f"cfg: {cfg_path}")
    print(f"results_dir: {results_dir}")
    print(f"viser: http://{cfg.get('viser_host', '127.0.0.1')}:{int(cfg.get('viser_port', 8081))}")
    print("Move runtime box gizmos in the browser, then click 'Capture And Reconstruct'.")

    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        sapien_capturer.close()


if __name__ == "__main__":
    main()
