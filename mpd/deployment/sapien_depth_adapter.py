from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")

import matplotlib.pyplot as plt
import numpy as np
import yaml

from torch_robotics.environments.primitives import MultiBoxField
from torch_robotics.environments.env_warehouse import EnvWarehouse
from torch_robotics.torch_kinematics_tree.utils.files import get_robot_path
from torch_robotics.torch_utils.torch_utils import DEFAULT_TENSOR_ARGS


def _rotation_matrix_from_euler_xyz_deg(euler_xyz_deg):
    rx, ry, rz = np.deg2rad(np.asarray(euler_xyz_deg, dtype=float))

    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)

    rot_x = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]])
    rot_y = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
    rot_z = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])
    return rot_z @ rot_y @ rot_x


def _make_pose_matrix(position, rotation_matrix=None, orientation_euler_xyz_deg=None):
    pose = np.eye(4, dtype=float)
    pose[:3, 3] = np.asarray(position, dtype=float)

    if rotation_matrix is not None and orientation_euler_xyz_deg is not None:
        raise ValueError("Provide either rotation_matrix or orientation_euler_xyz_deg, not both")

    if rotation_matrix is not None:
        pose[:3, :3] = np.asarray(rotation_matrix, dtype=float)
    elif orientation_euler_xyz_deg is not None:
        pose[:3, :3] = _rotation_matrix_from_euler_xyz_deg(orientation_euler_xyz_deg)

    return pose


def _normalize(vector):
    vector = np.asarray(vector, dtype=float)
    norm = np.linalg.norm(vector)
    if norm < 1e-8:
        raise ValueError("Cannot normalize a near-zero vector")
    return vector / norm


def make_camera_pose_look_at(position, target, up=(0.0, 0.0, 1.0)):
    position = np.asarray(position, dtype=float)
    target = np.asarray(target, dtype=float)
    up = np.asarray(up, dtype=float)

    forward = _normalize(target - position)
    left = _normalize(np.cross(up, forward))
    true_up = _normalize(np.cross(forward, left))

    pose = np.eye(4, dtype=float)
    pose[:3, 0] = forward
    pose[:3, 1] = left
    pose[:3, 2] = true_up
    pose[:3, 3] = position
    return pose


@dataclass
class SceneBoxSpec:
    name: str
    center: list[float]
    size: list[float]
    pose_world: list[list[float]]
    source: str = "environment"


@dataclass
class DepthCameraSpec:
    name: str
    width: int
    height: int
    fx: Optional[float] = None
    fy: Optional[float] = None
    cx: Optional[float] = None
    cy: Optional[float] = None
    fov_y_deg: Optional[float] = None
    pose_world: Optional[list[list[float]]] = None
    near: float = 0.1
    far: float = 4.0
    position: Optional[list[float]] = None
    target: Optional[list[float]] = None
    up: list[float] = field(default_factory=lambda: [0.0, 0.0, 1.0])
    orientation_euler_xyz_deg: Optional[list[float]] = None

    def resolved_pose_world(self):
        if self.pose_world is not None:
            return np.asarray(self.pose_world, dtype=float)
        if self.position is None:
            raise ValueError(f"Camera '{self.name}' needs either pose_world or position")
        if self.target is not None:
            return make_camera_pose_look_at(self.position, self.target, self.up)
        return _make_pose_matrix(
            position=self.position,
            orientation_euler_xyz_deg=self.orientation_euler_xyz_deg,
        )

    def resolved_intrinsics(self):
        if self.fx is not None and self.fy is not None and self.cx is not None and self.cy is not None:
            return {
                "fx": float(self.fx),
                "fy": float(self.fy),
                "cx": float(self.cx),
                "cy": float(self.cy),
            }

        if self.fov_y_deg is None:
            raise ValueError(
                f"Camera '{self.name}' needs either explicit fx/fy/cx/cy or fov_y_deg for derived intrinsics"
            )

        fov_y_rad = np.deg2rad(self.fov_y_deg)
        fy = 0.5 * self.height / np.tan(fov_y_rad / 2.0)
        fx = fy
        cx = (self.width - 1) / 2.0
        cy = (self.height - 1) / 2.0
        return {"fx": float(fx), "fy": float(fy), "cx": float(cx), "cy": float(cy)}


@dataclass
class DepthFrame:
    camera_name: str
    depth_meters: np.ndarray
    intrinsic_matrix: np.ndarray
    pose_world: np.ndarray


@dataclass
class DepthCaptureBundle:
    frames: list[DepthFrame]
    metadata: dict[str, Any]


def require_sapien():
    try:
        import sapien  # noqa: F401

        return sapien
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "SAPIEN is not installed in this environment. "
            "You can still export scene/camera specs now, but live depth capture requires installing SAPIEN first."
        ) from exc


def _object_field_to_scene_boxes(obj_field, source):
    scene_boxes = []
    obj_pos = np.asarray(obj_field.pos.detach().cpu(), dtype=float)
    obj_ori = np.asarray(obj_field.ori.detach().cpu(), dtype=float)

    for field_idx, primitive_field in enumerate(obj_field.fields):
        if not isinstance(primitive_field, MultiBoxField):
            continue

        centers = np.asarray(primitive_field.centers.detach().cpu(), dtype=float)
        sizes = np.asarray(primitive_field.sizes.detach().cpu(), dtype=float)
        for box_idx, (center_local, size) in enumerate(zip(centers, sizes)):
            center_world = center_local @ obj_ori.T + obj_pos
            pose_world = np.eye(4, dtype=float)
            pose_world[:3, :3] = obj_ori
            pose_world[:3, 3] = center_world
            scene_boxes.append(
                SceneBoxSpec(
                    name=f"{obj_field.name}_{field_idx}_{box_idx}",
                    center=center_world.tolist(),
                    size=size.tolist(),
                    pose_world=pose_world.tolist(),
                    source=source,
                )
            )
    return scene_boxes


def environment_to_scene_boxes(env):
    scene_boxes = []
    for obj_field in env.obj_fixed_list:
        scene_boxes.extend(_object_field_to_scene_boxes(obj_field, source="fixed"))
    for obj_field in env.obj_extra_list:
        scene_boxes.extend(_object_field_to_scene_boxes(obj_field, source="extra"))
    return scene_boxes


def build_warehouse_scene_spec(extra_boxes=None, rotation_z_axis_deg=0.0, tensor_args=DEFAULT_TENSOR_ARGS):
    env = EnvWarehouse(
        rotation_z_axis_deg=rotation_z_axis_deg,
        obj_extra_list=extra_boxes if extra_boxes is not None else [],
        tensor_args=tensor_args,
    )
    return dict(
        environment="EnvWarehouse",
        rotation_z_axis_deg=rotation_z_axis_deg,
        workspace_limits=np.asarray(env.limits.detach().cpu(), dtype=float).tolist(),
        boxes=[asdict(box_spec) for box_spec in environment_to_scene_boxes(env)],
    )


def parse_camera_specs(camera_specs_cfg):
    return [DepthCameraSpec(**camera_spec_cfg) for camera_spec_cfg in camera_specs_cfg]


def build_capture_request(scene_spec, camera_specs, metadata=None):
    return {
        "scene_spec": scene_spec,
        "camera_specs": [
            {
                **asdict(camera_spec),
                "pose_world": camera_spec.resolved_pose_world().tolist(),
                "intrinsics": camera_spec.resolved_intrinsics(),
            }
            for camera_spec in camera_specs
        ],
        "metadata": metadata or {},
    }


def save_yaml(data, path):
    with open(path, "w") as stream:
        yaml.dump(data, stream, Dumper=yaml.Dumper, allow_unicode=True)


def save_depth_capture_bundle(bundle: DepthCaptureBundle, path):
    arrays = {"metadata_json": np.asarray(json.dumps(bundle.metadata))}
    for idx, frame in enumerate(bundle.frames):
        arrays[f"frame_{idx}_camera_name"] = np.asarray(frame.camera_name)
        arrays[f"frame_{idx}_depth_meters"] = frame.depth_meters
        arrays[f"frame_{idx}_intrinsic_matrix"] = frame.intrinsic_matrix
        arrays[f"frame_{idx}_pose_world"] = frame.pose_world
    np.savez_compressed(path, **arrays)


def load_depth_capture_bundle(path):
    data = np.load(path, allow_pickle=False)

    metadata_json = data["metadata_json"]
    if isinstance(metadata_json, np.ndarray):
        metadata_json = metadata_json.item()
    metadata = json.loads(metadata_json)

    frames = []
    frame_idx = 0
    while f"frame_{frame_idx}_camera_name" in data.files:
        camera_name = data[f"frame_{frame_idx}_camera_name"]
        if isinstance(camera_name, np.ndarray):
            camera_name = camera_name.item()

        frames.append(
            DepthFrame(
                camera_name=str(camera_name),
                depth_meters=np.asarray(data[f"frame_{frame_idx}_depth_meters"], dtype=np.float32),
                intrinsic_matrix=np.asarray(data[f"frame_{frame_idx}_intrinsic_matrix"], dtype=np.float32),
                pose_world=np.asarray(data[f"frame_{frame_idx}_pose_world"], dtype=np.float32),
            )
        )
        frame_idx += 1

    return DepthCaptureBundle(frames=frames, metadata=metadata)


def _normalize_depth_for_preview(depth_meters, near=None, far=None):
    depth = np.asarray(depth_meters, dtype=np.float32)
    valid_mask = np.isfinite(depth) & (depth > 0.0)
    if not np.any(valid_mask):
        return np.zeros_like(depth, dtype=np.float32)

    valid_depth = depth[valid_mask]
    depth_min = float(valid_depth.min()) if near is None else float(near)
    depth_max = float(valid_depth.max()) if far is None else float(far)
    if depth_max <= depth_min:
        depth_max = depth_min + 1e-6

    normalized = np.zeros_like(depth, dtype=np.float32)
    normalized[valid_mask] = (valid_depth - depth_min) / (depth_max - depth_min)
    return 1.0 - np.clip(normalized, 0.0, 1.0)


def save_depth_preview_images(bundle: DepthCaptureBundle, output_dir, cmap="viridis"):
    os.makedirs(output_dir, exist_ok=True)

    saved_paths = []
    for frame in bundle.frames:
        preview = _normalize_depth_for_preview(frame.depth_meters)
        output_path = os.path.join(output_dir, f"{frame.camera_name}_depth_preview.png")
        plt.imsave(output_path, preview, cmap=cmap, vmin=0.0, vmax=1.0)
        saved_paths.append(output_path)
    return saved_paths


def _dilate_mask(mask, dilation_px):
    dilation_px = int(dilation_px)
    if dilation_px <= 0:
        return mask

    try:
        from scipy import ndimage
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "scipy is required for robot depth mask dilation. Install scipy or set robot_depth_mask_dilation_px: 0."
        ) from exc

    structure = np.ones((2 * dilation_px + 1, 2 * dilation_px + 1), dtype=bool)
    return ndimage.binary_dilation(mask, structure=structure)


def _apply_robot_depth_mask(depth_meters, robot_depth_meters, depth_epsilon_m=0.02, dilation_px=0):
    depth_meters = np.asarray(depth_meters, dtype=np.float32)
    robot_depth_meters = np.asarray(robot_depth_meters, dtype=np.float32)
    if depth_meters.shape != robot_depth_meters.shape:
        raise ValueError(
            f"Depth and robot depth shapes must match, got {depth_meters.shape} and {robot_depth_meters.shape}"
        )

    valid_depth = np.isfinite(depth_meters) & (depth_meters > 0.0)
    valid_robot = np.isfinite(robot_depth_meters) & (robot_depth_meters > 0.0)
    robot_mask = valid_depth & valid_robot & (np.abs(depth_meters - robot_depth_meters) <= float(depth_epsilon_m))
    robot_mask = _dilate_mask(robot_mask, dilation_px)

    depth_masked = depth_meters.copy()
    depth_masked[robot_mask] = 0.0
    return depth_masked, robot_mask


def _make_fovy_from_intrinsics(camera_spec: DepthCameraSpec):
    intrinsics = camera_spec.resolved_intrinsics()
    fy = intrinsics["fy"]
    return 2.0 * np.arctan(camera_spec.height / (2.0 * fy))


def _rotation_matrix_to_quaternion(rotation_matrix):
    rotation_matrix = np.asarray(rotation_matrix, dtype=np.float32)
    trace = float(np.trace(rotation_matrix))

    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (rotation_matrix[2, 1] - rotation_matrix[1, 2]) / s
        qy = (rotation_matrix[0, 2] - rotation_matrix[2, 0]) / s
        qz = (rotation_matrix[1, 0] - rotation_matrix[0, 1]) / s
    elif rotation_matrix[0, 0] > rotation_matrix[1, 1] and rotation_matrix[0, 0] > rotation_matrix[2, 2]:
        s = np.sqrt(1.0 + rotation_matrix[0, 0] - rotation_matrix[1, 1] - rotation_matrix[2, 2]) * 2.0
        qw = (rotation_matrix[2, 1] - rotation_matrix[1, 2]) / s
        qx = 0.25 * s
        qy = (rotation_matrix[0, 1] + rotation_matrix[1, 0]) / s
        qz = (rotation_matrix[0, 2] + rotation_matrix[2, 0]) / s
    elif rotation_matrix[1, 1] > rotation_matrix[2, 2]:
        s = np.sqrt(1.0 + rotation_matrix[1, 1] - rotation_matrix[0, 0] - rotation_matrix[2, 2]) * 2.0
        qw = (rotation_matrix[0, 2] - rotation_matrix[2, 0]) / s
        qx = (rotation_matrix[0, 1] + rotation_matrix[1, 0]) / s
        qy = 0.25 * s
        qz = (rotation_matrix[1, 2] + rotation_matrix[2, 1]) / s
    else:
        s = np.sqrt(1.0 + rotation_matrix[2, 2] - rotation_matrix[0, 0] - rotation_matrix[1, 1]) * 2.0
        qw = (rotation_matrix[1, 0] - rotation_matrix[0, 1]) / s
        qx = (rotation_matrix[0, 2] + rotation_matrix[2, 0]) / s
        qy = (rotation_matrix[1, 2] + rotation_matrix[2, 1]) / s
        qz = 0.25 * s

    quaternion = np.asarray([qw, qx, qy, qz], dtype=np.float32)
    quaternion /= np.linalg.norm(quaternion)
    return quaternion


def _matrix_to_sapien_pose(sapien_module, pose_world):
    pose_world = np.asarray(pose_world, dtype=np.float32)
    position = pose_world[:3, 3]
    quaternion = _rotation_matrix_to_quaternion(pose_world[:3, :3])
    return sapien_module.Pose(position, quaternion)


def _build_box_actor(scene, sapien_module, box_spec, add_collision=True, color_by_source=True, color_override=None):
    center = np.asarray(box_spec["center"], dtype=float)
    size = np.asarray(box_spec["size"], dtype=float)
    half_size = (size / 2.0).tolist()
    pose_world = np.asarray(box_spec["pose_world"], dtype=float)
    source = box_spec.get("source", "environment")

    if color_override is not None:
        color = list(color_override)
    elif color_by_source:
        if source == "extra":
            color = [0.85, 0.25, 0.25]
        elif source == "fixed":
            color = [0.65, 0.65, 0.65]
        else:
            color = [0.5, 0.5, 0.5]
    else:
        color = [0.65, 0.65, 0.65]

    builder = scene.create_actor_builder()
    if add_collision:
        builder.add_box_collision(half_size=half_size)
    builder.add_box_visual(half_size=half_size, material=color)
    actor = builder.build_static(name=box_spec["name"])
    actor.set_pose(_matrix_to_sapien_pose(sapien_module, pose_world))
    return actor


def _configure_camera(camera, camera_spec: DepthCameraSpec):
    intrinsics = camera_spec.resolved_intrinsics()
    camera.set_perspective_parameters(
        near=float(camera_spec.near),
        far=float(camera_spec.far),
        fx=float(intrinsics["fx"]),
        fy=float(intrinsics["fy"]),
        cx=float(intrinsics["cx"]),
        cy=float(intrinsics["cy"]),
        skew=0.0,
    )


def _make_intrinsic_matrix(camera_spec: DepthCameraSpec):
    intrinsics = camera_spec.resolved_intrinsics()
    return np.asarray(
        [
            [intrinsics["fx"], 0.0, intrinsics["cx"]],
            [0.0, intrinsics["fy"], intrinsics["cy"]],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def _apply_stereo_sensor_overrides(sensor_config, overrides):
    if overrides is None:
        return sensor_config

    for key, value in overrides.items():
        if not hasattr(sensor_config, key):
            raise ValueError(f"Unknown StereoDepthSensorConfig field: {key}")
        if key in {"rgb_intrinsic", "ir_intrinsic"}:
            value = np.asarray(value, dtype=np.float32)
        elif key in {"rgb_resolution", "ir_resolution"}:
            value = tuple(int(v) for v in value)
        elif key in {"trans_pose_l", "trans_pose_r"}:
            try:
                import sapien
            except ModuleNotFoundError as exc:
                raise ModuleNotFoundError("SAPIEN is required to construct stereo sensor pose overrides") from exc

            if isinstance(value, dict):
                position = value.get("p", [0.0, 0.0, 0.0])
                quaternion = value.get("q", [1.0, 0.0, 0.0, 0.0])
                value = sapien.Pose(position, quaternion)
            else:
                raise ValueError(
                    f"Stereo sensor override '{key}' must be a dict with 'p' and optional 'q' fields"
                )
        setattr(sensor_config, key, value)
    return sensor_config


def _make_stereo_sensor_config(camera_spec: DepthCameraSpec, sensor_config_overrides=None):
    from sapien.sensor import StereoDepthSensorConfig

    sensor_config = StereoDepthSensorConfig()
    sensor_config.rgb_resolution = (int(camera_spec.width), int(camera_spec.height))
    sensor_config.rgb_intrinsic = _make_intrinsic_matrix(camera_spec)
    sensor_config.min_depth = float(camera_spec.near)
    sensor_config.max_depth = float(camera_spec.far)
    return _apply_stereo_sensor_overrides(sensor_config, sensor_config_overrides)


def _default_panda_urdf_path():
    return os.path.join(
        get_robot_path().as_posix(),
        "franka_description",
        "robots",
        "panda_arm_hand_fixed_gripper.urdf",
    )


def _load_robot_articulation(scene, sapien_module, robot_cfg):
    if not robot_cfg or not robot_cfg.get("enabled", False):
        return None

    urdf_path = robot_cfg.get("urdf_path", _default_panda_urdf_path())
    if not os.path.isabs(urdf_path):
        urdf_path = os.path.abspath(urdf_path)

    package_dir = robot_cfg.get("package_dir", get_robot_path().as_posix())
    if not os.path.isabs(package_dir):
        package_dir = os.path.abspath(package_dir)

    loader = scene.create_urdf_loader()
    loader.fix_root_link = bool(robot_cfg.get("fix_root_link", True))
    articulation = loader.load(urdf_path, package_dir=package_dir)

    root_pose_world = robot_cfg.get("root_pose_world")
    if root_pose_world is not None:
        articulation.set_root_pose(_matrix_to_sapien_pose(sapien_module, root_pose_world))

    qpos = robot_cfg.get("qpos")
    if qpos is not None:
        qpos = np.asarray(qpos, dtype=np.float32)
        if qpos.shape != (articulation.dof,):
            raise ValueError(
                f"Robot qpos shape mismatch: expected {(articulation.dof,)}, got {qpos.shape}"
            )
        articulation.set_qpos(qpos)

    return articulation


def _create_scene(scene_spec, add_ground=False, include_scene_boxes=True):
    sapien = require_sapien()

    try:
        scene = sapien.Scene()
    except Exception as exc:
        raise RuntimeError(
            "SAPIEN is installed, but creating a render scene failed. "
            "This usually means the current runtime cannot initialize the required rendering backend "
            "(for example Vulkan / offscreen GPU support)."
        ) from exc

    scene.set_timestep(1 / 240.0)
    if add_ground:
        scene.add_ground(altitude=0.0)

    scene.set_ambient_light([0.35, 0.35, 0.35])
    scene.add_directional_light([0.2, 0.5, -1.0], [0.8, 0.8, 0.8], shadow=False)
    scene.add_point_light([1.5, -1.0, 2.2], [1.2, 1.2, 1.2], shadow=False)
    scene.add_point_light([0.2, 1.2, 1.8], [0.8, 0.8, 0.8], shadow=False)

    actors = []
    if include_scene_boxes:
        for box_spec in scene_spec["boxes"]:
            actors.append(_build_box_actor(scene, sapien, box_spec))

    return sapien, scene, actors


def _create_render_cameras(scene, sapien_module, camera_specs):
    cameras = []
    for camera_spec in camera_specs:
        fovy = _make_fovy_from_intrinsics(camera_spec)
        camera = scene.add_camera(
            camera_spec.name,
            int(camera_spec.width),
            int(camera_spec.height),
            float(fovy),
            float(camera_spec.near),
            float(camera_spec.far),
        )
        camera.entity.pose = _matrix_to_sapien_pose(sapien_module, camera_spec.resolved_pose_world())
        _configure_camera(camera, camera_spec)
        cameras.append(camera)
    return cameras


def _create_stereo_sensors(scene, sapien_module, camera_specs, sensor_config_overrides=None):
    from sapien.sensor import StereoDepthSensor

    sensors = []
    for camera_spec in camera_specs:
        mount = scene.create_actor_builder().build_kinematic(name=f"{camera_spec.name}_mount")
        mount.set_pose(sapien_module.Pose())
        sensor_config = _make_stereo_sensor_config(camera_spec, sensor_config_overrides=sensor_config_overrides)
        sensor = StereoDepthSensor(
            sensor_config,
            mount,
            _matrix_to_sapien_pose(sapien_module, camera_spec.resolved_pose_world()),
        )
        sensors.append(sensor)
    return sensors


def _create_scene_and_capturers(
    scene_spec,
    camera_specs,
    add_ground=False,
    capture_backend="render_camera",
    stereo_sensor_config_overrides=None,
    robot_cfg=None,
    include_scene_boxes=True,
):
    sapien, scene, actors = _create_scene(
        scene_spec=scene_spec,
        add_ground=add_ground,
        include_scene_boxes=include_scene_boxes,
    )
    robot = _load_robot_articulation(scene, sapien, robot_cfg)

    if capture_backend == "render_camera":
        capturers = _create_render_cameras(scene, sapien, camera_specs)
    elif capture_backend == "stereo_depth_sensor":
        capturers = _create_stereo_sensors(
            scene,
            sapien,
            camera_specs,
            sensor_config_overrides=stereo_sensor_config_overrides,
        )
    else:
        raise ValueError(
            f"Unsupported capture backend '{capture_backend}'. "
            "Expected one of: render_camera, stereo_depth_sensor"
        )

    return sapien, scene, actors, capturers, robot


def _depth_from_position_picture(position_picture, near, far):
    # SAPIEN's Position picture stores camera-space XYZ; the linear depth is -Z.
    depth = -np.asarray(position_picture[..., 2], dtype=np.float32)
    valid = np.isfinite(depth) & (depth > near) & (depth < far)
    depth_clean = np.zeros_like(depth, dtype=np.float32)
    depth_clean[valid] = depth[valid]
    return depth_clean


def _capture_depth_with_render_cameras(scene_spec, camera_specs, add_ground=False, robot_cfg=None):
    sapien, scene, _, capturers, robot = _create_scene_and_capturers(
        scene_spec=scene_spec,
        camera_specs=camera_specs,
        add_ground=add_ground,
        capture_backend="render_camera",
        robot_cfg=robot_cfg,
    )

    try:
        scene.step()
        scene.update_render()
        frames = []
        for capturer, camera_spec in zip(capturers, camera_specs):
            capturer.take_picture()
            position_picture = capturer.get_picture("Position")
            depth_meters = _depth_from_position_picture(
                position_picture,
                near=float(camera_spec.near),
                far=float(camera_spec.far),
            )
            frames.append(
                DepthFrame(
                    camera_name=camera_spec.name,
                    depth_meters=depth_meters,
                    intrinsic_matrix=np.asarray(capturer.get_intrinsic_matrix(), dtype=np.float32),
                    pose_world=np.asarray(capturer.entity.pose.to_transformation_matrix(), dtype=np.float32),
                )
            )
    finally:
        del capturers
        del robot
        del scene
        del sapien

    return frames


def _capture_robot_only_depth_with_render_cameras(scene_spec, camera_specs, add_ground=False, robot_cfg=None):
    if not robot_cfg or not robot_cfg.get("enabled", False):
        return []

    sapien, scene, _, capturers, robot = _create_scene_and_capturers(
        scene_spec=scene_spec,
        camera_specs=camera_specs,
        add_ground=add_ground,
        capture_backend="render_camera",
        robot_cfg=robot_cfg,
        include_scene_boxes=False,
    )

    try:
        scene.step()
        scene.update_render()
        frames = []
        for capturer, camera_spec in zip(capturers, camera_specs):
            capturer.take_picture()
            position_picture = capturer.get_picture("Position")
            depth_meters = _depth_from_position_picture(
                position_picture,
                near=float(camera_spec.near),
                far=float(camera_spec.far),
            )
            frames.append(
                DepthFrame(
                    camera_name=camera_spec.name,
                    depth_meters=depth_meters,
                    intrinsic_matrix=np.asarray(capturer.get_intrinsic_matrix(), dtype=np.float32),
                    pose_world=np.asarray(capturer.entity.pose.to_transformation_matrix(), dtype=np.float32),
                )
            )
    finally:
        del capturers
        del robot
        del scene
        del sapien

    return frames


def _capture_depth_with_stereo_sensors(
    scene_spec,
    camera_specs,
    add_ground=False,
    stereo_sensor_config_overrides=None,
    robot_cfg=None,
):
    frames = []
    for camera_spec in camera_specs:
        sapien, scene, _, capturers, robot = _create_scene_and_capturers(
            scene_spec=scene_spec,
            camera_specs=[camera_spec],
            add_ground=add_ground,
            capture_backend="stereo_depth_sensor",
            stereo_sensor_config_overrides=stereo_sensor_config_overrides,
            robot_cfg=robot_cfg,
        )

        try:
            scene.step()
            scene.update_render()
            capturer = capturers[0]
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
        finally:
            del capturers
            del robot
            del scene
            del sapien

    return frames


def capture_depth_with_sapien(
    scene_spec,
    camera_specs,
    metadata=None,
    add_ground=False,
    capture_backend="render_camera",
    stereo_sensor_config_overrides=None,
    robot_cfg=None,
    mask_robot_from_depth=False,
    robot_depth_mask_epsilon_m=0.02,
    robot_depth_mask_dilation_px=0,
):
    try:
        if capture_backend == "render_camera":
            frames = _capture_depth_with_render_cameras(
                scene_spec=scene_spec,
                camera_specs=camera_specs,
                add_ground=add_ground,
                robot_cfg=robot_cfg,
            )
        elif capture_backend == "stereo_depth_sensor":
            frames = _capture_depth_with_stereo_sensors(
                scene_spec=scene_spec,
                camera_specs=camera_specs,
                add_ground=add_ground,
                stereo_sensor_config_overrides=stereo_sensor_config_overrides,
                robot_cfg=robot_cfg,
            )
        else:
            raise ValueError(
                f"Unsupported capture backend '{capture_backend}'. "
                "Expected one of: render_camera, stereo_depth_sensor"
            )
    except Exception as exc:
        raise RuntimeError(
            "SAPIEN scene creation succeeded, but depth capture failed while updating render or reading camera outputs."
        ) from exc

    robot_mask_summary = None
    if mask_robot_from_depth and robot_cfg and robot_cfg.get("enabled", False):
        robot_only_frames = _capture_robot_only_depth_with_render_cameras(
            scene_spec=scene_spec,
            camera_specs=camera_specs,
            add_ground=add_ground,
            robot_cfg=robot_cfg,
        )
        robot_frame_map = {frame.camera_name: frame for frame in robot_only_frames}
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
                depth_epsilon_m=robot_depth_mask_epsilon_m,
                dilation_px=robot_depth_mask_dilation_px,
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
            **(metadata or {}),
            "capture_backend": capture_backend,
            "robot_depth_mask_applied": bool(mask_robot_from_depth and robot_cfg and robot_cfg.get("enabled", False)),
            "robot_depth_mask_backend": "render_camera" if mask_robot_from_depth else None,
            "robot_depth_mask_epsilon_m": float(robot_depth_mask_epsilon_m),
            "robot_depth_mask_dilation_px": int(robot_depth_mask_dilation_px),
            "robot_depth_mask_summary": robot_mask_summary,
        },
    )
