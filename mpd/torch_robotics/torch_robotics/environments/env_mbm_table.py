import numpy as np
import torch

import torch_robotics.robots as tr_robots
from torch_robotics.environments.env_base import EnvBase
from torch_robotics.environments.primitives import MultiBoxField, ObjectField
from torch_robotics.torch_utils.torch_utils import DEFAULT_TENSOR_ARGS


# Box-only approximation of Motion Bench Maker's table scene:
# https://github.com/KavrakiLab/motion_bench_maker/blob/main/configs/scenes/table/scene_table.yaml
# The upstream scene includes cylinders; here we approximate them with axis-aligned boxes
# using the cylinder diameter in x/y and the cylinder height in z so the scene can flow
# through the existing box-based planning and Viser/SAPIEN scene adapters.
_MBM_TABLE_SCENE_OBJECTS = [
    {"name": "Can1", "shape": "cylinder", "dimensions": [0.12, 0.03], "position": [0.85, 0.0, 0.8]},
    {"name": "Cube", "shape": "box", "dimensions": [0.25, 0.25, 0.25], "position": [0.75, 0.4, 0.85]},
    {
        "name": "table_leg_left_back",
        "shape": "box",
        "dimensions": [0.05, 0.05, 0.7],
        "position": [1.5, 0.85, 0.35],
    },
    {
        "name": "table_leg_left_front",
        "shape": "box",
        "dimensions": [0.05, 0.05, 0.7],
        "position": [0.6, 0.85, 0.35],
    },
    {
        "name": "table_leg_right_back",
        "shape": "box",
        "dimensions": [0.05, 0.05, 0.7],
        "position": [1.5, -0.85, 0.35],
    },
    {
        "name": "table_leg_right_front",
        "shape": "box",
        "dimensions": [0.05, 0.05, 0.7],
        "position": [0.6, -0.85, 0.35],
    },
    {"name": "table_top", "shape": "box", "dimensions": [1.2, 2.0, 0.04], "position": [1.05, 0.0, 0.7]},
    {"name": "Object1", "shape": "cylinder", "dimensions": [0.35, 0.05], "position": [1.35, 0.0, 0.85]},
    {"name": "Object2", "shape": "box", "dimensions": [0.2, 0.02, 0.4], "position": [1.05, -0.2, 0.9]},
    {"name": "Object3", "shape": "box", "dimensions": [0.02, 0.2, 0.4], "position": [0.65, 0.2, 0.9]},
    {"name": "Object4", "shape": "box", "dimensions": [0.2, 0.05, 0.35], "position": [0.65, -0.2, 0.9]},
    {"name": "Object5", "shape": "box", "dimensions": [0.2, 0.05, 0.35], "position": [1.05, 0.2, 0.9]},
]


def _quat_xyzw_to_rotmat(quat_xyzw):
    x, y, z, w = np.asarray(quat_xyzw, dtype=np.float32)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float32,
    )


def _box_size_from_scene_spec(obj_spec):
    dims = np.asarray(obj_spec["dimensions"], dtype=np.float32)
    if obj_spec["shape"] == "box":
        return dims
    if obj_spec["shape"] == "cylinder":
        height, radius = dims
        diameter = 2.0 * radius
        return np.array([diameter, diameter, height], dtype=np.float32)
    raise ValueError(f"Unsupported MBM primitive shape: {obj_spec['shape']}")


def create_mbm_table_scene_fields(
    base_offset_position=(0.1, 0.1, -0.5),
    base_offset_orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
    tensor_args=DEFAULT_TENSOR_ARGS,
):
    base_offset_position = np.asarray(base_offset_position, dtype=np.float32)
    base_offset_rotation = _quat_xyzw_to_rotmat(base_offset_orientation_xyzw)
    obj_fields = []
    for obj_spec in _MBM_TABLE_SCENE_OBJECTS:
        center = np.asarray(obj_spec["position"], dtype=np.float32)
        size = _box_size_from_scene_spec(obj_spec).reshape(1, 3)
        center = (base_offset_rotation @ center) + base_offset_position
        primitive = MultiBoxField(center.reshape(1, 3), size, tensor_args=tensor_args)
        obj_fields.append(ObjectField([primitive], name=obj_spec["name"]))
    return obj_fields


class EnvMBMTable(EnvBase):
    def __init__(
        self,
        base_offset_position=(0.1, 0.1, -0.5),
        base_offset_orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
        tensor_args=DEFAULT_TENSOR_ARGS,
        **kwargs,
    ):
        base_offset_position = np.asarray(base_offset_position, dtype=np.float32)
        super().__init__(
            limits=torch.tensor([[-1.0, -1.2, -0.1], [1.9, 1.2, 1.6]], **tensor_args)
            + torch.as_tensor(base_offset_position, **tensor_args),
            obj_fixed_list=create_mbm_table_scene_fields(
                base_offset_position=base_offset_position,
                base_offset_orientation_xyzw=base_offset_orientation_xyzw,
                tensor_args=tensor_args,
            ),
            tensor_args=tensor_args,
            **kwargs,
        )

    def get_gpmp2_params(self, robot=None):
        params = dict(
            opt_iters=250,
            num_samples=64,
            sigma_start=1e-3,
            sigma_gp=1e-1,
            sigma_goal_prior=1e-3,
            sigma_coll=1e-4,
            step_size=5e-1,
            sigma_start_init=1e-4,
            sigma_goal_init=1e-4,
            sigma_gp_init=0.1,
            sigma_start_sample=1e-3,
            sigma_goal_sample=1e-3,
            solver_params={
                "delta": 1e-2,
                "trust_region": True,
                "method": "cholesky",
            },
        )
        if isinstance(robot, tr_robots.RobotPanda):
            return params
        raise NotImplementedError

    def get_rrt_connect_params(self, robot=None):
        params = dict(n_iters=10000, step_size=torch.pi / 80, n_radius=torch.pi / 4, n_pre_samples=50000, max_time=15)
        if isinstance(robot, tr_robots.RobotPanda):
            return params
        raise NotImplementedError
