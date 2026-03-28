from collections.abc import Mapping, Sequence

import numpy as np
import torch

from torch_robotics.environments.primitives import MultiBoxField, ObjectField
from torch_robotics.robots import RobotPanda
from torch_robotics.torch_utils.torch_utils import DEFAULT_TENSOR_ARGS


def _as_box_array(values, field_name):
    array = np.asarray(values, dtype=float)
    if array.shape != (3,):
        raise ValueError(f"{field_name} must have shape (3,), got {array.shape}")
    return array


def box_spec_to_object_field(box_spec, tensor_args=DEFAULT_TENSOR_ARGS, default_name="runtime_box"):
    if isinstance(box_spec, Mapping):
        center = box_spec.get("center")
        size = box_spec.get("size")
        name = box_spec.get("name", default_name)
    elif isinstance(box_spec, Sequence) and len(box_spec) == 2:
        center, size = box_spec
        name = default_name
    else:
        raise TypeError("Each box spec must be a mapping with center/size or a (center, size) pair")

    center_array = _as_box_array(center, "center")
    size_array = _as_box_array(size, "size")
    primitive = MultiBoxField(center_array[None, :], size_array[None, :], tensor_args=tensor_args)
    return ObjectField([primitive], name=name)


def build_object_fields_from_boxes(box_specs, tensor_args=DEFAULT_TENSOR_ARGS):
    if box_specs is None:
        return []

    object_fields = []
    for idx, box_spec in enumerate(box_specs):
        object_fields.append(
            box_spec_to_object_field(
                box_spec,
                tensor_args=tensor_args,
                default_name=f"runtime_box_{idx}",
            )
        )
    return object_fields


def _sphere_intersects_axis_aligned_box(sphere_center, sphere_radius, box_center, box_size, box_margin=0.0):
    sphere_center = np.asarray(sphere_center, dtype=np.float32)
    box_center = np.asarray(box_center, dtype=np.float32)
    half_size = 0.5 * np.asarray(box_size, dtype=np.float32) + float(box_margin)
    lower = box_center - half_size
    upper = box_center + half_size
    closest_point = np.clip(sphere_center, lower, upper)
    distance_sq = np.sum((sphere_center - closest_point) ** 2)
    return bool(distance_sq <= float(sphere_radius) ** 2)


def filter_box_specs_for_panda_q(
    box_specs,
    q,
    gripper=False,
    sphere_margin=0.0,
    box_margin=0.0,
):
    if box_specs is None:
        return [], []

    q_tensor = torch.as_tensor(q, dtype=torch.float32)
    if q_tensor.ndim != 1:
        raise ValueError(f"Expected q with shape (dof,), got {tuple(q_tensor.shape)}")

    tensor_args = {"device": torch.device("cpu"), "dtype": torch.float32}
    robot = RobotPanda(gripper=gripper, tensor_args=tensor_args)
    try:
        sphere_centers = robot.fk_map_collision(q_tensor.cpu()).squeeze(0).detach().cpu().numpy().astype(np.float32)
        sphere_radii = robot.link_collision_spheres_radii.detach().cpu().numpy().astype(np.float32).reshape(-1)
        sphere_radii = sphere_radii + float(sphere_margin)

        kept_box_specs = []
        removed_box_specs = []
        for idx, box_spec in enumerate(box_specs):
            box_center = np.asarray(box_spec["center"], dtype=np.float32)
            box_size = np.asarray(box_spec["size"], dtype=np.float32)
            collides = any(
                _sphere_intersects_axis_aligned_box(
                    sphere_center=sphere_centers[sphere_idx],
                    sphere_radius=sphere_radii[sphere_idx],
                    box_center=box_center,
                    box_size=box_size,
                    box_margin=box_margin,
                )
                for sphere_idx in range(sphere_centers.shape[0])
            )

            enriched_box_spec = dict(box_spec)
            enriched_box_spec.setdefault("name", f"runtime_box_{idx}")
            if collides:
                removed_box_specs.append(enriched_box_spec)
            else:
                kept_box_specs.append(enriched_box_spec)

        return kept_box_specs, removed_box_specs
    finally:
        robot.cleanup()
