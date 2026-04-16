import torch

from mpd.deployment.nvblox_bridge import require_nvblox_torch
from torch_robotics.torch_utils.torch_utils import DEFAULT_TENSOR_ARGS, to_torch


class NvbloxEsdfField:
    def __init__(
        self,
        mapper,
        mapper_id=0,
        tensor_args=DEFAULT_TENSOR_ARGS,
        unknown_distance=None,
        unknown_distance_replacement=None,
    ):
        _, _, _, QueryType, _ = require_nvblox_torch()
        from nvblox_torch.constants import constants

        self.mapper = mapper
        self.mapper_id = int(mapper_id)
        self.tensor_args = tensor_args
        self.query_type_esdf = QueryType.ESDF
        self.unknown_distance = (
            float(constants.esdf_unknown_distance()) if unknown_distance is None else float(unknown_distance)
        )
        self.unknown_distance_replacement = (
            self.unknown_distance
            if unknown_distance_replacement is None
            else float(unknown_distance_replacement)
        )

    def _prepare_query(self, x):
        x_t = to_torch(x, **self.tensor_args)
        if x_t.shape[-1] != 3:
            raise ValueError(f"NvbloxEsdfField expects queries with last dimension 3, got shape {tuple(x_t.shape)}")
        x_flat = x_t.reshape(-1, 3)
        query_spheres = torch.cat(
            (
                x_flat,
                torch.zeros((x_flat.shape[0], 1), device=x_flat.device, dtype=x_flat.dtype),
            ),
            dim=1,
        )
        return x_t, x_flat, query_spheres

    def _mask_unknown(self, sdf_vals, sdf_grad=None):
        unknown_mask = sdf_vals >= (self.unknown_distance - 1e-6)
        if not torch.any(unknown_mask):
            return sdf_vals, sdf_grad

        sdf_vals = sdf_vals.clone()
        sdf_vals[unknown_mask] = float(self.unknown_distance_replacement)
        if sdf_grad is not None:
            sdf_grad = sdf_grad.clone()
            sdf_grad[unknown_mask] = 0.0
        return sdf_vals, sdf_grad

    def __call__(self, x, **kwargs):
        return self.compute_signed_distance(x, **kwargs)

    def compute_cost(self, x, **kwargs):
        return self.compute_signed_distance(x, **kwargs)

    def compute_signed_distance(self, x, get_gradient=False, **kwargs):
        x_t, x_flat, query_spheres = self._prepare_query(x)
        if get_gradient:
            query_output = torch.zeros((x_flat.shape[0], 4), device=x_flat.device, dtype=x_flat.dtype)
            sdf_vals = self.mapper.query_differentiable_layer(
                self.query_type_esdf,
                query=query_spheres,
                output=query_output,
                mapper_id=self.mapper_id,
            )
            sdf_vals = sdf_vals.reshape(-1)
            sdf_grad = query_output[:, :3]
            sdf_vals, sdf_grad = self._mask_unknown(sdf_vals, sdf_grad)
            return sdf_vals.view(x_t.shape[:-1]), sdf_grad.view(x_t.shape)

        query_output = torch.zeros((x_flat.shape[0], 1), device=x_flat.device, dtype=x_flat.dtype)
        sdf_vals = self.mapper.query_layer(
            self.query_type_esdf,
            query=query_spheres,
            output=query_output,
            mapper_id=self.mapper_id,
        )
        sdf_vals = sdf_vals.reshape(-1)
        sdf_vals, _ = self._mask_unknown(sdf_vals, None)
        return sdf_vals.view(x_t.shape[:-1])

    def zero_grad(self):
        pass
