# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import warnings
from typing import List, Literal, Optional, Union

import torch
import torch.nn as nn

from warpconvnet.geometry.base.coords import Coords
from warpconvnet.geometry.coords.search.search_configs import RealSearchConfig
from warpconvnet.geometry.coords.search.continuous import RealSearchMode
from warpconvnet.geometry.types.points import Points
from warpconvnet.nn.modules.base_module import BaseSpatialModule
from warpconvnet.nn.encodings import SinusoidalEncoding
from warpconvnet.nn.modules.mlp import FeatureMLPBlock, FeatureResidualMLPBlock
from warpconvnet.ops.reductions import REDUCTION_TYPES_STR, REDUCTIONS, row_reduction

__all__ = ["PointConv"]


def _get_module_input_channel(module: nn.Module) -> int:
    """Recursively call the function to extract the input channel of the module"""
    if isinstance(module, nn.Linear):
        return module.in_features
    elif isinstance(module, nn.Sequential):
        return _get_module_input_channel(module[0])
    elif isinstance(module, nn.Module):
        # Find first module and call the function
        for name, module in module.named_children():
            return _get_module_input_channel(module)
    else:
        raise ValueError(f"Unsupported module type: {type(module)}")


class PointConv(BaseSpatialModule):
    """Point convolution operating on :class:`~warpconvnet.geometry.types.points.Points`.

    Parameters
    ----------
    in_channels : int
        Number of input feature channels.
    out_channels : int
        Number of output feature channels.
    neighbor_search_args : :class:`~warpconvnet.geometry.coords.search.search_configs.RealSearchConfig`
        Configuration for neighbor search.
    pooling_reduction : :class:`~warpconvnet.ops.reductions.REDUCTIONS`, optional
        Reduction used when downsampling points. Required when ``out_point_type`` is
        ``"downsample"``.
    pooling_voxel_size : float, optional
        Size of voxels used for downsampling when ``out_point_type`` is ``"downsample"``.
    edge_transform_mlp : nn.Module, optional
        MLP applied to constructed edge features.
    out_transform_mlp : nn.Module, optional
        MLP applied after neighborhood reduction.
    mlp_block : type, optional
        Class used to build ``edge_transform_mlp`` and ``out_transform_mlp`` when not
        provided. Defaults to :class:`FeatureMLPBlock`.
    hidden_dim : int, optional
        Hidden dimension of the automatically created MLPs.
    channel_multiplier : int, optional
        Multiplier used when ``hidden_dim`` is ``None``. Defaults to ``2``.
    use_rel_pos : bool, optional
        Include relative coordinates of neighbors as part of the edge features.
    use_rel_pos_encode : bool, optional
        Include sinusoidal encoding of relative coordinates in the edge features.
    pos_encode_dim : int, optional
        Dimension of the positional encoding. Defaults to ``32``.
    pos_encode_range : float, optional
        Range of the positional encoding. Defaults to ``4``.
    reductions : list of str, optional
        Reductions applied over the neighbor dimension. Defaults to ``("mean",)``.
    out_point_type : {"provided", "downsample", "same"}, optional
        Determines the coordinate set on which output features are computed.
    provided_in_channels : int, optional
        Number of channels of the provided query points when ``out_point_type`` is ``"provided"``.
    bias : bool, optional
        Whether linear layers contain biases. Defaults to ``True``.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        neighbor_search_args: RealSearchConfig,
        pooling_reduction: Optional[REDUCTIONS] = None,
        pooling_voxel_size: Optional[float] = None,
        edge_transform_mlp: Optional[nn.Module] = None,
        out_transform_mlp: Optional[nn.Module] = None,
        mlp_block: Union[FeatureMLPBlock, FeatureResidualMLPBlock] = FeatureMLPBlock,
        hidden_dim: Optional[int] = None,
        channel_multiplier: int = 2,
        use_rel_pos: bool = False,
        use_rel_pos_encode: bool = False,
        pos_encode_dim: int = 32,
        pos_encode_range: float = 4,
        reductions: List[REDUCTION_TYPES_STR] = ("mean",),
        out_point_type: Literal["provided", "downsample", "same"] = "same",
        provided_in_channels: Optional[int] = None,
        bias: bool = True,
    ):
        super().__init__()
        assert (
            isinstance(reductions, (tuple, list)) and len(reductions) > 0
        ), f"reductions must be a list or tuple of length > 0, got {reductions}"
        if out_point_type == "provided":
            assert pooling_reduction is None
            assert pooling_voxel_size is None
            assert (
                provided_in_channels is not None
            ), "provided_in_channels must be provided for provided type"
        elif out_point_type == "downsample":
            assert (
                pooling_reduction is not None and pooling_voxel_size is not None
            ), "pooling_reduction and pooling_voxel_size must be provided for downsample type"
            assert (
                provided_in_channels is None
            ), "provided_in_channels must be None for downsample type"
            # print warning if search radius is not \sqrt(3) times the downsample voxel size
            if (
                pooling_voxel_size is not None
                and neighbor_search_args.mode == RealSearchMode.RADIUS
                and neighbor_search_args.radius < pooling_voxel_size * (3**0.5)
            ):
                warnings.warn(
                    f"neighbor search radius {neighbor_search_args.radius} is less than sqrt(3) times the downsample voxel size {pooling_voxel_size}",
                    stacklevel=2,
                )
        elif out_point_type == "same":
            assert (
                pooling_reduction is None and pooling_voxel_size is None
            ), "pooling_reduction and pooling_voxel_size must be None for same type"
            assert (
                provided_in_channels is None
            ), "provided_in_channels must be None for same type"
        if (
            pooling_reduction is not None
            and pooling_voxel_size is not None
            and neighbor_search_args.mode == RealSearchMode.RADIUS
            and pooling_voxel_size > neighbor_search_args.radius
        ):
            raise ValueError(
                f"downsample_voxel_size {pooling_voxel_size} must be <= radius {neighbor_search_args.radius}"
            )

        assert isinstance(neighbor_search_args, RealSearchConfig)
        self.reductions = reductions
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_rel_pos = use_rel_pos
        self.use_rel_pos_encode = use_rel_pos_encode
        self.out_point_feature_type = out_point_type
        self.neighbor_search_args = neighbor_search_args
        self.pooling_reduction = pooling_reduction
        self.pooling_voxel_size = pooling_voxel_size
        self.positional_encoding = SinusoidalEncoding(
            pos_encode_dim, data_range=pos_encode_range
        )
        # When down voxel size is not None, there will be out_point_features will be provided as an additional input
        if provided_in_channels is None:
            provided_in_channels = in_channels
        if hidden_dim is None:
            hidden_dim = channel_multiplier * max(out_channels, in_channels)
        if edge_transform_mlp is None:
            edge_in_channels = in_channels + provided_in_channels
            if use_rel_pos_encode:
                edge_in_channels += pos_encode_dim * 3
            elif use_rel_pos:
                edge_in_channels += 3
            edge_transform_mlp = mlp_block(
                in_channels=edge_in_channels,
                out_channels=out_channels,
                hidden_channels=hidden_dim,
                bias=bias,
            )
        self.edge_transform_mlp = edge_transform_mlp
        self.edge_mlp_in_channels = _get_module_input_channel(edge_transform_mlp)
        if out_transform_mlp is None:
            out_transform_mlp = mlp_block(
                in_channels=out_channels * len(reductions),
                out_channels=out_channels,
                hidden_channels=hidden_dim,
                bias=bias,
            )
        self.out_transform_mlp = out_transform_mlp

    def __repr__(self):
        out_str = f"{self.__class__.__name__}(in_channels={self.in_channels} out_channels={self.out_channels}"
        if self.use_rel_pos_encode:
            out_str += f" rel_pos_encode={self.use_rel_pos_encode}"
        if self.pooling_reduction is not None:
            out_str += f" pooling={self.pooling_reduction}"
        if self.neighbor_search_args is not None:
            out_str += f" neighbor={self.neighbor_search_args}"
        out_str += ")"
        return out_str

    def forward(
        self,
        in_pc: Points,
        query_pc: Optional[Points] = None,
    ) -> Points:
        """
        When out_point_features is None, the output will be generated on the
        in_point_features.batched_coordinates.
        """
        if self.out_point_feature_type == "provided":
            assert (
                query_pc is not None
            ), "query_point_features must be provided for the provided type"
        elif self.out_point_feature_type == "downsample":
            assert query_pc is None
            query_pc = in_pc.voxel_downsample(
                self.pooling_voxel_size,
                reduction=self.pooling_reduction,
            )
        elif self.out_point_feature_type == "same":
            assert query_pc is None
            query_pc = in_pc

        in_num_channels = in_pc.num_channels
        query_num_channels = query_pc.num_channels
        assert (
            in_num_channels
            + query_num_channels
            + self.use_rel_pos_encode * self.positional_encoding.num_channels * 3
            + (not self.use_rel_pos_encode) * self.use_rel_pos * 3
            == self.edge_mlp_in_channels
        ), f"input features shape {in_pc.feature_tensor.shape} and query feature shape {query_pc.feature_tensor.shape} does not match the edge_transform_mlp input channels {self.edge_mlp_in_channels}"

        # Get the neighbors
        neighbors = in_pc.neighbors(
            query_coords=query_pc.batched_coordinates,
            search_args=self.neighbor_search_args,
        )
        neighbor_indices = neighbors.neighbor_indices.long().view(-1)
        neighbor_row_splits = neighbors.neighbor_row_splits
        num_reps = neighbor_row_splits[1:] - neighbor_row_splits[:-1]

        # repeat the self features using num_reps
        rep_in_features = in_pc.feature_tensor[neighbor_indices]
        self_features = torch.repeat_interleave(
            query_pc.feature_tensor.view(-1, query_num_channels).contiguous(),
            num_reps,
            dim=0,
        )
        edge_features = [rep_in_features, self_features]
        if self.use_rel_pos or self.use_rel_pos_encode:
            in_rep_vertices = in_pc.coordinate_tensor.view(-1, 3)[neighbor_indices]
            self_vertices = torch.repeat_interleave(
                query_pc.coordinate_tensor.view(-1, 3).contiguous(),
                num_reps,
                dim=0,
            )
            rel_coords = in_rep_vertices.view(-1, 3) - self_vertices.view(-1, 3)
            if self.use_rel_pos_encode:
                edge_features.append(self.positional_encoding(rel_coords))
            elif self.use_rel_pos:
                edge_features.append(rel_coords)
        edge_features = torch.cat(edge_features, dim=1)
        edge_features = self.edge_transform_mlp(edge_features)
        # if in_weight is not None:
        #     assert in_weight.shape[0] == in_point_features.features.shape[0]
        #     rep_weights = in_weight[neighbor_indices]
        #     edge_features = edge_features * rep_weights.squeeze().unsqueeze(-1)

        out_features = []
        for reduction in self.reductions:
            out_features.append(
                row_reduction(edge_features, neighbor_row_splits, reduction=reduction)
            )
        out_features = torch.cat(out_features, dim=-1)
        out_features = self.out_transform_mlp(out_features)

        return Points(
            batched_coordinates=Coords(
                batched_tensor=query_pc.coordinate_tensor,
                offsets=query_pc.offsets,
            ),
            batched_features=out_features,
            **query_pc.extra_attributes,
        )
