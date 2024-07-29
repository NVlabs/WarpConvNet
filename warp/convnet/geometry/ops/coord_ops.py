from typing import Optional

import torch
from jaxtyping import Float
from torch import Tensor

from warp.convnet.geometry.ops.neighbor_search_continuous import NeighborSearchResult

__all__ = [
    "relative_coords",
]


def relative_coords(
    neighbor_coordinates: Float[Tensor, "N D"],  # noqa: F722,F821
    neighbor_search_result: NeighborSearchResult,
    query_coordinates: Optional[Float[Tensor, "M D"]] = None,  # noqa: F722,F821
):
    if query_coordinates is None:
        query_coordinates = neighbor_coordinates
    target = neighbor_coordinates[neighbor_search_result.neighbors_index]
    # repeat query coordinates for each neighbor
    rs = neighbor_search_result.neighbors_row_splits
    num_reps = rs[1:] - rs[:-1]
    source = torch.repeat_interleave(query_coordinates, num_reps, dim=0)
    assert source.shape == target.shape, f"Shape mismatch {source.shape} != {target.shape}"
    return target - source
