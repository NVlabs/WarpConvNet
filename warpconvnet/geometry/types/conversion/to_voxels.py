from typing import Optional, Union, Literal


def points_to_voxels(
    points: "Points",
    voxel_size: float,
    reduction: Union["REDUCTIONS", "REDUCTION_TYPES_STR"] = "random",
    unique_method: Literal["torch", "ravel"] = "torch",
    return_to_unique: bool = False,
) -> "Voxels":
    """
    Convert the point collection to a spatially sparse tensor.
    """
    from warpconvnet.nn.functional.point_pool import point_pool

    st = point_pool(
        points,
        reduction=reduction,
        downsample_voxel_size=voxel_size,
        return_type="voxel",
        unique_method=unique_method,
        return_to_unique=return_to_unique,
    )
    return st
