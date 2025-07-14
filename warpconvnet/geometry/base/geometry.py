# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field
from typing import Any, Dict, Union, Optional

import torch
from torch import Tensor

from warpconvnet.geometry.base.features import Features
from warpconvnet.geometry.features.ops.convert import to_batched_features
from warpconvnet.geometry.coords.ops.batch_index import batch_indexed_coordinates

from .coords import Coords


def amp_aware_dtype(func):
    """Decorator to handle dtype conversion based on autocast context.

    Usage:
        @amp_aware_dtype
        def features(self) -> Tensor:
            return self.batched_features.batched_tensor
    """

    def wrapper(self, *args, **kwargs):
        tensor = func(self, *args, **kwargs)
        if torch.is_autocast_enabled():
            amp_dtype = torch.get_autocast_gpu_dtype()
            if amp_dtype is not None:
                return tensor.to(dtype=amp_dtype)
        return tensor

    return wrapper


@dataclass
class Geometry:
    """A base class for all geometry objects such as sparse voxels, points, etc.

    This class provides a unified interface for handling different types of geometric data
    with associated features. It supports both concatenated and padded feature representations.

    Args:
        batched_coordinates (Coords): Coordinate information for the geometry.
        batched_features (Union[CatFeatures, PadFeatures, Tensor]): Feature data associated with the coordinates.
        **kwargs: Additional arguments to be stored as extra attributes.

    Properties:
        num_spatial_dims (int): Number of spatial dimensions in the coordinates.
        coordinate_tensor (Tensor): The raw coordinate tensor.
        features (Tensor): The raw feature tensor.
        device: The device where the tensors are stored.
        num_channels (int): Number of feature channels.
        batch_size (int): Size of the batch.
        dtype: Data type of the features.
    """

    batched_coordinates: Coords
    batched_features: Features
    _extra_attributes: Dict[str, Any] = field(default_factory=dict, init=True)  # Store extra args

    def __init__(
        self,
        batched_coordinates: Union[Coords, Tensor],
        batched_features: Union["CatFeatures", "PadFeatures", Tensor],  # noqa: F821
        **kwargs,
    ):
        offsets = kwargs.pop("offsets", None)
        if isinstance(batched_coordinates, Tensor):
            assert (
                offsets is not None
            ), "offsets must be provided when batched_coordinates is a tensor"
            batched_coordinates = Coords(batched_coordinates, offsets)
        self.batched_coordinates = batched_coordinates
        self.batched_features = to_batched_features(
            batched_features,
            batched_coordinates.offsets,
            device=kwargs.get("device", None),
        )

        assert (batched_coordinates.offsets == self.batched_features.offsets).all()
        # Extra arguments for subclasses
        # First check _extra_attributes in kwargs. This happens when we use dataclasses.replace
        if "_extra_attributes" in kwargs:
            attr = kwargs.pop("_extra_attributes")
            assert isinstance(attr, dict), f"_extra_attributes must be a dictionary, got {attr}"
            # Update kwargs
            for k, v in attr.items():
                kwargs[k] = v
        self._extra_attributes = kwargs

    def __getitem__(self, idx: int) -> "Geometry":
        coords = self.batched_coordinates[idx]
        features = self.batched_features[idx]
        return self.__class__(
            batched_coordinates=coords,
            batched_features=features,
            offsets=torch.tensor([0, len(coords)]),
            **self._extra_attributes,
        )

    def to(
        self,
        device: Optional[str | torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> "Geometry":
        """Move the geometry to the specified device.

        Args:
            device (str): Target device (e.g., 'cuda', 'cpu') or dtype (e.g., torch.float16, torch.bfloat16).

        Returns:
            Geometry: A new Geometry instance on the target device.
        """
        if device is None:
            device = self.device
        return self.__class__(
            batched_coordinates=self.batched_coordinates.to(device=device),
            batched_features=self.batched_features.to(device=device, dtype=dtype),
            **self._extra_attributes,
        )

    @property
    def num_spatial_dims(self) -> int:
        return self.batched_coordinates.num_spatial_dims

    @property
    def coordinate_tensor(self) -> Tensor:
        return self.batched_coordinates.batched_tensor

    @property
    def coordinates(self) -> Tensor:
        return self.batched_coordinates.batched_tensor

    @property
    def nested_coordinates(self) -> Tensor:
        return self.batched_coordinates.to_nested()

    @property
    def batch_indexed_coordinates(self) -> Tensor:
        return batch_indexed_coordinates(self.coordinate_tensor, self.offsets)

    @property
    @amp_aware_dtype
    def feature_tensor(self) -> Tensor:
        return self.batched_features.batched_tensor

    @property
    @amp_aware_dtype
    def features(self) -> Tensor:
        return self.batched_features.batched_tensor

    @property
    @amp_aware_dtype
    def nested_features(self) -> Tensor:
        return self.batched_features.to_nested()

    @property
    @amp_aware_dtype
    def padded_features(self) -> "PadFeatures":  # noqa: F821
        """
        Explicitly convert batched features to padded features if necessary
        """
        if self.batched_features.is_pad:
            return self.batched_features
        elif self.batched_features.is_cat:
            return self.batched_features.to_pad()
        else:
            raise ValueError(f"Unsupported features type: {type(self.batched_features)}")

    def to_pad(self, pad_multiple: Optional[int] = None) -> "Geometry":
        """Convert features to padded format.

        Args:
            pad_multiple (Optional[int]): Padding multiple for feature dimensions.

        Returns:
            Geometry: A new Geometry instance with padded features.
        """
        if self.batched_features.is_pad and self.batched_features.pad_multiple == pad_multiple:
            return self
        return self.replace(batched_features=self.batched_features.to_pad(pad_multiple))

    def to_cat(self) -> "Geometry":
        """Convert features to concatenated format.

        Returns:
            Geometry: A new Geometry instance with concatenated features.
        """
        if self.batched_features.is_cat:
            return self
        return self.replace(batched_features=self.batched_features.to_cat())

    @property
    def offsets(self):
        return self.batched_features.offsets

    @property
    def device(self):
        return self.batched_features.device

    @property
    def num_channels(self):
        return self.batched_features.num_channels

    @property
    def batch_size(self) -> int:
        return len(self.offsets) - 1

    @property
    def shape(self):
        raise ValueError("Specify shape in subclass")

    @property
    def dtype(self):
        return self.batched_features.dtype

    def sort(self):
        raise NotImplementedError

    def _apply_feature_transform(self, feature_transform_fn):
        """Apply a transformation function to the features.

        Args:
            feature_transform_fn (callable): Function to transform the features.

        Returns:
            Geometry: A new Geometry instance with transformed features.
        """
        out_features = feature_transform_fn(self.feature_tensor)
        return self.replace(batched_features=out_features)

    def half(self):
        return self._apply_feature_transform(lambda x: x.half())

    def float(self):
        return self._apply_feature_transform(lambda x: x.float())

    def double(self):
        return self._apply_feature_transform(lambda x: x.double())

    def binary_op(self, value: object, op: str) -> "Geometry":
        if isinstance(value, Geometry):
            assert self.equal_shape(value), f"Shapes do not match. {self} != {value}"
            return self._apply_feature_transform(lambda x: getattr(x, op)(value.feature_tensor))
        elif isinstance(value, (int, float)) or (torch.is_tensor(value) and value.numel() == 1):
            return self._apply_feature_transform(lambda x: getattr(x, op)(value))
        elif isinstance(value, torch.Tensor):
            assert self.equal_shape(value)
            return self._apply_feature_transform(lambda x: getattr(x, op)(value))
        else:
            raise NotImplementedError

    def __add__(self, value: object) -> "Geometry":
        return self.binary_op(value, "__add__")

    def __sub__(self, value: object) -> "Geometry":
        return self.binary_op(value, "__sub__")

    def __mul__(self, value: object) -> "Geometry":
        return self.binary_op(value, "__mul__")

    def __truediv__(self, value: object) -> "Geometry":
        return self.binary_op(value, "__truediv__")

    def __floordiv__(self, value: object) -> "Geometry":
        return self.binary_op(value, "__floordiv__")

    def __mod__(self, value: object) -> "Geometry":
        return self.binary_op(value, "__mod__")

    def __pow__(self, value: object) -> "Geometry":
        return self.binary_op(value, "__pow__")

    def equal_rigorous(self, value: object) -> bool:
        raise NotImplementedError

    def equal_shape(self, value: object) -> bool:
        return self.batched_coordinates.equal_shape(
            value.batched_coordinates
        ) and self.batched_features.equal_shape(value.batched_features)

    def __str__(self) -> str:
        """Short representation of the object."""
        return f"{self.__class__.__name__}(feature_shape={self.feature_tensor.shape}, coords_shape={self.batched_coordinates.shape})"

    def __repr__(self) -> str:
        """Detailed representation of the object."""
        out_str = f"{self.__class__.__name__}(offsets={self.offsets.tolist()}, feature_shape={self.feature_tensor.shape}, coords_shape={self.batched_coordinates.shape}, device={self.device}, dtype={self.feature_tensor.dtype}"
        if self._extra_attributes:
            out_dict = {k: v for k, v in self._extra_attributes.items() if v is not None}
            # if out_dict has values, add it to the string
            if out_dict:
                out_str += ", "
                out_str += ", ".join([f"{k}={v}" for k, v in out_dict.items()])
        out_str += ")"
        return out_str

    def __len__(self) -> int:
        return len(self.batched_coordinates)

    def numel(self):
        return self.offsets[-1] * self.num_channels

    @property
    def extra_attributes(self):
        return self._extra_attributes.copy()

    @property
    def cache(self):
        return self._extra_attributes.get("_cache")

    def replace(
        self,
        batched_coordinates: Optional[Coords] = None,
        batched_features: Optional[
            Union["CatFeatures", "PadFeatures", Tensor]  # noqa: F821
        ] = None,
        **kwargs,
    ):
        """Create a new instance with replaced coordinates and/or features.

        Args:
            batched_coordinates (Optional[Coords]): New coordinates to use.
            batched_features (Optional[Union[CatFeatures, PadFeatures, Tensor]]): New features to use.
            **kwargs: Additional arguments to update in extra attributes.

        Returns:
            Geometry: A new Geometry instance with the specified replacements.
        """
        # Combine extra attributes and kwargs
        if "_extra_attributes" in kwargs:  # flatten extra attributes
            _extra_attributes = kwargs.pop("_extra_attributes")
            kwargs = {**_extra_attributes, **kwargs}

        assert "batched_features" not in kwargs, "Use features instead of batched_features"

        new_coords = (
            batched_coordinates if batched_coordinates is not None else self.batched_coordinates
        )
        new_features = batched_features if batched_features is not None else self.batched_features
        if isinstance(new_features, torch.Tensor):
            new_features = to_batched_features(new_features, new_coords.offsets)

        new_kwargs = {**self._extra_attributes, **kwargs}
        return self.__class__(new_coords, new_features, **new_kwargs)
