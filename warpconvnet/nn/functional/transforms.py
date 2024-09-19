from typing import Callable, Sequence

import torch
import torch.nn.functional as F

from warpconvnet.geometry.base_geometry import BatchedFeatures, BatchedSpatialFeatures


def apply_feature_transform(
    input: BatchedSpatialFeatures,
    transform: Callable,
):
    assert isinstance(
        input, BatchedSpatialFeatures
    ), f"Expected BatchedSpatialFeatures, got {type(input)}"
    return input.replace(
        batched_features=BatchedFeatures(transform(input.feature_tensor), offsets=input.offsets),
    )


def create_activation_function(torch_func):
    def wrapper(input: BatchedSpatialFeatures):
        return apply_feature_transform(input, torch_func)

    return wrapper


# Instantiate common activation functions
relu = create_activation_function(F.relu)
gelu = create_activation_function(F.gelu)
silu = create_activation_function(F.silu)
tanh = create_activation_function(F.tanh)
sigmoid = create_activation_function(F.sigmoid)
leaky_relu = create_activation_function(F.leaky_relu)
elu = create_activation_function(F.elu)
softmax = create_activation_function(F.softmax)
log_softmax = create_activation_function(F.log_softmax)


# Normalization functions
def create_norm_function(torch_norm_func):
    def wrapper(input: BatchedSpatialFeatures, *args, **kwargs):
        return apply_feature_transform(input, lambda x: torch_norm_func(x, *args, **kwargs))

    return wrapper


# Instantiate common normalization functions
layer_norm = create_norm_function(F.layer_norm)
# layer_norm(input, normalized_shape, weight=None, bias=None, eps=1e-5)
batch_norm = create_norm_function(F.batch_norm)
# batch_norm(input, running_mean, running_var, weight=None, bias=None, training=False, momentum=0.1, eps=1e-5)
instance_norm = create_norm_function(F.instance_norm)
# instance_norm(input, running_mean=None, running_var=None, weight=None, bias=None, use_input_stats=True, momentum=0.1, eps=1e-5)
group_norm = create_norm_function(F.group_norm)
# group_norm(input, num_groups, weight=None, bias=None, eps=1e-5)


# Concatenation
def cat(*inputs: BatchedSpatialFeatures, dim: int = 1):
    # If called with a single sequence argument, unpack it
    if len(inputs) == 1 and isinstance(inputs[0], Sequence):
        inputs = inputs[0]
    assert all(
        isinstance(input, BatchedSpatialFeatures) for input in inputs
    ), f"Expected all inputs to be BatchedSpatialFeatures, got {type(inputs)}"
    assert all(
        torch.allclose(input.offsets, inputs[0].offsets) for input in inputs
    ), "All inputs must have the same offsets"
    return inputs[0].replace(
        batched_features=BatchedFeatures(
            torch.cat([input.feature_tensor for input in inputs], dim=dim),
            offsets=inputs[0].offsets,
        )
    )
