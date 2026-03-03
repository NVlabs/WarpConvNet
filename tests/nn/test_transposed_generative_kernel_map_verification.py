
import torch
import pytest
from warpconvnet.geometry.types.voxels import Voxels
from warpconvnet.nn.functional.sparse_conv.helper import (
    generate_output_coords_and_kernel_map,
    STRIDED_CONV_MODE,
)
from warpconvnet.geometry.coords.ops.serialization import POINT_ORDERING

def test_transposed_generative_kernel_map_simple_manual():
    """
    Manually verify the kernel map for transposed generative convolution.
    
    Scenario:
    - 1 batch
    - Input: 1 point at (1, 1, 1)
    - Kernel: 3x3x3
    - Stride: 2
    - Dilation: 1
    - Transposed: True
    - Generative: True
    
    Expected Output:
    - Upsampled input: (1,1,1) * (2,2,2) = (2,2,2)
    - Output coords: 27 points centered at (2,2,2) (from 1,1,1 to 3,3,3)
    - Kernel map:
        - For each kernel offset k, there should be a correspondence.
        - Since input is just one point (index 0), in_maps should be all 0s.
        - out_maps should point to the index of the neighbor in the output coords.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Setup Input
    # Single point at (1,1,1)
    coords = torch.tensor([[1, 1, 1]], dtype=torch.int32, device=device)
    features = torch.tensor([[1.0]], dtype=torch.float32, device=device)
    offsets = torch.tensor([0, 1], dtype=torch.int32, device=device)
    
    input_voxels = Voxels(
        batched_coordinates=coords,
        batched_features=features,
        offsets=offsets,
        device=device
    )
    
    # 2. Parameters
    kernel_size = (3, 3, 3)
    kernel_dilation = (1, 1, 1)
    stride = (2, 2, 2)
    
    # 3. Run generate_output_coords_and_kernel_map
    batch_indexed_out_coords, out_offsets, kernel_map = generate_output_coords_and_kernel_map(
        input_sparse_tensor=input_voxels,
        kernel_size=kernel_size,
        kernel_dilation=kernel_dilation,
        stride=stride,
        generative=True,
        transposed=True,
        stride_mode=STRIDED_CONV_MODE.STRIDE_ONLY, # default
        order=POINT_ORDERING.MORTON_XYZ, # deterministic order for output
    )
    
    # 4. Verification
    
    # A. Output Coordinates
    # Should be 27 points.
    assert batch_indexed_out_coords.shape[0] == 27
    assert out_offsets[0] == 0
    assert out_offsets[1] == 27
    
    # Check that (2,2,2) is in output (scaled input coords)
    # Neighbors range from (1,1,1) to (3,3,3)
    # Remove batch index (which is 0)
    out_xyz = batch_indexed_out_coords[:, 1:]
    
    # Check range
    assert out_xyz.min() == 1
    assert out_xyz.max() == 3
    
    # Check specific center point existence
    center_point = torch.tensor([2, 2, 2], dtype=torch.int32, device=device)
    matches = torch.all(out_xyz == center_point, dim=1)
    assert torch.any(matches)
    center_idx = torch.where(matches)[0].item()
    
    # B. Kernel Map
    # kernel_map.offsets tells us how many matches per kernel element.
    # For a dense 3x3x3 block generated from 1 point, every kernel weight should map the input point to exactly one output point.
    # So each kernel offset should have 1 match.
    
    # kernel_map.offsets size is num_kernels + 1 = 27 + 1 = 28
    assert kernel_map.offsets.shape[0] == 27 + 1
    
    # Each interval should be size 1
    counts = kernel_map.offsets[1:] - kernel_map.offsets[:-1]
    assert torch.all(counts == 1)
    
    # C. Check Mapping Direction
    # We want to ensure in_maps points to Input and out_maps points to Output.
    # Input has 1 point (index 0).
    # Output has 27 points (indices 0..26).
    
    all_in_indices = kernel_map.in_maps
    all_out_indices = kernel_map.out_maps
    
    # Since there is only 1 input point at index 0, all in_maps entries must be 0.
    assert torch.all(all_in_indices == 0)
    
    # out_maps should cover all output indices 0..26 exactly once (since each kernel weight maps to a unique output location).
    unique_out_indices = torch.unique(all_out_indices)
    assert unique_out_indices.shape[0] == 27
    assert unique_out_indices.min() == 0
    assert unique_out_indices.max() == 26
    
    # D. Check Center Kernel Mapping
    # The center kernel weight (index 13) should map the input point to the center output point (2,2,2).
    center_kernel_k = 13
    start = kernel_map.offsets[center_kernel_k]
    # end = kernel_map.offsets[center_kernel_k + 1] # = start + 1
    
    mapped_out_idx = kernel_map.out_maps[start].item()
    mapped_in_idx = kernel_map.in_maps[start].item()
    
    assert mapped_in_idx == 0
    assert mapped_out_idx == center_idx
    
    print("Simple manual test passed!")

def test_transposed_generative_kernel_map_two_points_overlap():
    """
    Verify kernel map with two close points that share some output neighbors.
    
    Input: (0,0,0) and (0,0,1)
    Kernel: 3x3x3
    Output: Union of 3x3x3 centered at (0,0,0) and (0,0,1).
    Overlap region: z=0 and z=1 planes overlap? 
    (0,0,0) ranges z: -1 to 1
    (0,0,1) ranges z: 0 to 2
    Overlap at z=0 and z=1.
    
    We check if the mapping correctly identifies (in_idx, out_idx) pairs.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Input: 2 points
    coords = torch.tensor([
        [0, 0, 0],
        [0, 0, 1]
    ], dtype=torch.int32, device=device)
    # offsets
    offsets = torch.tensor([0, 2], dtype=torch.int32, device=device)
    
    input_voxels = Voxels(
        batched_coordinates=coords,
        batched_features=torch.zeros((2, 1), device=device),
        offsets=offsets,
        device=device
    )
    
    batch_indexed_out_coords, out_offsets, kernel_map = generate_output_coords_and_kernel_map(
        input_sparse_tensor=input_voxels,
        kernel_size=(3, 3, 3),
        kernel_dilation=(1, 1, 1),
        stride=(1, 1, 1),
        generative=True,
        transposed=True
    )
    
    # Validate in_maps are within [0, 1]
    assert kernel_map.in_maps.max() <= 1
    assert kernel_map.in_maps.min() >= 0
    
    # Validate out_maps are within valid output range
    num_out = batch_indexed_out_coords.shape[0]
    assert kernel_map.out_maps.max() < num_out
    
    # Check a specific kernel weight, e.g. center (0,0,0) which is index 13 (in 0..26)
    # It should map in[0]->out_at(0,0,0) and in[1]->out_at(0,0,1)
    center_kernel_idx = 13
    start = kernel_map.offsets[center_kernel_idx]
    end = kernel_map.offsets[center_kernel_idx+1]
    
    # Should have 2 matches for the center kernel (one for each input point)
    
    assert (end - start) == 2
    
    indices_in = kernel_map.in_maps[start:end]
    indices_out = kernel_map.out_maps[start:end]
    
    # Ensure both input points are represented
    assert 0 in indices_in
    assert 1 in indices_in
    
    # Get output coord for input 0
    idx_out_for_0 = indices_out[indices_in == 0].item()
    coord_out_0 = batch_indexed_out_coords[idx_out_for_0, 1:] # remove batch
    assert torch.equal(coord_out_0, torch.tensor([0, 0, 0], dtype=torch.int32, device=device))
    
    # Get output coord for input 1
    idx_out_for_1 = indices_out[indices_in == 1].item()
    coord_out_1 = batch_indexed_out_coords[idx_out_for_1, 1:]
    assert torch.equal(coord_out_1, torch.tensor([0, 0, 1], dtype=torch.int32, device=device))

    print("Two points overlap test passed!")

if __name__ == "__main__":
    pass
