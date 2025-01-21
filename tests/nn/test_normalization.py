import pytest
import torch
import warp as wp

from warpconvnet.geometry.types.points import Points
from warpconvnet.geometry.types.voxels import Voxels
from warpconvnet.nn.modules.normalizations import LayerNorm, RMSNorm


@pytest.fixture
def setup_points():
    """Setup test points with random coordinates."""
    wp.init()
    torch.manual_seed(0)
    device = torch.device("cuda:0")

    B, min_N, max_N, C = 3, 1000, 10000, 7
    Ns = torch.randint(min_N, max_N, (B,))
    coords = [torch.rand((N, 3)) for N in Ns]
    features = [torch.rand((N, C)).requires_grad_() for N in Ns]
    points = Points(coords, features).to(device)

    # Convert to sparse voxels
    voxel_size = 0.01
    voxels = points.to_sparse(voxel_size)

    return points, voxels


def test_rms_norm_points(setup_points):
    """Test RMSNorm with point cloud input."""
    points: Points = setup_points[0]
    device = points.device

    # Create normalization layer
    rms_norm = RMSNorm(points.num_channels).to(device)

    # Forward pass
    normed_pc = rms_norm(points)

    # Verify output properties
    assert normed_pc.batch_size == points.batch_size
    assert normed_pc.num_channels == points.num_channels

    # Test gradient flow
    normed_pc.features.sum().backward()
    assert rms_norm.norm.weight.grad is not None


def test_rms_norm_voxels(setup_points):
    """Test RMSNorm with voxel input."""
    voxels: Voxels = setup_points[1]
    device = voxels.device

    # Create normalization layer
    rms_norm = RMSNorm(voxels.num_channels).to(device)

    # Forward pass
    normed_voxels = rms_norm(voxels)

    # Verify output properties
    assert normed_voxels.batch_size == voxels.batch_size
    assert normed_voxels.num_channels == voxels.num_channels

    # Test gradient flow
    normed_voxels.features.sum().backward()
    assert rms_norm.norm.weight.grad is not None


def test_layer_norm_points(setup_points):
    """Test LayerNorm with point cloud input."""
    points: Points = setup_points[0]
    device = points.device

    # Create normalization layer
    layer_norm = LayerNorm(points.num_channels).to(device)

    # Forward pass
    normed_pc = layer_norm(points)

    # Verify output properties
    assert normed_pc.batch_size == points.batch_size
    assert normed_pc.num_channels == points.num_channels

    # Test gradient flow
    normed_pc.features.sum().backward()
    assert layer_norm.norm.weight.grad is not None


def test_layer_norm_voxels(setup_points):
    """Test LayerNorm with voxel input."""
    voxels: Voxels = setup_points[1]
    device = voxels.device

    # Create normalization layer
    layer_norm = LayerNorm(voxels.num_channels).to(device)

    # Forward pass
    normed_voxels = layer_norm(voxels)

    # Verify output properties
    assert normed_voxels.batch_size == voxels.batch_size
    assert normed_voxels.num_channels == voxels.num_channels

    # Test gradient flow
    normed_voxels.features.sum().backward()
    assert layer_norm.norm.weight.grad is not None
