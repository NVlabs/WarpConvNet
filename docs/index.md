# WarpConvNet

Welcome to the WarpConvNet documentation.

WarpConvNet is a high-performance 3D deep learning library built on NVIDIA's Warp framework. It provides efficient implementations of point cloud processing, sparse voxel convolutions, attention mechanisms for 3D data, and geometric operations.

## 🚀 Quick Start

```python
import warpconvnet as wcn

# Create geometry from point cloud
geometry = wcn.geometry.Points(coords, features)

# Voxelize for sparse convolution
voxels = geometry.voxelize(voxel_size=0.05)

# Apply sparse convolution
features = voxels.sparse_conv(kernel_size=3)
```

## 📚 Documentation Sections

- **Getting Started**: Installation and quick start guides
- **User Guide**: Comprehensive tutorials and concepts
- **API Reference**: Complete API documentation
- **Examples**: Real-world usage examples
- **Diagrams**: Architecture and data flow visualizations

## 🔗 Links

- [GitHub Repository](https://github.com/nvidia/warpconvnet)
- [Installation Guide](getting_started/installation.md)
- [Quick Start Tutorial](getting_started/quickstart.md)
