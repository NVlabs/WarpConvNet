# Core

The top-level `warpconvnet` package re-exports the main subpackages and
sets global configuration from environment variables at import time.

## Subpackages

| Package                 | Description                                                                              |
| ----------------------- | ---------------------------------------------------------------------------------------- |
| `warpconvnet.geometry`  | Geometry containers (`Points`, `Voxels`, `Grid`, `FactorGrid`) and coordinate operations |
| `warpconvnet.nn`        | Neural network modules and functional APIs for sparse/point convolutions                 |
| `warpconvnet.ops`       | Low-level tensor operations (reductions, sampling)                                       |
| `warpconvnet.utils`     | Internal utilities (logging, caching, timing)                                            |
| `warpconvnet.constants` | Environment-variable configuration and algorithm selection                               |

## Import example

```python
import warpconvnet

# Geometry types
from warpconvnet.geometry.types.points import Points
from warpconvnet.geometry.types.voxels import Voxels

# Neural network modules
from warpconvnet.nn.modules.sparse_conv import SparseConv3d
from warpconvnet.nn.modules.point_conv import PointConv
from warpconvnet.nn.modules.sequential import Sequential

# Operations
from warpconvnet.ops.reductions import REDUCTIONS, row_reduction
from warpconvnet.ops.sampling import farthest_point_sampling
```
