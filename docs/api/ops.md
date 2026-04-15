# Operations

Low-level tensor operations for sparse and segmented data.

Defined in `warpconvnet/ops/`.

## Reductions

```python
from warpconvnet.ops.reductions import REDUCTIONS, row_reduction
```

### `REDUCTIONS`

Enum of supported reduction types:

| Member              | Value      | Description                      |
| ------------------- | ---------- | -------------------------------- |
| `REDUCTIONS.MIN`    | `"min"`    | Element-wise minimum per segment |
| `REDUCTIONS.MAX`    | `"max"`    | Element-wise maximum per segment |
| `REDUCTIONS.MEAN`   | `"mean"`   | Mean per segment                 |
| `REDUCTIONS.SUM`    | `"sum"`    | Sum per segment                  |
| `REDUCTIONS.MUL`    | `"mul"`    | Product per segment              |
| `REDUCTIONS.VAR`    | `"var"`    | Variance per segment             |
| `REDUCTIONS.STD`    | `"std"`    | Standard deviation per segment   |
| `REDUCTIONS.RANDOM` | `"random"` | Random sample per segment        |

### `row_reduction`

```python
row_reduction(
    features: Tensor,    # (N, F) — concatenated features
    row_offsets: Tensor,  # (M+1,) — segment boundaries
    reduction: REDUCTIONS,
    eps: float = 1e-6,
) -> Tensor  # (M, F) — one row per segment
```

Apply a reduction over contiguous segments of a concatenated feature tensor.
`row_offsets` uses CSR format: segment `i` spans rows
`row_offsets[i]` to `row_offsets[i+1]`.

## Sampling

```python
from warpconvnet.ops.sampling import farthest_point_sampling
```

### `farthest_point_sampling`

```python
farthest_point_sampling(
    points: Tensor,   # (N, 3) — packed point cloud
    offsets: Tensor,   # (B+1,) — batch offsets
    K: int,            # number of points to sample per batch item
) -> Tensor  # (B*K,) int32 — global indices of selected points
```

Iteratively selects the point farthest from all previously selected points,
producing a well-spread subset. Runs on GPU via a CUDA kernel.
