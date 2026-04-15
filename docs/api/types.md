# Types

Type aliases used across the WarpConvNet API.

Defined in `warpconvnet/types.py`.

## `NestedTensor`

Alias for `torch.Tensor`. Used to annotate tensors that represent
nested/ragged data (e.g., a batch of variable-length point clouds packed
into a single tensor).

```python
from warpconvnet.types import NestedTensor
```

## `IterableTensor`

```python
IterableTensor = Union[Tensor, List[Tensor], Tuple[Tensor, ...], NestedTensor]
```

Annotates arguments that accept either a single tensor or a sequence of
tensors. Used in APIs that can take a list of per-sample tensors or a
pre-concatenated batch tensor.

```python
from warpconvnet.types import IterableTensor
```
