# Gradient Checkpointing

Gradient checkpointing reduces activation memory by discarding selected
intermediate tensors during the forward pass and recomputing them during
backpropagation. It trades additional compute for a lower training-memory
footprint; it does not reduce parameter, optimizer-state, or gradient memory.

WarpConvNet provides a common checkpointing contract for sparse convolution,
attention, and other geometry-aware blocks:

```python
from warpconvnet.nn.modules import (
    GradientCheckpointingMixin,
    GradientCheckpointingModelMixin,
    configure_gradient_checkpointing,
)
```

The API follows the Hugging Face `PreTrainedModel` conventions without making
Transformers a WarpConvNet dependency.

## Quick start

A checkpoint-aware model exposes model-level controls:

```python
model.train()
model.gradient_checkpointing_enable()

assert model.supports_gradient_checkpointing
assert model.is_gradient_checkpointing

# Training loop ...

model.gradient_checkpointing_disable()
```

The standalone helper provides the same tree walk for an `nn.Module` container
that does not inherit `GradientCheckpointingModelMixin`:

```python
from warpconvnet.nn.modules import configure_gradient_checkpointing

num_boundaries = configure_gradient_checkpointing(model, enable=True)
print(f"Enabled {num_boundaries} checkpoint boundaries")
```

By default, the helper raises if it finds no checkpoint-aware descendants. This
guards against a silent no-op:

```python
num_boundaries = configure_gradient_checkpointing(
    model,
    enable=True,
    strict=False,  # Return 0 instead of raising.
)
```

Checkpointing is active only while all three conditions hold:

1. the boundary has been enabled;
2. the module is in training mode; and
3. PyTorch gradient recording is enabled.

Evaluation and `torch.no_grad()` therefore use the ordinary forward path.

## Hugging Face Trainer

The standard Trainer flag can enable WarpConvNet boundaries:

```python
from transformers import TrainingArguments

args = TrainingArguments(
    output_dir="outputs/run",
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
)
```

This works in either of these arrangements:

- the model passed to Trainer inherits `GradientCheckpointingModelMixin`; or
- a WarpConvNet model is a descendant of a Hugging Face `PreTrainedModel`
  whose checkpointing tree walk reaches the WarpConvNet blocks.

`use_reentrant=False` is required. WarpConvNet forwards may carry tensors
inside `Geometry`, `Points`, or `Voxels` objects, which the older reentrant
checkpoint implementation cannot handle reliably. The WarpConvNet helper uses
non-reentrant mode by default and rejects an explicitly reentrant PyTorch
checkpoint function.

Recent Transformers versions also default to non-reentrant checkpointing, so
`gradient_checkpointing=True` alone is sufficient there. Supplying the kwargs
explicitly makes the requirement clear and remains safer across Transformers
versions.

## Defining a checkpoint boundary

Checkpointing cannot safely infer arbitrary boundaries from an `nn.Module`
tree. A leaf block must declare where its forward computation can be replayed.

```python
import torch.nn as nn

from warpconvnet.nn.modules import (
    GradientCheckpointingMixin,
    GradientCheckpointingModelMixin,
)


class ResidualBlock(GradientCheckpointingMixin, nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self._init_gradient_checkpointing()
        self.layers = nn.Sequential(
            nn.Linear(channels, 4 * channels),
            nn.GELU(),
            nn.Linear(4 * channels, channels),
        )

    def _forward(self, x):
        return x + self.layers(x)

    def forward(self, x):
        return self._gradient_checkpointed_call(self._forward, x)


class Network(GradientCheckpointingModelMixin, nn.Module):
    def __init__(self, channels: int, depth: int):
        super().__init__()
        self.blocks = nn.ModuleList(
            ResidualBlock(channels) for _ in range(depth)
        )

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x
```

The mixins do not add parameters or persistent buffers, so this integration
does not change state-dict keys. Existing constructors may optionally expose a
local default:

```python
self._init_gradient_checkpointing(use_checkpoint=True)
```

The model-level API is generally preferable because it enables every declared
boundary consistently.

## Choosing granularity

Checkpoint a repeated compute-heavy block rather than every small operation.
Very small boundaries add dispatch and recomputation overhead while retaining
most surrounding activations. Very large boundaries recompute more work and
may interact with skip tensors or mutable state.

| Architecture        | Recommended boundary                                                     |
| ------------------- | ------------------------------------------------------------------------ |
| SpaCeFormer         | One `SpaCeFormerBlockBase` attention/FFN block                           |
| PointTransformerV3  | One `PatchAttentionBlock`                                                |
| MinkUNet            | One `BasicBlock` or `BottleneckBlock`, with BatchNorm state preservation |
| TRELLIS sparse VAE  | One sparse ConvNeXt or channel/spatial residual block                    |
| Dense or sparse DiT | One modulated transformer block                                          |

All three architectures provide these boundaries directly. MinkUNet also
preserves stateful BatchNorm buffers during recomputation, as discussed below.

## SpaCeFormer

All pre-norm, post-norm, and stream-norm blocks inherit a checkpoint boundary
from `SpaCeFormerBlockBase`. Enable every encoder and decoder block at runtime:

```python
from warpconvnet.models import SpaCeFormer

model = SpaCeFormer(
    in_channels=6,
    out_channels=20,
    enc_attn_types="ssccc",
    dec_attn_types="sscc",
).cuda()

model.gradient_checkpointing_enable()
```

Alternatively, enable the same boundaries during construction:

```python
model = SpaCeFormer(
    in_channels=6,
    out_channels=20,
    use_checkpoint=True,
).cuda()
```

SpaCeFormer chooses the window offset or serialized point order outside each
block and passes that value into the checkpoint boundary. The same order is
therefore used during recomputation. Leave `preserve_rng_state` enabled when
the model uses attention dropout or `DropPath`, so the recomputed stochastic
masks match the original forward pass.

See [SpaCeFormer](../models/space_former.md) for architecture and input
examples.

## PointTransformerV3 (PTv3)

Each `PatchAttentionBlock` is a checkpoint boundary. The entire encoder and
decoder can be controlled through the model:

```python
from warpconvnet.models import PointTransformerV3

model = PointTransformerV3(
    in_channels=7,
    out_channels=20,
    shuffle_orders=True,
).cuda()
model.gradient_checkpointing_enable()
```

Constructor-time enablement is also supported:

```python
model = PointTransformerV3(
    in_channels=7,
    out_channels=20,
    use_checkpoint=True,
).cuda()
```

The model selects the serialization order before invoking a block, so the
selected value becomes an explicit checkpoint input and remains stable during
recomputation. As with SpaCeFormer, retain RNG state when dropout or `DropPath`
is nonzero.

See [PointTransformerV3](../models/point_transformer_v3.md) for the full model
configuration.

## MinkUNet

Each MinkUNet `BasicBlock` and `BottleneckBlock` is a checkpoint boundary:

```python
from warpconvnet.models import MinkUNet34

model = MinkUNet34(
    in_channels=3,
    out_channels=20,
    use_checkpoint=True,
).cuda()
```

Runtime and Hugging Face controls work identically:

```python
model.gradient_checkpointing_disable()
model.gradient_checkpointing_enable()
```

MinkUNet residual blocks contain training-mode `BatchNorm1d`. Although replay
uses the same mini-batch statistics, a naïve checkpoint would advance
`running_mean`, `running_var`, and `num_batches_tracked` again. WarpConvNet uses
the non-reentrant checkpoint
[`context_fn`](https://docs.pytorch.org/docs/stable/checkpoint.html) mechanism
to snapshot those buffers when recomputation begins and restore them on every
exit, including exceptions and retained-graph backward calls. The logical
forward therefore performs exactly one persistent BatchNorm update.

This behavior is covered for:

- `BasicBlock` and `BottleneckBlock`, including a downsample branch;
- BatchNorm momentum `0.1`, `1.0`, and `None`;
- BatchNorm with and without running statistics;
- ordinary backward, `autograd.grad`, retained graphs, and gradient
  accumulation;
- DDP with `broadcast_buffers` both enabled and disabled; and
- real sparse-convolution CUDA backward under autocast.

The state-preservation context uses Python module state and is intentionally
unsupported under `torch.compile`; WarpConvNet raises a targeted error instead
of allowing a silently incorrect or partially compiled path. PyTorch requires
compiled checkpoint contexts to be `TorchDispatchMode` instances. Concurrent
forwards through the same checkpointed block instance are also unsupported.
SyncBatchNorm and FSDP deployments should be validated for the application's
process topology before use.

See [MinkUNet](../models/mink_unet.md) for model variants.

## Existing checkpoint-aware modules

WarpConvNet currently defines boundaries in these reusable block families:

- SpaCeFormer `PreNormBlock`, `PostNormBlock`, and `StreamNormBlock`;
- PointTransformerV3 `PatchAttentionBlock`;
- MinkUNet `BasicBlock` and `BottleneckBlock`;
- `SparseConvNeXtBlock3d`;
- `SparseChannelToSpatialResBlock3d` and
  `SparseSpatialToChannelResBlock3d`;
- dense `ModulatedTransformerBlock` and
  `ModulatedTransformerCrossBlock`; and
- sparse `ModulatedSparseTransformerBlock` and
  `ModulatedSparseTransformerCrossBlock`.

`SpaCeFormer`, `PointTransformerV3`, all MinkUNet variants, and the TRELLIS.2
sparse VAE and flow containers expose the aggregate model-level API. The
standalone tree helper can also find the same blocks when they are nested
inside another application model.

## Verification checklist

When adding a new boundary:

1. Confirm `configure_gradient_checkpointing(model)` returns a nonzero count.
2. Compare checkpointed and ordinary outputs and parameter gradients with
   dropout disabled and identical inputs.
3. Instrument the block and confirm one forward call without checkpointing and
   two calls after a checkpointed backward pass.
4. Test every structured input and output type used by the block.
5. Measure both peak memory and step time on a representative sparse input.
6. Check mutable state such as BatchNorm statistics, caches, counters, and
   random order selection.

A simple CUDA memory measurement is:

```python
torch.cuda.reset_peak_memory_stats()

loss = model(batch).feature_tensor.float().square().mean()
loss.backward()

peak_gib = torch.cuda.max_memory_allocated() / 1024**3
print(f"Peak allocated memory: {peak_gib:.2f} GiB")
```

Compare complete training steps, including backward and optimizer state, rather
than measuring only the model's forward pass.

## Limitations

- Checkpointing increases backward compute because enabled regions are replayed.
- It cannot reduce parameters, gradients, optimizer state, or temporary kernel
  workspace.
- An arbitrary model tree is not checkpointed merely because the tree helper
  was called; its leaf boundaries must first use
  `GradientCheckpointingMixin`.
- Stateful or side-effecting forwards require special care because they run
  again during backward.
- Disabling RNG preservation can produce incorrect gradients for stochastic
  attention, dropout, or `DropPath`.
- State-preserving Python recomputation contexts, including MinkUNet's
  BatchNorm handling, are not supported under `torch.compile`.
