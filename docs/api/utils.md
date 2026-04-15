# Utils

Internal utilities used by WarpConvNet. These are not part of the public API
and may change without notice.

| Module                              | Purpose                                                 |
| ----------------------------------- | ------------------------------------------------------- |
| `warpconvnet.utils.benchmark_cache` | Persistent auto-tuning cache (msgpack)                  |
| `warpconvnet.utils.autotune_warmup` | Scripts for pre-populating the benchmark cache          |
| `warpconvnet.utils.timer`           | CUDA-aware timing helpers                               |
| `warpconvnet.utils.logger`          | Logging setup                                           |
| `warpconvnet.utils.dist`            | Distributed training helpers                            |
| `warpconvnet.utils.type_cast`       | Mixed-precision dtype utilities                         |
| `warpconvnet.utils.ravel`           | N-dimensional index raveling/unraveling                 |
| `warpconvnet.utils.unique`          | Unique-value operations on tensors                      |
| `warpconvnet.utils.argsort`         | Argsort utilities                                       |
| `warpconvnet.utils.nested`          | Nested tensor helpers                                   |
| `warpconvnet.utils.ntuple`          | Tuple padding/expansion (like `torch.nn.modules.utils`) |
