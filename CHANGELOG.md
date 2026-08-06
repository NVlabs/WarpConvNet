# Changelog

## Unreleased

- Benchmark cache version bumped to **17.0**; all existing entries are discarded and
  re-benchmarked on first use. There is no v16 -> v17 migration, by design.

  Why: until this release the CUTLASS backward backends did not forward
  `requires_grad`, so they defaulted to computing BOTH gradients. Every CUTLASS
  candidate timed during a single-direction sweep was therefore doing roughly twice
  the necessary work, and was ranked accordingly. Measured inflation of dgrad
  candidate times: ~1.65-1.9x at wide channels (C>=384), ~3.6x at C=128 — largest
  exactly where the dgrad leg is cheapest. Every cached backward ranking produced
  before this release is affected, and a warm cache is never re-tuned, so the fix
  would otherwise never reach existing installs.

  This is **performance recovery, not correctness recovery**. The contamination
  pushed CUTLASS DOWN the rankings, so caches selected a different, valid backend
  (typically `mask_gemm`). No incorrect results are baked into an existing cache —
  the cost is a slower kernel where CUTLASS would now win.

  **Multi-rank users: pre-warm before your next run.** A version bump means a cold
  cache, and a cold cache in a collective-synchronised training step means one rank
  auto-tunes while its peers sit in the next collective. A slow tune there can exceed
  the peers' collective timeout, and it does not self-heal — a sweep killed by the
  timeout never completes, so it never caches, so the next run repeats it. Run
  `scripts/populate_benchmark_cache.py` for your shapes in a single process first.
  Auto-tune now emits a one-shot warning if it runs inside an initialized
  `torch.distributed` context.

- `cute_grouped` weight gradients are now emitted in fp32 rather than the compute
  dtype. This fixes overflow to inf/NaN for AMP weight gradients above the fp16 max
  of 65504 (field values reach ~2.5e6), which previously depended on whether
  auto-tune happened to select `cute_grouped`. **Numerics change:** runs whose
  auto-tune selects this backend are not bit-comparable to earlier runs. On one
  A100 cache `cute_grouped` won ~5% of dgrad and ~7% of wgrad configs, so this is
  not a dormant path — plan ablations accordingly.

- `SpatialFeatureAttention` now masks padded tokens correctly (#26). The flash
  path uses `flash_attn_varlen_qkvpacked_func` on an unpadded packed layout, and
  the non-flash path sets padded-key scores to `-inf` before softmax. Previously
  padded tokens leaked into the attention of active tokens. Outputs of models
  using this module will change.

- Renamed attention plumbing modules for clarity: `ToAttention` ->
  `GeometryToPaddedBatch`, `ToSpatialFeatures` -> `PaddedBatchToGeometry`
  (and `ToAttentionWithoutMask` -> `GeometryToPaddedBatchNoMask`). No aliases;
  update imports.

- SM90 CuTe non-mask GEMM inner-autotune cache entries now use the registry
  identity `(op, backend, tile_id)`. Existing warm cache entries under
  `cute_gemm_sm90_AD_gather_scatter` are intentionally not migrated and will be
  rebenchmarked under `nonmask_gemm_ad_gather_scatter.cute_sm90` after upgrade.
