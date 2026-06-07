# Changelog

## Unreleased

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
