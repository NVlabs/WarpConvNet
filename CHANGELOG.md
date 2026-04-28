# Changelog

## Unreleased

- SM90 CuTe non-mask GEMM inner-autotune cache entries now use the registry
  identity `(op, backend, tile_id)`. Existing warm cache entries under
  `cute_gemm_sm90_AD_gather_scatter` are intentionally not migrated and will be
  rebenchmarked under `nonmask_gemm_ad_gather_scatter.cute_sm90` after upgrade.
