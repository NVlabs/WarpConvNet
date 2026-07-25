# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Mask-based fused implicit GEMM for sparse convolution.

Processes all kernel offsets in a single CUDA launch using bitmask-based
offset skipping and mask_argsort for warp-coherent output ordering.
"""

import os
from typing import Literal, Optional, Tuple

import torch
from torch import Tensor

import warpconvnet._C as _C
from warpconvnet.geometry.coords.search.search_results import IntSearchResult
from warpconvnet.nn.functional.sparse_conv.detail.tile_metadata import (
    known_tile_ids,
    tile_launch_rejection,
)
from warpconvnet.utils.type_cast import _min_dtype


# Sort strategy for mask_argsort. Controls how voxels are ordered prior to
# kernel dispatch:
#   - "mask_bit": stable argsort on the raw uint32 pair_mask word(s). Groups
#                 voxels with identical bitmasks contiguously (default).
#   - "gray_code": treat pair_mask as a Gray code, decode to binary, and sort
#                  by the decoded key. Induces a Gray-order linearization
#                  so consecutive blocks see Hamming-adjacent active-offset
#                  patterns. Expected to improve cache reuse on output rows
#                  when a block transitions between mask groups.
# Override at process start via WARPCONVNET_MASK_SORT={mask_bit,gray_code}.
_MaskSortStrategy = Literal["mask_bit", "gray_code"]


def _default_mask_sort_strategy() -> _MaskSortStrategy:
    val = os.environ.get("WARPCONVNET_MASK_SORT", "mask_bit").strip().lower()
    if val not in ("mask_bit", "gray_code"):
        # Unknown value: fall back to default rather than crash. Mis-typed
        # env vars must not break correctness.
        return "mask_bit"
    return val  # type: ignore[return-value]


def _gray_to_binary_uint32(x: Tensor) -> Tensor:
    """Decode a Gray-code uint32 tensor to its binary representation.

    Standard inverse-Gray: binary[i] = XOR of bits {i, i+1, ..., 31} of gray.
    Implemented as iterated `x ^= x >> shift` doublings (5 steps for 32 bits).
    Operates element-wise; preserves shape and dtype.
    """
    # Cast to int64 to avoid signed-shift surprises while keeping bit-exact
    # uint32 semantics. Final result re-cast to int32 by caller.
    y = x.to(torch.int64) & 0xFFFFFFFF
    y = y ^ (y >> 1)
    y = y ^ (y >> 2)
    y = y ^ (y >> 4)
    y = y ^ (y >> 8)
    y = y ^ (y >> 16)
    return y


_FP16_ABSMAX = 65504.0  # torch.finfo(torch.float16).max
# Rescale target leaves one bit of headroom under fp16 max so boundary values
# cannot round to inf, without scaling small entries toward subnormals more
# than necessary.
_FP16_RESCALE_TARGET = 32768.0


def _fp16_safe_cast(t: Tensor) -> Tuple[Tensor, Tensor]:
    """Cast an fp32 tensor to fp16, exactly rescaling when out of fp16 range.

    An unconditional ``.half()`` turns any fp32 value beyond 65504 into inf,
    which the kernel accumulates into NaN. Under a GradScaler (loss scale up
    to 2**16) fp32 grad_output beyond fp16 range is the NORMAL case, not an
    anomaly — observed in the wild at absmax 3.3e5: finite fp32 in, NaN out,
    while explicit_gemm was exact; under a scaler the NaN never surfaces, it
    just silently halves the loss scale forever.

    Returns ``(t_fp16, scale)`` with ``t == t_fp16 * scale`` exact in the
    exponent: ``scale`` is a power of two, so the divide and the caller's
    multiply-back shift exponents without touching mantissas — full fp16
    precision at unlimited range (strictly better than a bf16 reroute, which
    permanently costs 2 mantissa bits). The mask GEMM ops are (bi)linear in
    each operand, so callers multiply their OUTPUT by the product of the
    input scales.

    ``scale`` is a 0-dim fp32 CUDA tensor and every step here is device-side:
    NO host sync — a ``.item()``/``bool()`` in this path would serialize the
    stream on every sparse conv and break CUDA-graph capture. The clamp makes
    in-range tensors get exponent 0 (scale 1); an all-zero tensor hits
    ``log2(0) = -inf`` and also clamps to 0; a non-finite absmax yields a
    non-finite scale and the NaN/inf propagates, same as an unguarded cast.
    """
    if t.numel() == 0:
        return t.half(), torch.ones((), dtype=torch.float32, device=t.device)
    m = t.abs().max().float()
    exp = torch.clamp(torch.ceil(torch.log2(m / _FP16_RESCALE_TARGET)), min=0.0)
    scale = torch.exp2(exp)
    t16 = (t * torch.exp2(-exp)).half()
    return t16, scale


def _fp16_safe_cast_memo(t: Tensor, scratch: Optional[dict]) -> Tuple[Tensor, Tensor]:
    """`_fp16_safe_cast` memoized in a per-autograd-backward scratch dict.

    dgrad and wgrad are dispatched as separate calls within one backward and
    cast the same grad_output/in_features/weight; sharing the cast + absmax
    halves that work. Keyed by ``id(t)`` — sound only because the scratch
    lives for a single backward invocation, during which the saved tensors
    cannot be mutated. Do NOT replace with a tensor-attribute + ``_version``
    cache: ``.data``-level writes (EMA updates, DeepSpeed flat-buffer swaps)
    change values without bumping ``_version``, and a stale absmax silently
    reintroduces the very inf this cast exists to prevent.
    """
    if scratch is None:
        return _fp16_safe_cast(t)
    hit = scratch.get(id(t))
    if hit is None:
        hit = _fp16_safe_cast(t)
        scratch[id(t)] = hit
    return hit


def _build_pair_table(
    kernel_map: IntSearchResult,
    N_out: int,
    device: torch.device,
) -> Tensor:
    """Build the forward pair_table [K * N_out] from kernel_map."""
    K = len(kernel_map)
    if hasattr(kernel_map, "_pair_table") and kernel_map._pair_table is not None:
        return kernel_map._pair_table.reshape(-1).contiguous()

    pair_table = torch.empty(K * N_out, dtype=torch.int32, device=device)
    pair_table.fill_(-1)
    L = kernel_map.in_maps.shape[0]
    if L > 0 and hasattr(_C.gemm, "csr_to_pair_table_cuda"):
        offsets_gpu = kernel_map.offsets.to(device=device, dtype=torch.int32)
        _C.gemm.csr_to_pair_table_cuda(
            kernel_map.in_maps.int(),
            kernel_map.out_maps.int(),
            offsets_gpu,
            pair_table,
            N_out,
            K,
        )
    return pair_table


# DISPATCH_MW template boundaries that mask_gemm_bindings.cu instantiates. This
# mirrors the binding's DISPATCH_MW macro (a warpconvnet-side .cu decision, the
# SoT for which MW templates this build actually has) and matches warpgemm's
# MASK_WORDS_TIERS / round_mask_words_to_tier(K) contract. tests/csrc/
# test_tile_metadata_drift.py asserts it stays the union of the snapshot's
# dispatch_mask_words so a warpgemm tier extension cannot drift silently.
_MASK_WORDS_TIERS = (1, 2, 4, 8, 12)


def _dispatched_mask_words(K: int) -> int:
    """Round mask_words up to the next DISPATCH_MW template boundary.

    mask_gemm_bindings.cu's DISPATCH_MW macro picks templates at MW=1, 2, 4, 8,
    12 (``_MASK_WORDS_TIERS``). Kernel templates use their compile-time MW as the
    stride when indexing ``pair_mask[row * MW + word]``; allocating pair_mask with
    a smaller stride than the dispatched MW reads past the allocation → illegal
    address / silent-wrong output.

    Raises for K > 384 (mask_words > 12): the binding has no MW>12 kernel, so
    such configs must route to a non-mask backend. Mirrors warpgemm
    round_mask_words_to_tier, which raises past its top tier; autotune skips the
    mask candidate and execution falls back to explicit_gemm.
    """
    mw_rt = (K + 31) // 32
    for mw in _MASK_WORDS_TIERS:
        if mw_rt <= mw:
            return mw
    raise ValueError(
        f"K={K} needs mask_words={mw_rt} > {_MASK_WORDS_TIERS[-1]}; mask_gemm has "
        "no MW>12 kernel (max kernel_volume 384). Route K>384 to explicit_gemm."
    )


def _build_mask_and_argsort(
    pair_table: Tensor,
    N: int,
    K: int,
    device: torch.device,
    sort_strategy: Optional[_MaskSortStrategy] = None,
) -> Tuple[Tensor, Tensor]:
    """Build pair_mask and mask_argsort from a pair_table [K * N].

    For K <= 32: pair_mask is [N] int32 (single uint32 bitmask per voxel).
    For K > 32: pair_mask is [N * mask_words_padded] int32, interleaved as
                pair_mask[voxel_i * mask_words_padded + word_w].
                mask_words_padded is the next DISPATCH_MW template boundary
                so the kernel's stride matches what the caller allocates.
    mask_argsort is always [N] int32 (voxel permutation).

    sort_strategy: see _default_mask_sort_strategy() docstring. None reads
    the WARPCONVNET_MASK_SORT env var (default "mask_bit"). The choice is
    semantic-preserving — both yield valid permutations of [0, N).
    """
    mask_words = _dispatched_mask_words(K)
    pair_mask = torch.zeros(N * mask_words, dtype=torch.int32, device=device)
    _C.gemm.build_pair_mask_cuda(pair_table, pair_mask, K, mask_words)

    strategy: _MaskSortStrategy = (
        sort_strategy if sort_strategy is not None else _default_mask_sort_strategy()
    )

    if mask_words == 1:
        word0 = pair_mask
    else:
        # Stride view of word 0 of each voxel. .contiguous() forces a copy
        # so the sort/decoded key isn't strided in subsequent ops.
        word0 = pair_mask[::mask_words].contiguous()

    if strategy == "gray_code":
        # Decode pair_mask (Gray) -> binary, then stable-sort. This puts
        # voxels with Hamming-adjacent mask bits at adjacent positions,
        # improving output-row cache reuse across consecutive blocks.
        # For mask_words > 1, the decoded word-0 key is the dominant
        # signal; ties on word 0 fall back to the natural (stable) order
        # which still groups identical patterns. We deliberately don't
        # multi-key sort the higher words — empirically the fwd kernel
        # only consults word 0 first (NB: a future opt could chain).
        key = _gray_to_binary_uint32(word0).int()
    else:  # "mask_bit" (default, legacy)
        key = word0

    # Fast path: cub::DeviceRadixSort via direct binding bypasses torch.argsort
    # Python+dispatcher overhead. Saves ~150us per call at N=2928 vs
    # torch.argsort(stable=True). Non-stable: voxels with identical mask
    # may be reordered within their group — usually semantic-preserving for
    # cache coherence, but under investigation as a slow-drift training
    # convergence delta vs previous stable-sort behavior.
    #
    # Set WARPCONVNET_FORCE_STABLE_ARGSORT=1 to force torch.argsort(stable=True)
    # — diagnostic A/B for train-trajectory-divergence checks. Adds
    # ~150us per call; only set when investigating numerics.
    _force_stable = os.environ.get("WARPCONVNET_FORCE_STABLE_ARGSORT", "0").strip() in (
        "1",
        "true",
        "True",
    )
    if _force_stable or not hasattr(_C.gemm, "mask_argsort_cuda"):
        mask_argsort = torch.argsort(key, stable=True).int()
    else:
        mask_argsort = torch.empty(N, dtype=torch.int32, device=device)
        _C.gemm.mask_argsort_cuda(key.contiguous(), mask_argsort)
    return pair_mask, mask_argsort


def _kernel_map_to_mask_data(
    kernel_map: IntSearchResult,
    num_out_coords: int,
    device: torch.device,
    sort_strategy: Optional[_MaskSortStrategy] = None,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Convert IntSearchResult to mask-based pair_table + mask + argsort.

    Returns:
        pair_table: [K * N_out] int32, flattened
        pair_mask: [N_out * mask_words] int32 (uint32 bitmask, interleaved)
        mask_argsort: [N_out] int32 permutation
    """
    K = len(kernel_map)
    N_out = num_out_coords
    pair_table = _build_pair_table(kernel_map, N_out, device)
    pair_mask, mask_argsort = _build_mask_and_argsort(
        pair_table, N_out, K, device, sort_strategy=sort_strategy
    )
    return pair_table, pair_mask, mask_argsort


def _build_reverse_mask_data(
    pair_table: Tensor,
    N_in: int,
    N_out: int,
    K: int,
    device: torch.device,
    sort_strategy: Optional[_MaskSortStrategy] = None,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Build reverse pair_table + mask + argsort for atomicAdd-free dgrad.

    The forward pair_table maps (offset_k, out_row) -> in_row.
    The reverse maps (offset_k, in_row) -> out_row, enabling the dgrad
    kernel to iterate over input rows and gather from grad_output.

    Fast path uses a fused CUDA kernel that emits reverse_pair_table AND
    reverse_pair_mask in a single launch (atomicOr on the bitmask).
    Eliminates ~0.7-1.0ms of host-driven torch.where + scatter + separate
    pair_mask launch at small-N high-K shapes.

    Returns:
        reverse_pair_table: [K * N_in] int32
        reverse_pair_mask: [N_in * mask_words] int32 (uint32 bitmask, interleaved)
        reverse_mask_argsort: [N_in] int32 permutation
    """
    mask_words = _dispatched_mask_words(K)

    if hasattr(_C.gemm, "build_reverse_mask_data_cuda"):
        reverse_pair_table = torch.full((K * N_in,), -1, dtype=torch.int32, device=device)
        reverse_pair_mask = torch.zeros(N_in * mask_words, dtype=torch.int32, device=device)
        _C.gemm.build_reverse_mask_data_cuda(
            pair_table.contiguous(),
            reverse_pair_table,
            reverse_pair_mask,
            N_in,
            N_out,
            K,
            mask_words,
        )
        reverse_flat = reverse_pair_table

        strategy: _MaskSortStrategy = (
            sort_strategy if sort_strategy is not None else _default_mask_sort_strategy()
        )
        if mask_words == 1:
            word0 = reverse_pair_mask
        else:
            word0 = reverse_pair_mask[::mask_words].contiguous()
        if strategy == "gray_code":
            key = _gray_to_binary_uint32(word0).int()
        else:
            key = word0
        if hasattr(_C.gemm, "mask_argsort_cuda"):
            reverse_mask_argsort = torch.empty(N_in, dtype=torch.int32, device=device)
            _C.gemm.mask_argsort_cuda(key.contiguous(), reverse_mask_argsort)
        else:
            reverse_mask_argsort = torch.argsort(key, stable=True).int()
        return reverse_flat, reverse_pair_mask, reverse_mask_argsort

    # Legacy fallback: torch.where + scatter, then separate pair_mask launch.
    pair_table_2d = pair_table.reshape(K, N_out)
    reverse_pair_table = torch.full((K, N_in), -1, dtype=torch.int32, device=device)

    valid = pair_table_2d >= 0
    k_idx, out_idx = torch.where(valid)
    in_idx = pair_table_2d[k_idx, out_idx].long()
    reverse_pair_table[k_idx, in_idx] = out_idx.int()

    reverse_flat = reverse_pair_table.reshape(-1).contiguous()
    reverse_pair_mask, reverse_mask_argsort = _build_mask_and_argsort(
        reverse_flat, N_in, K, device, sort_strategy=sort_strategy
    )
    return reverse_flat, reverse_pair_mask, reverse_mask_argsort


def _get_mask_data(
    kernel_map: IntSearchResult,
    num_out_coords: int,
    device: torch.device,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Get or compute mask data, cached on the kernel_map object."""
    if kernel_map._mask_data is None:
        kernel_map._mask_data = _kernel_map_to_mask_data(kernel_map, num_out_coords, device)
    return kernel_map._mask_data


def _get_reverse_mask_data(
    kernel_map: IntSearchResult,
    num_in_coords: int,
    num_out_coords: int,
    device: torch.device,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Get or compute reverse mask data, cached on the kernel_map object."""
    if kernel_map._reverse_mask_data is None:
        K = len(kernel_map)
        fwd_pair_table, _, _ = _get_mask_data(kernel_map, num_out_coords, device)
        kernel_map._reverse_mask_data = _build_reverse_mask_data(
            fwd_pair_table, num_in_coords, num_out_coords, K, device
        )
    return kernel_map._reverse_mask_data


# ---------------------------------------------------------------------------
# Tile-id catalogs used by tile selection below.
#
# SOURCE-OF-TRUTH BOUNDARY — read before editing:
#
# Whether a tile can be dispatched at MaskWords>1 (K>32) is a fact OWNED BY
# WARPGEMM's registry, not by this file and not regex-derivable from tile names.
# It is NOT the per-tile ``mask_words`` in ``tile_metadata.py`` (that is each
# tile's single *canonical compiled* MW, ≈1 for nearly all tiles); the binding's
# ``DISPATCH_MW`` macro re-instantiates the same tile at MW=1,2,4,8,12, so e.g.
# tile 41 reports ``mask_words=1`` yet is the aligned K>32 default and is
# correctly absent below. Two distinct reasons a tile is MW1-only:
#   1. STRUCTURAL (pcoff): the precomputed-offset prescan buffer
#      ``precomputed_in_rows[MaskWords*32*tM]`` scales with MW and blows the
#      ~100KB smem budget on SM86/89 — a genuine per-arch capability limit.
#      -> {54,55,56,57,58,59,63} fwd and {905..911} dgrad-wt.
#   2. INSTANTIATION GAP (32x32): the 32x32 template is MW-generic and could
#      compile at MW>1, but warpgemm ships no 32x32 MW>1 tile_id. Treated
#      MW1-only here (safe: K>32 falls back to a 64x64 tile). -> {28,32,33} fwd
#      and {903} dgrad-wt.
#
# AUTHORIZATION vs AVAILABILITY — the guard is an AVAILABILITY fact:
# "this build's DISPATCH_MW macro has no MW>1 kernel for this tile, so do not
# dispatch it there." warpgemm's forthcoming per-tile ``dispatch_mask_words`` is
# an AUTHORIZATION ceiling ("validated-correct to instantiate at these MW"), NOT
# a guarantee a binary exists in the .so. So binding_instantiated(MW) ⊆
# dispatch_mask_words, and the guard must be a SUPERSET of the MW1-only-authorized
# tiles. Dropping a tile from the guard (e.g. the 32x32 set once its sweep
# validates (1,2,4)) must be SIMULTANEOUS with this binding fanning that
# kernel_struct out to MW2/4 — never on field presence alone. Over-guarding is
# safe (K>32 falls back to a 64x64 tile); under-guarding crashes.
# tests/csrc/test_tile_metadata_drift.py asserts that subset invariant once the
# field ships, and guards these literals structurally until then.
#
# Routing K>32 to an MW=1-only tile makes the kernel device-assert
# ``K <= MaskWords*32`` — a context-killing abort for the whole process — so the
# guard must happen here, at the Python level.
#
# The wcn-only fallback tiles 70/71/72 (scalar), 80/81/82 (f32-out) and 300-307
# (strided) are warpconvnet-binding-only and do NOT appear in warpgemm metadata.
# ---------------------------------------------------------------------------

# Pcoff forward tile IDs — structural MW1-only (prescan smem ceiling).
_PCOFF_FWD_TILES = frozenset({54, 55, 56, 57, 58, 59, 63})

# 32x32 forward tiles — split by binding AVAILABILITY (per-(struct,config), not
# per-shape). The warpgemm sweep validated all three at MW2/4, so the field
# authorizes 28/32/33 = (1,2,4) — but availability follows what this binding
# instantiates:
#   - tile 28 (1s_flat + Tile32x32x32_F16Accum config) IS instantiated at MW2/4
#     (mask_gemm_kernels_fwd.cu launch_mask_gemm_fwd_32x32_f16acc_mw), so it is
#     MW<=4-capable (K<=128); the binding has no 32x32 MW8/12, so K>128 rejects.
#   - tiles 32/33 (1s_flat_direpi structs) are NOT instantiated at MW>1, so they
#     stay MW1-only. (They are in no candidate pool, so this is moot in practice,
#     but the guard must track availability per struct, not tile shape.)
_32X32_MW4_FWD_TILES = frozenset({28})  # instantiated MW1/2/4 (K<=128)
_32X32_MW1_FWD_TILES = frozenset({32, 33})  # _direpi structs, MW1 only
_32X32_FWD_TILES = _32X32_MW4_FWD_TILES | _32X32_MW1_FWD_TILES  # all 32x32 (tests)

# Forward tiles that must not be dispatched at MW>1 (pcoff + 32x32 _direpi).
_MW1_ONLY_FWD_TILES = _32X32_MW1_FWD_TILES | _PCOFF_FWD_TILES

# Forward tiles capped at MW4: 32x32 tile 28 (no MW8/12 launcher → reject K>128).
_MW4_MAX_FWD_TILES = _32X32_MW4_FWD_TILES

# wcn-only strided forward tiles (not in warpgemm metadata).
_STRIDED_FWD_TILES = frozenset(range(300, 308))

# dgrad_wt tile_ids (canonical 900-911) that route through fwd kernels but are
# MW1-only: 903 (32x32 instantiation gap) + 905-911 (pcoff structural limit).
# 900/901/902/904 are flat/fused and MW-capable, so correctly excluded.
_MW1_ONLY_DGRAD_WT_TILES = frozenset({903, 905, 906, 907, 908, 909, 910, 911})

# All canonical dgrad_wt tile ids the binding's dgrad arm accepts. The
# fwd_as_dgrad selector whitelists against this set: params["tile_id"] is
# untrusted input (env overrides, stale benchmark caches, mis-assembled pools
# can carry foreign ids such as the fwd pcoff family 54-59/63), and a foreign
# MW1-only fwd tile dispatched at K>32 device-asserts `K <= MaskWords*32`,
# poisoning the CUDA context mid-autotune.
_DGRAD_WT_TILES = frozenset(range(900, 912))

# Native dgrad pcoff tile_ids. MW=1 only.
_DGRAD_PCOFF_TILES = frozenset({64, 65, 66, 67, 68, 69})

# wcn-only scalar/f32-out fallback tiles (not in warpgemm metadata). Referenced
# by the selectors and asserted absent-from-metadata by the drift guard.
# NOTE: the strided 300-307 family DOES appear in schema-7 metadata (op
# "forward", tier experimental); only the raw-integer fallbacks stay
# metadata-absent.
_WCN_ONLY_FWD_TILES = frozenset({70, 71, 72, 80, 81, 82}) | _STRIDED_FWD_TILES

# Launch-time authorization allowlists: raw-integer tile ids the binding
# dispatches that have no warpgemm metadata record. Everything else must pass
# tile_launch_rejection() — exact device arch in compile_archs + launchable
# backend — before entering the C++ switch, INCLUDING pinned/cached tile ids
# (Blackwell handoff §3). dgrad_wt 900-911 carry op="forward" in metadata
# (they are fwd kernel aliases), native dgrad ids carry op="dgrad".
_METADATA_ABSENT_FWD_LAUNCH_IDS = frozenset({70, 71, 72, 80, 82})
_METADATA_ABSENT_DGRAD_LAUNCH_IDS = frozenset({70, 71, 72, 81})
_METADATA_ABSENT_WGRAD_LAUNCH_IDS = frozenset({73})


def _require_launchable(op: str, tile_id: int, metadata_absent_ok: frozenset) -> None:
    """Raise (autotune-skippable) if a resolved tile id may not launch here.

    ``op`` is the metadata op space ("forward"/"dgrad"/"wgrad"), which is not
    always the launch direction: dgrad_wt aliases validate in "forward".
    """
    if tile_id in metadata_absent_ok:
        return
    reason = tile_launch_rejection(op, tile_id)
    if reason is not None:
        raise RuntimeError(f"mask_gemm tile authorization failed: {reason}. Candidate skipped.")


def _select_fwd_tile(
    tile_id: int,
    mask_words: int,
    use_f32_output: bool,
    cin_aligned: bool,
    cout_aligned: bool,
) -> Tuple[int, bool]:
    """Resolve the forward tile_id + whether an f32-output tile is used.

    Pure decision extracted from ``_mask_gemm_forward_logic`` so the routing
    rules are testable in isolation and a future metadata-backed migration can
    swap this body without touching the launch code. Behavior is identical to
    the previous inline ladder.

    Returns ``(resolved_tile_id, use_f32_out_tile)``. Raises if the resolved
    tile cannot service ``mask_words`` (MW=1-only tile with K>32).
    """
    use_strided_tile = tile_id in _STRIDED_FWD_TILES
    use_f32_out_tile = use_f32_output and mask_words == 1 and not use_strided_tile
    if use_f32_out_tile:
        # f32-output tiles: aligned -> 80, otherwise scalar-in -> 82.
        tile_id = 80 if (cin_aligned and cout_aligned) else 82
    elif not use_strided_tile and (not cin_aligned or not cout_aligned):
        # Scalar path. Only tile 70 supports mask_words > 1; force it.
        if mask_words > 1:
            tile_id = 70
        elif not cin_aligned and not cout_aligned:
            tile_id = 70
        elif not cin_aligned:
            tile_id = 71
        else:
            tile_id = 72

    # MW=1-only tiles (28, pcoff 54-63) would silently fall back to MW=1 in the
    # binding and device-assert `K <= MaskWords*32`. Raise so autotune catches
    # it as a Python RuntimeError and skips the candidate.
    if mask_words > 1 and tile_id in _MW1_ONLY_FWD_TILES:
        raise RuntimeError(
            f"Tile {tile_id} only supports mask_words==1 (got {mask_words}). "
            "Use a tile with MW>1 instantiation for K>32."
        )
    if mask_words > 4 and tile_id in _MW4_MAX_FWD_TILES:
        raise RuntimeError(
            f"Tile {tile_id} (32x32) only supports mask_words<=4 (K<=128), got "
            f"{mask_words}. No 32x32 MW8/12 kernel; use a 64x64+ tile for K>128."
        )
    return tile_id, use_f32_out_tile


def _select_dgrad_tile(
    use_fwd_for_dgrad: bool,
    params: dict,
    mask_words: int,
    use_f32_out_tile: bool,
    vec_width: int,
    C_in_g: int,
    C_out_g: int,
    use_fp16_accum: bool,
) -> Tuple[int, bool]:
    """Resolve ``(dgrad_tile_id, use_fwd_fallback)`` for the dgrad launch.

    ``use_fwd_fallback=True`` routes through ``backend.fwd`` (wcn-only fwd tile
    ids, caller has pre-transposed the weight); ``False`` routes through
    ``backend.dgrad``. Pure decision extracted from ``_mask_gemm_backward_logic``
    so the two routing ladders (fwd-as-dgrad vs native) are testable in isolation
    and a future ``dispatch_mask_words``-backed migration is surgical. Behavior
    is identical to the previous inline ladders.
    """
    if use_fwd_for_dgrad:
        # Autotune pool uses canonical dgrad_wt tile_ids 900-911; the binding's
        # dgrad arm routes those to LAUNCH_FWD after the caller pre-transposes
        # weight. Alignment fallbacks instead use wcn-only fwd ids (70-72, 80, 82)
        # routed through backend.fwd directly.
        dgrad_tile = params.get("tile_id", 900)
        fwd_cin_aligned = C_out_g % vec_width == 0
        fwd_cout_aligned = C_in_g % vec_width == 0
        use_fwd_fallback = False
        if use_f32_out_tile:
            dgrad_tile = 80 if (fwd_cin_aligned and fwd_cout_aligned) else 82
            use_fwd_fallback = True
        elif not fwd_cin_aligned or not fwd_cout_aligned:
            if mask_words > 1:
                dgrad_tile = 70
            elif not fwd_cin_aligned and not fwd_cout_aligned:
                dgrad_tile = 70
            elif not fwd_cin_aligned:
                dgrad_tile = 71
            else:
                dgrad_tile = 72
            use_fwd_fallback = True
        elif dgrad_tile not in _DGRAD_WT_TILES:
            # Foreign tile id (e.g. fwd pcoff 54-59/63 from a stale benchmark
            # cache or a mis-assembled pool). Reject at selection so autotune
            # skips the candidate; a foreign MW1-only fwd tile reaching the
            # kernel at K>32 device-asserts and poisons the CUDA context.
            raise RuntimeError(
                f"mask_gemm_fwd_as_dgrad requires a dgrad_wt tile_id in "
                f"900-911, got {dgrad_tile}. Foreign/stale tile id — "
                f"candidate skipped."
            )
        elif mask_words > 1:
            # AVAILABILITY guard, wider than the structural
            # _MW1_ONLY_DGRAD_WT_TILES set: the binding's dgrad_wt arm routes
            # every alias through the MW=1 launch_mask_gemm_fwd launcher and
            # has no DISPATCH_MW arm, so NO wt tile can service K>32 in this
            # build — including 900/901/902/904, which warpgemm metadata
            # authorizes at MW>1 but this binding does not instantiate
            # (verified 2026-07-22: all four device-assert `K <= MaskWords*32`
            # at kv=64, poisoning the CUDA context mid-autotune). Skip the
            # candidate so autotune moves on; routing to a different tile
            # would benchmark the wrong kernel under this algo name.
            raise RuntimeError(
                f"mask_gemm_fwd_as_dgrad tile {dgrad_tile} cannot service "
                f"mask_words={mask_words} (K>32): the dgrad_wt binding arm "
                f"has no MaskWords>1 instantiation. Candidate skipped."
            )
        return dgrad_tile, use_fwd_fallback

    # Native dgrad path (backend.dgrad).
    cin_aligned = C_in_g % vec_width == 0
    cout_aligned = C_out_g % vec_width == 0
    if not cin_aligned or not cout_aligned:
        # wcn-only scalar dgrad tiles 70/71/72.
        if mask_words > 1:
            dgrad_tile = 70
        elif not cin_aligned and not cout_aligned:
            dgrad_tile = 70
        elif not cout_aligned:
            dgrad_tile = 71
        else:
            dgrad_tile = 72
    elif mask_words > 1:
        # Native dgrad tiles only have MW=1 instantiations in warpconvnet
        # bindings. Route MW>1 to scalar tile 70 (SAB_SE MW), which supports up
        # to MW=12. For better perf at MW>1 aligned shapes, prefer the
        # mask_gemm_fwd_as_dgrad algo in the pool.
        dgrad_tile = 70
    elif use_f32_out_tile:
        dgrad_tile = 81  # wcn-only dgrad f32-out
    elif params.get("tile_id") in _DGRAD_PCOFF_TILES:
        # Native dgrad pcoff: autotune-driven tile_id 64-69 wins for
        # E1 hoist on dgrad. MW=1 only, fp16 only.
        dgrad_tile = params["tile_id"]
    else:
        # Canonical dgrad tile selection by per-group channel size.
        # Migration map (previous wcn ids -> canonical):
        #   50->12 (32x32)         53->22 (64x64 F16Accum)
        #   51->0  (64x64 2s)      54->24 (64x128 F16Accum)
        #   52->1  (64x128 2s)
        #
        # F16Accum tiles (22, 24) accumulate the K*C reduction in fp16 — at
        # C>=64 this 2-3x worse rel_diff vs explicit_gemm reference degrades
        # MinkUNet ScanNet AMP training convergence (per-algo grad sweep
        # 2026-05-21). Gate the F16Acc choice on use_fp16_accum, NOT
        # compute_dtype (fp16 inputs are normal under AMP and don't imply user
        # wants fp16 accumulator).
        C = max(C_in_g, C_out_g)
        if C <= 48:
            dgrad_tile = 12  # ex-50: 32x32
        elif C <= 96:
            dgrad_tile = 22 if use_fp16_accum else 0  # F16Acc / f32
        else:
            dgrad_tile = 24 if use_fp16_accum else 1  # F16Acc / f32
    return dgrad_tile, False


def _mask_gemm_forward_logic(
    in_features: Tensor,
    weight: Tensor,
    kernel_map: IntSearchResult,
    num_out_coords: int,
    params: dict,
    groups: int = 1,
) -> Tensor:
    """Fused mask-GEMM forward. Selects a tile by channel alignment + output
    dtype + mask_words, then issues a single ``_C.mask_gemm.fwd`` launch
    (groups>1 handled via grid.z)."""
    K = len(kernel_map)
    mask_words = _dispatched_mask_words(K)

    # Submanifold identity shortcut: center offset maps each voxel to itself
    # Enable when stride=1 (N_in == N_out) and K is odd
    N_in_fwd = in_features.shape[0]
    is_submanifold = (N_in_fwd == num_out_coords) and (K % 2 == 1)
    identity_offset = K // 2 if is_submanifold else -1

    tile_id = params.get("tile_id", 41)
    pair_table, pair_mask, mask_argsort = _get_mask_data(
        kernel_map, num_out_coords, in_features.device
    )

    # Per-group channel dimensions
    if groups > 1:
        # weight: [K, G, C_in_g, C_out_g]
        C_in_g = weight.shape[2]
        C_out_g = weight.shape[3]
    else:
        # weight: [K, C_in, C_out]
        C_in_g = weight.shape[1]
        C_out_g = weight.shape[2]

    # Cast to fp16 if needed (mask_gemm kernels require fp16/bf16), with exact
    # power-of-two rescaling for values beyond fp16 range — see _fp16_safe_cast.
    # The op is bilinear in (in_features, weight): output scales multiply.
    orig_dtype = in_features.dtype
    use_f32_output = orig_dtype == torch.float32
    fwd_scale = None  # 0-dim device tensor on the fp32 path
    if orig_dtype == torch.float32:
        _in, s_in = _fp16_safe_cast(in_features)
        _w, s_w = _fp16_safe_cast(weight)
        fwd_scale = s_in * s_w
    else:
        _in = in_features
        _w = weight

    # Select tile based on per-group channel alignment and output dtype
    # (mask_words > 1 / K > 32 routing is handled inside _select_fwd_tile).
    vec_width = 16 // _in.element_size()
    cin_aligned = C_in_g % vec_width == 0
    cout_aligned = C_out_g % vec_width == 0
    tile_id, use_f32_out_tile = _select_fwd_tile(
        tile_id, mask_words, use_f32_output, cin_aligned, cout_aligned
    )
    _require_launchable("forward", tile_id, _METADATA_ABSENT_FWD_LAUNCH_IDS)
    out_dtype = torch.float32 if use_f32_out_tile else _in.dtype

    # Single launch handles groups=1 and groups>1 via grid.z
    C_out_total = C_out_g * groups
    output = torch.zeros((num_out_coords, C_out_total), dtype=out_dtype, device=in_features.device)
    status = _C.mask_gemm.fwd(
        _in,
        _w,
        output,
        pair_table,
        pair_mask,
        mask_argsort,
        K,
        tile_id,
        mask_words,
        identity_offset,
        1.0,
        groups,
    )
    if status != 0:
        raise RuntimeError(f"mask_gemm fwd failed: status={status}, tile={tile_id}")

    output = output.to(dtype=orig_dtype)
    if fwd_scale is not None:
        # Exact pow2 multiply-back (0-dim device tensor, no sync).
        output = output * fwd_scale
    return output


def _mask_gemm_backward_logic(
    algo: str,
    grad_output: Tensor,
    in_features: Tensor,
    weight: Tensor,
    kernel_map: IntSearchResult,
    num_out_coords: int,
    device: torch.device,
    needs_input_grad: Tuple[bool, ...],
    params: dict,
    weight_T: Optional[Tensor] = None,
    groups: int = 1,
    use_fp16_accum: bool = False,
    scratch: Optional[dict] = None,
) -> Tuple[Optional[Tensor], Optional[Tensor]]:
    """Fused mask-GEMM backward (dgrad and/or wgrad).

    ``algo`` selects the dgrad path: "mask_gemm" uses the native dgrad kernel
    (W read with an in-shared-memory stride transpose); "mask_gemm_fwd_as_dgrad"
    reuses the fwd kernel after the caller pre-transposes the weight.
    """
    use_fwd_for_dgrad = algo == "mask_gemm_fwd_as_dgrad"

    K = weight.shape[0]
    mask_words = _dispatched_mask_words(K)

    # Default wgrad tile_id is canonical 0 (= ex-wcn 60, 64x64 f32 direct).
    tile_id = params.get("tile_id", 0)
    split_k = params.get("split_k", 64)
    N_in = in_features.shape[0]

    # Submanifold identity shortcut for dgrad
    # For submanifold conv, reverse_pair_table[K//2, j] == j (identity)
    is_submanifold_bwd = (N_in == num_out_coords) and (K % 2 == 1)
    identity_offset_bwd = K // 2 if is_submanifold_bwd else -1

    # Per-group channel dimensions
    if groups > 1:
        C_in_g = weight.shape[2]
        C_out_g = weight.shape[3]
        C_in = groups * C_in_g
        C_out = groups * C_out_g
    else:
        C_in = in_features.shape[1]
        C_out = weight.shape[2]
        C_in_g = C_in
        C_out_g = C_out

    # Cast to fp16 if needed (mask_gemm kernels require fp16/bf16), with exact
    # power-of-two rescaling for values beyond fp16 range — see _fp16_safe_cast.
    # dgrad is bilinear in (grad_output, weight) and wgrad in (in_features,
    # grad_output), so each leg's output is multiplied back by the product of
    # its operands' scales.
    orig_dtype = grad_output.dtype
    use_f32_output = orig_dtype == torch.float32
    s_go = s_in = s_w = None  # 0-dim device tensors on the fp32 path
    if orig_dtype == torch.float32:
        _go, s_go = _fp16_safe_cast_memo(grad_output, scratch)
        _in, s_in = _fp16_safe_cast_memo(in_features, scratch)
        _w, s_w = _fp16_safe_cast_memo(weight, scratch)
    else:
        _go = grad_output
        _in = in_features
        _w = weight

    compute_dtype = _go.dtype

    grad_in = None
    grad_weight = None

    if needs_input_grad[0]:
        rev_pt, rev_pm, rev_as = _get_reverse_mask_data(
            kernel_map, N_in, num_out_coords, grad_output.device
        )

        # Native mask_gemm.dgrad reads W[K, G, Cin, Cout] with a stride-transpose
        # in shared memory so MMA reduces over Cout (correct for dX = dY @ W^T).
        # mask_gemm_fwd_as_dgrad reuses the fwd kernel, whose B-loader reduces
        # over the first channel axis — so W must be pre-transposed to swap the
        # axes before the kernel sees it. .contiguous() is mandatory: fwd's
        # vectorized cp.async needs 16-byte-aligned strides and a .transpose()
        # view does not satisfy that.
        #
        # Reuse caller-provided weight_T when groups==1 (its [K, C_out, C_in]
        # layout matches what fwd_as_dgrad needs after .transpose(-1,-2)). Saves
        # one 54MB copy per bwd call at C=1024 K=27 fp16.
        # The cached transpose holds UNSCALED values, so when it is used the
        # dgrad multiply-back must apply s_go only — track which weight tensor
        # feeds the kernel instead of comparing s_w to 1 (a host-side scale
        # check would sync the stream).
        dgrad_w_is_scaled = True
        if use_fwd_for_dgrad:
            if weight_T is not None and groups == 1 and weight_T.dtype == _w.dtype:
                _w_dgrad = weight_T
                dgrad_w_is_scaled = False
            else:
                _w_dgrad = _w.transpose(-1, -2).contiguous()
        else:
            _w_dgrad = _w.contiguous()

        # Select dgrad tile + routing (mask_words>1 guard handled inside).
        vec_width = 16 // _go.element_size()
        use_f32_out_tile = use_f32_output and mask_words == 1
        dgrad_out_dtype = torch.float32 if use_f32_out_tile else compute_dtype

        dgrad_tile, use_fwd_fallback = _select_dgrad_tile(
            use_fwd_for_dgrad,
            params,
            mask_words,
            use_f32_out_tile,
            vec_width,
            C_in_g,
            C_out_g,
            use_fp16_accum,
        )
        # Fwd-fallback (wcn-only fwd ids, weight pre-transposed) -> backend.fwd;
        # otherwise backend.dgrad (native, or the dgrad_wt 900-911 arm).
        if use_fwd_fallback or dgrad_tile in _DGRAD_WT_TILES:
            # wt aliases are fwd kernels; metadata records them under "forward".
            _require_launchable("forward", dgrad_tile, _METADATA_ABSENT_FWD_LAUNCH_IDS)
        else:
            _require_launchable("dgrad", dgrad_tile, _METADATA_ABSENT_DGRAD_LAUNCH_IDS)
        dgrad_fn = _C.mask_gemm.fwd if use_fwd_fallback else _C.mask_gemm.dgrad

        # Single launch handles groups=1 and groups>1 via grid.z
        C_in_total = C_in_g * groups
        grad_in = torch.zeros((N_in, C_in_total), dtype=dgrad_out_dtype, device=grad_output.device)
        status = dgrad_fn(
            _go,
            _w_dgrad,
            grad_in,
            rev_pt,
            rev_pm,
            rev_as,
            K,
            dgrad_tile,
            mask_words,
            identity_offset_bwd,
            1.0,
            groups,
        )
        if status != 0:
            raise RuntimeError(f"mask_gemm dgrad failed: status={status}")

        grad_in = grad_in.to(dtype=orig_dtype)
        if s_go is not None:
            # Exact power-of-two multiply-back (0-dim device tensors, no
            # sync; scale is 1.0 for in-range inputs so this is a no-op
            # numerically). Gated on the host-known fp32 path only.
            dgrad_scale = s_go * s_w if dgrad_w_is_scaled else s_go
            grad_in = grad_in * dgrad_scale

    if needs_input_grad[1]:
        backend = _C.mask_gemm

        # Wgrad via mask_gemm wgrad kernel with reduced_mask
        pair_table, pair_mask, mask_argsort = _get_mask_data(
            kernel_map, num_out_coords, grad_output.device
        )

        # Build reduced_mask (cached on kernel_map)
        if not hasattr(kernel_map, "_reduced_mask") or kernel_map._reduced_mask is None:
            kernel_map._reduced_mask = backend.build_reduced_mask(
                pair_mask, mask_argsort, 32, mask_words
            )

        # Select wgrad tile based on per-group channel dims
        vec_width = 16 // _in.element_size()
        if C_in_g % vec_width != 0 or C_out_g % vec_width != 0:
            wgrad_tile = 73
        else:
            wgrad_tile = tile_id
            if (
                wgrad_tile not in _METADATA_ABSENT_WGRAD_LAUNCH_IDS
                and wgrad_tile not in known_tile_ids("wgrad")
            ):
                # Shared pool entries carry fwd/dgrad tile ids in params; the
                # binding's wgrad switch routes ids with no WgradTile arm to
                # canonical tile 0. Mirror that mapping BEFORE authorization so
                # the gate checks the tile that will actually run (a wgrad-
                # namespace id that fails arch/backend still raises below).
                wgrad_tile = 0
        _require_launchable("wgrad", wgrad_tile, _METADATA_ABSENT_WGRAD_LAUNCH_IDS)

        # Single launch: [K, G, C_in_g, C_out_g] for groups>1, [K, C_in_g, C_out_g] for groups=1
        if groups == 1:
            grad_weight = torch.zeros(
                (K, C_in_g, C_out_g), dtype=torch.float32, device=grad_output.device
            )
        else:
            grad_weight = torch.zeros(
                (K, groups, C_in_g, C_out_g),
                dtype=torch.float32,
                device=grad_output.device,
            )
        status = backend.wgrad(
            _in,
            _go,
            grad_weight,
            pair_table,
            pair_mask,
            mask_argsort,
            kernel_map._reduced_mask,
            K,
            wgrad_tile,
            split_k,
            1.0,
            groups,
        )
        if status != 0:
            raise RuntimeError(f"mask_gemm wgrad failed: status={status}")
        if s_go is not None:
            # grad_weight buffer is fp32; exact pow2 multiply-back before the
            # final cast (0-dim device tensors, no sync).
            grad_weight = grad_weight * (s_go * s_in)
        grad_weight = grad_weight.to(dtype=weight.dtype)

    return grad_in, grad_weight
