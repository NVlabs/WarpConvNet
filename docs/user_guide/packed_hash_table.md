# Packed Hash Table

**Created**: 2026-04-19 06:08:39
**Edited**: 2026-04-19 06:08:39

`PackedHashTable` is WarpConvNet's coordinate lookup structure: a GPU hash
table from 4D integer coordinates $(b, x, y, z)$ to their row index in a
voxel geometry's coordinate tensor. It is the data structure that makes
sparse convolution's kernel-map construction fast — every output row has
to locate up to $|\mathcal{K}|$ input neighbors by coordinate, and a
hash-table probe per neighbor is the inner loop of the whole pipeline.

This page explains how it's laid out, why it uses 64-bit packed keys,
and how the **hierarchical coarse→fine search** accelerates lookups for
large kernels ($K \ge 125$, i.e. $5^3$ and above).

## Why a packed hash table

The previous layout (`TorchHashTable`) stored keys as `int32[N, 4]`
vectors, compared them elementwise on probe, and hashed with Murmur over
4 ints. That cost:

- 128-bit load per key on every probe (two 64-bit loads).
- A 4-way elementwise equality check.
- A multi-round hash mix.

The packed layout collapses all three into a single 64-bit integer:

- **One 64-bit load** per probe (one instruction on SM80+).
- **One `==` compare** — the whole coordinate fits in a register.
- **Splitmix64** — two multiplies + three xor-shifts, excellent
  avalanche for arbitrary 64-bit keys.

Net effect: **~2.5–6x faster kernel-map generation** relative to the
vector-key layout. Kept `TorchHashTable` available for the radius-search
path which uses 3D real coordinates; all voxel/discrete search paths
have migrated to `PackedHashTable`.

Source: `warpconvnet/geometry/coords/search/packed_hashmap.py`,
`warpconvnet/csrc/cuhash_hash_table.cu`,
`warpconvnet/csrc/include/cuhash/hash_functions.cuh`.

## Key layout

A 4D coordinate $(b, x, y, z)$ packs into a single `uint64_t`:

```
 MSB                                                                 LSB
┌──────┬──────────┬──────────────────┬──────────────────┬──────────────────┐
│ bit  │   b      │        x         │        y         │        z         │
│ 63   │ 9 bits   │     18 bits      │     18 bits      │     18 bits      │
│ valid│ unsigned │   signed (2's)   │   signed (2's)   │   signed (2's)   │
└──────┴──────────┴──────────────────┴──────────────────┴──────────────────┘
 bit    62……54      53……36              35……18              17…… 0
```

| Field               | Bits    | Range                     | Notes                                                                                                                |
| ------------------- | ------- | ------------------------- | -------------------------------------------------------------------------------------------------------------------- |
| `valid`             | 1       | `{1}`                     | Bit 63 is always 1 for occupied slots — `kEmpty = 0` cannot collide with any legitimate packed key.                  |
| `b` (batch)         | 9       | $[0, 511]$                | Unsigned.                                                                                                            |
| `x, y, z` (spatial) | 18 each | $[-131{,}072, 131{,}071]$ | Signed two's-complement. At 0.01 m voxel size this spans $\pm 1.3$ km per axis, adequate for LiDAR / outdoor scenes. |

Compile-time constants live in `cuhash/hash_functions.cuh`:
`kBatchMax=511`, `kCoordMin=-131072`, `kCoordMax=131071`,
`kValidBit=1ULL<<63`. The Python `PackedHashTable` class exposes
`BATCH_MAX`, `COORD_MIN`, `COORD_MAX` and validates on `insert()`.

## Hashing

Every packed key goes through **Splitmix64** before the capacity mask:

```cpp
__device__ uint32_t hash(uint64_t key, uint32_t capacity_mask) {
  key ^= key >> 30;
  key *= 0xBF58476D1CE4E5B9ull;
  key ^= key >> 27;
  key *= 0x94D049BB133111EBull;
  key ^= key >> 31;
  return static_cast<uint32_t>(key) & capacity_mask;
}
```

Two multiplies plus three xor-shifts. Excellent avalanche (flipping any
single input bit flips half the output bits on average) with minimal
instruction count. The final `& capacity_mask` reduces the 32-bit output
modulo capacity — **capacity is always a power of two**, so the modulo
is one bit-and instruction.

A secondary hash (`double_hash_stride`) is available for the
double-hashing probe strategy; it returns an **odd** stride so that
every slot is reachable under power-of-2 capacity (odd is coprime with
$2^k$).

## Probe strategies

Three `SearchMode` values are exposed:

| Mode          | Probe sequence                                                | When to use                                                                                             |
| ------------- | ------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------- |
| `LINEAR`      | $h, h+1, h+2, \dots$                                          | Default. Cache-friendly; fastest at low load factor ($< 0.5$).                                          |
| `DOUBLE_HASH` | $h, h+s, h+2s, \dots$ with odd $s$ from the secondary hash    | Better clustering behavior at higher load factors. Use if you cannot guarantee table capacity $\ge 2N$. |
| `WARP_COOP`   | Warp-cooperative probe; 32 threads probe 32 slots in parallel | Experimental; rarely wins unless table is nearly full.                                                  |

Default load factor is **0.5** (`from_coords` rounds capacity up to
$\text{next\_pow2}(2N)$), which keeps linear probes short. If
`use_double_hash=True` is set at construction, `LINEAR` is auto-upgraded
to `DOUBLE_HASH` on search to match the insertion strategy.

## Python API

```python
from warpconvnet.geometry.coords.search.packed_hashmap import (
    PackedHashTable, SearchMode,
)
```

### Constructing

```python
# Factory: build and insert in one call
ht = PackedHashTable.from_coords(
    coords,          # int32 [N, 4] on CUDA
    use_double_hash=False,
)

# Or explicit construction
ht = PackedHashTable(capacity=1 << 20, device="cuda")
ht.insert(coords)
```

Capacity is rounded up to the next power of two. `insert()` raises
`ValueError` on out-of-range batch/spatial values and `RuntimeError` if
the table fills during insertion.

### Searching

```python
# Returns int32 [M] of original indices, or -1 if not found
indices = ht.search(query_coords, mode=SearchMode.LINEAR)
```

### Generative coordinate expansion

`PackedHashTable` is the backing structure that produces the output
coordinate set for [generative convolution](./sparse_convolutions.md#how-mathcalctextout-is-chosen).
The expansion primitive is `expand_with_offsets`: it inserts every
$(\text{base} + \text{offset})$ combination atomically and deduplicates
via the hash table itself (a second insert of the same packed key is a
no-op).

User-facing call chain:

```
SparseConv3d(generative=True)
   └─ _apply_generative_policy  (warpconvnet/nn/functional/sparse_conv/helper.py)
       └─ IntCoords.expand      (warpconvnet/geometry/coords/integer.py)
           └─ expand_coords     (warpconvnet/geometry/coords/ops/expand.py)
               └─ PackedHashTable.expand_with_offsets   ← this page
```

`expand_coords` chunks the kernel offsets (`kernel_batch` at a time),
grows the hash table if the next chunk would exceed load factor 0.5,
and reads the deduplicated result from `ht.vector_keys` at the end. The
result is the set of all grid points reachable by *any* offset from any
input coordinate — exactly the $\mathcal{C}^\text{out}$ of the
generative regime.

Direct use of `expand_with_offsets` is rare; most callers go through
`IntCoords.expand()`.

### Unique-index recovery

```python
# Deduplicate and return sorted unique row indices
unique_idx = ht.unique_index  # int32 tensor
```

## Hierarchical coarse→fine search

The flat kernel-map search probes one hash-table entry per
(output row) × (kernel offset) pair. For small kernels ($K=27$ for a
$3^3$ kernel) that's fine. For large kernels — $K=125$ for $5^3$,
$K=343$ for $7^3$, and beyond — the flat search becomes the bottleneck
because most offsets land on empty coordinates in 3D sparse data.

Hierarchical search exploits this: **probe a coarse grid first, then
skip fine probes whose coarse cell is empty.**

### Two-level structure

A **coarse hash table** is built at stride $S$ (power of two, default
$S = 4$) over the same coordinate set. For every occupied fine
coordinate $(b, x, y, z)$, the coarse table stores
$(b, \lfloor x/S \rfloor, \lfloor y/S \rfloor, \lfloor z/S \rfloor)$.
Arithmetic right-shift `>> log2(S)` gives the floor-divide in one
instruction for power-of-two stride.

For kernel size $K$ and stride $S$, the **coarse kernel footprint** is

$$
K_c = \left\lceil \frac{K}{S} \right\rceil^3
$$

For $K = 7$, $S = 4$: $K_c = 2^3 = 8$ coarse cells cover the kernel
(rounded up so the coarse neighbourhood contains every fine offset).
In practice warpconvnet uses $K_c = 27$ (a $3^3$ coarse
neighbourhood) to provide safety margin; the kernel schedules
$K_c \le 32$ so a single `uint32` bitmask holds the result.

### Two-pass algorithm

```
Input: fine hash table H_f (capacity C_f)
       coarse hash table H_c (capacity C_c, built from fine coords at stride S)
       query coords Q: int32 [M, 4]
       fine kernel offsets O_f: int32 [K, 4]
       coarse kernel offsets O_c: int32 [K_c, 4]

Pass 1 — coarse_probe (per-query, K_c probes each):
  for q in 0..M:
    mask = 0
    for c in 0..K_c:
      coarse_q = (q.b, floor(q.xyz / S) + O_c[c])
      if search(H_c, coarse_q) >= 0:
        mask |= (1 << c)
    coarse_masks[q] = mask

Pass 2 — fine_search_pruned ((M, K) grid):
  for q, k in (0..M, 0..K):
    coarse_idx = which coarse cell does O_f[k] map to?
    if (coarse_masks[q] >> coarse_idx) & 1 == 0:
      found[k, q] = -1      # skipped, coarse cell empty
    else:
      found[k, q] = search(H_f, q + O_f[k])
```

Expected savings for $7^3$ at typical indoor-scene occupancy
(~20% of coarse cells non-empty):

- **Flat search**: $K = 343$ fine probes per query.
- **Hierarchical**: $K_c = 27$ coarse probes + $0.2 \cdot 343 \approx 69$
  fine probes ≈ **96 total**, a ~3.6× reduction.

### Fused C++ launcher

`_C.cuhash.hierarchical_kernel_map` does the entire pipeline — coarse
table build, pass 1, pass 2, postprocess count + scatter, pair-table
build — in **one Python-to-C++ host call**. No Python round-trips
between kernel launches; all intermediate buffers live on the device.

Python side:

```python
from warpconvnet.geometry.coords.search.hierarchical_search import (
    kernel_map_from_size_hierarchical,
)
result = kernel_map_from_size_hierarchical(
    fine_ht,
    query_coords,
    kernel_size=(7, 7, 7),
    stride=4,       # coarse stride, must be power of 2
)
# result.in_maps, result.out_maps, result.offsets, result._pair_table
```

### Auto-selection: when does the hierarchical path run?

The dispatcher in
`warpconvnet/geometry/coords/search/torch_discrete.py::_kernel_map_from_size`
selects the hierarchical path automatically:

```python
_K = prod(kernel_size)
is_odd_kernel_all = all(k % 2 == 1 for k in kernel_size)
if _K >= 125 and is_odd_kernel_all and not skip_symmetric_kernel_map:
    # hierarchical (coarse + pruned fine)
    ...
else:
    # flat search
    ...
```

Concrete gating:

| Kernel                                     | $K$ | Path                   |
| ------------------------------------------ | --- | ---------------------- |
| $3 \times 3 \times 3$                      | 27  | Flat                   |
| $5 \times 5 \times 5$                      | 125 | **Hierarchical**       |
| $7 \times 7 \times 7$                      | 343 | **Hierarchical**       |
| $3 \times 3 \times 5$ (non-cubic, all odd) | 45  | Flat (below threshold) |
| $2 \times 2 \times 2$ (even)               | 8   | Flat                   |

The even-kernel guard is because the coarse-footprint math assumes
symmetric offsets around the centre, which only holds for odd kernel
sizes in all axes.

## When NOT to use the packed table

The packed layout assumes:

- **4D integer coordinates** with batch in $[0, 511]$ and spatial in
  $\pm 131\,072$.
- Discrete voxel search (no radius, no kNN).

For continuous-space point-cloud neighbour search (radius / kNN on
real-valued coordinates), use the legacy `TorchHashTable` — it's kept
precisely because `PackedHashTable` doesn't apply to that case. See
`warpconvnet/geometry/coords/search/torch_hashmap.py`.

## Source files

| File                                                        | Purpose                                                                                     |
| ----------------------------------------------------------- | ------------------------------------------------------------------------------------------- |
| `warpconvnet/geometry/coords/search/packed_hashmap.py`      | Python `PackedHashTable` class + `SearchMode` enum.                                         |
| `warpconvnet/geometry/coords/search/hierarchical_search.py` | Python wrapper for the fused hierarchical launcher.                                         |
| `warpconvnet/geometry/coords/search/torch_discrete.py`      | Dispatcher: chooses flat vs hierarchical per $K$.                                           |
| `warpconvnet/csrc/cuhash_hash_table.cu`                     | `packed_prepare` / `packed_insert` / `packed_search` / `packed_expand_insert` CUDA kernels. |
| `warpconvnet/csrc/cuhash_kernel_map.cu`                     | Flat + hierarchical kernel-map kernels.                                                     |
| `warpconvnet/csrc/include/cuhash/hash_functions.cuh`        | Key packing, `pack_key_4d` / `unpack_key_4d`, `Splitmix64Hash`, double-hash stride.         |
| `warpconvnet/csrc/bindings/cuhash_bindings.cpp`             | pybind11 wrappers (`_C.cuhash.*`).                                                          |
