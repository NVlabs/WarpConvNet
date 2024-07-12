from typing import Literal, Optional

import torch
import torch.bin
from jaxtyping import Float, Int
from torch import Tensor

import warp as wp

snippet = """
    __shared__ int shared_offsets[128];

    int curr_tid = threadIdx.x;

    // Load offsets into shared memory
    if (curr_tid < offsets_len) {
        shared_offsets[curr_tid] = offsets[curr_tid];
    }
    __syncthreads();

    // Find bin
    int bin = -1;
    for (int i = 0; i < offsets_len - 1; i++) {
        int start = shared_offsets[i];
        int end = shared_offsets[i + 1];
        if (start <= tid && tid < end) {
            bin = i;
            break;
        }
    }

    batch_index_wp[tid] = bin;
    """


@wp.func_native(snippet)
def _find_bin_native(
    offsets: wp.array(dtype=wp.int32),
    offsets_len: int,
    tid: int,
    batch_index_wp: wp.array(dtype=wp.int32),
):
    ...


@wp.func
def _find_bin(offsets: wp.array(dtype=wp.int32), tid: int) -> int:
    N = offsets.shape[0] - 1
    bin_id = int(-1)
    for i in range(N):
        start = offsets[i]
        end = offsets[i + 1]
        if start <= tid < end:
            bin_id = i
            break
    return bin_id


@wp.kernel
def _batch_index(
    offsets: wp.array(dtype=wp.int32),
    batch_index_wp: wp.array(dtype=wp.int32),
) -> None:
    tid = wp.tid()
    if offsets.shape[0] > 128:
        batch_index_wp[tid] = _find_bin(offsets, tid)
    else:
        _find_bin_native(offsets, offsets.shape[0], tid, batch_index_wp)


def batch_index_from_offset(
    offsets: Int[Tensor, "B+1"],  # noqa: F821
    backend: Literal["torch", "warp"] = "warp",
    device: Optional[str] = None,
) -> Int[Tensor, "N"]:  # noqa: F821
    assert isinstance(offsets, torch.Tensor), "offsets must be a torch.Tensor"

    if device is not None:
        offsets = offsets.to(device)

    if backend == "torch":
        batch_index = (
            torch.arange(len(offsets) - 1)
            .to(offsets)
            .repeat_interleave(offsets[1:] - offsets[:-1])
        )
        return batch_index

    N = offsets[-1].item()
    offsets_wp = wp.from_torch(offsets.int(), dtype=wp.int32).to(device)
    batch_index_wp = wp.zeros(shape=(N,), dtype=wp.int32, device=device)
    wp.launch(
        _batch_index,
        N,
        inputs=[offsets_wp, batch_index_wp],
    )
    return wp.to_torch(batch_index_wp)


def batch_indexed_coordinates(
    batched_coords: Float[Tensor, "N 3"],  # noqa: F821
    offsets: Int[Tensor, "B + 1"],  # noqa: F821
    backend: Literal["torch", "warp"] = "warp",
) -> Float[Tensor, "N 4"]:  # noqa: F821
    device = str(batched_coords.device)
    batch_index = batch_index_from_offset(offsets, device=device, backend=backend)
    batched_coords = torch.cat([batch_index.unsqueeze(1), batched_coords], dim=1)
    return batched_coords
