import unittest

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from warp.convnet.geometry.ops.neighbor_search_continuous import (
    NEIGHBOR_SEARCH_MODE,
    NeighborSearchArgs,
)
from warp.convnet.geometry.point_collection import PointCollection
from warp.convnet.nn.mlp import MLPBlock, PointCollectionTransform
from warp.convnet.nn.point_conv import PointConv


class TestFSDP(unittest.TestCase):
    def setUp(self):
        self.B, min_N, max_N, self.C = 3, 1000, 10000, 7
        self.Ns = torch.randint(min_N, max_N, (self.B,))
        self.coords = [torch.rand((N, 3)) for N in self.Ns]
        self.features = [torch.rand((N, self.C), requires_grad=True) for N in self.Ns]
        self.pc = PointCollection(self.coords, self.features)

    def test_fsdp(self):
        device = dist.get_rank()
        print(f"Rank {device} is running test_fsdp")
        pc = self.pc.to(device=device)
        # Create conv layer
        in_channels, out_channels = self.C, 16
        search_arg = NeighborSearchArgs(
            mode=NEIGHBOR_SEARCH_MODE.RADIUS,
            radius=0.1,
        )
        torch.cuda.set_device(device)
        model = nn.Sequential(
            PointConv(
                in_channels,
                out_channels,
                neighbor_search_args=search_arg,
            ),
            PointCollectionTransform(
                MLPBlock(out_channels, hidden_channels=32, out_channels=out_channels)
            ),
        ).to(device)

        fsdp_model = FSDP(model)
        # print the model only on rank 0
        if device == 0:
            print(fsdp_model)

        fsdp_model.train()
        optim = torch.optim.Adam(fsdp_model.parameters(), lr=0.0001)
        for _ in range(100):
            out = fsdp_model(pc)
            loss = out.features.mean()
            loss.backward()
            optim.step()


if __name__ == "__main__":
    """
    Run with torch run

    torchrun --nproc_per_node=2 warp/convnet/tests/test_fsdp.py
    """
    dist.init_process_group(backend="nccl")
    unittest.main()
