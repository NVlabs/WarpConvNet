# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from warpconvnet.geometry.coords.search.search_configs import RealSearchConfig, RealSearchMode
from warpconvnet.geometry.types.points import Points
from warpconvnet.nn.modules.mlp import MLPBlock
from warpconvnet.nn.modules.point_conv import PointConv
from warpconvnet.ops.reductions import REDUCTIONS


@unittest.skipUnless(
    dist.is_available() and dist.is_initialized(),
    "requires an initialized torch.distributed process group (run via torchrun)",
)
class TestFSDP(unittest.TestCase):
    def setUp(self):
        self.B, min_N, max_N, self.C = 3, 1000, 10000, 7
        self.Ns = torch.randint(min_N, max_N, (self.B,))
        self.coords = [torch.rand((N, 3)) for N in self.Ns]
        self.features = [torch.rand((N, self.C), requires_grad=True) for N in self.Ns]
        self.pc = Points(self.coords, self.features)

    def test_fsdp(self):
        device = dist.get_rank()
        print(f"Rank {device} is running test_fsdp")

        # Create conv layer
        in_channels, out_channels = self.C, 16
        search_arg = RealSearchConfig(
            mode=RealSearchMode.RADIUS,
            radius=0.4,
        )
        torch.cuda.set_device(device)
        model = nn.Sequential(
            PointConv(
                in_channels,
                out_channels,
                neighbor_search_args=search_arg,
                pooling_reduction=REDUCTIONS.MEAN,
                pooling_voxel_size=0.2,
                out_point_type="downsample",
            ),
            MLPBlock(out_channels, hidden_channels=32, out_channels=out_channels),
        ).to(device)

        fsdp_model = FSDP(model)
        # print the model only on rank 0
        if device == 0:
            print(fsdp_model)

        fsdp_model.train()
        optim = torch.optim.Adam(fsdp_model.parameters(), lr=0.0001)
        for _ in range(100):
            pc = self.pc.to(device)
            pc = pc.voxel_downsample(0.1)
            out = fsdp_model(pc)
            assert out.voxel_size is not None
            loss = out.feature_tensor.mean()
            loss.backward()
            optim.step()


if __name__ == "__main__":
    """
    Run with torch run

    torchrun --nproc_per_node=2 warp/convnet/tests/test_fsdp.py
    """
    dist.init_process_group(backend="nccl")
    unittest.main()
