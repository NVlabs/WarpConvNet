# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest

import torch

from warpconvnet.geometry.coords.real import RealCoords
from warpconvnet.geometry.coords.search.knn import knn_search
from warpconvnet.geometry.coords.ops.voxel import voxel_downsample_mapping
from warpconvnet.geometry.types.points import (
    Points,
)
from warpconvnet.geometry.types.voxels import Voxels


class TestVoxelOps(unittest.TestCase):
    def setUp(self) -> None:
        # Set random seed
        torch.manual_seed(0)

        self.B, min_N, max_N, self.C = 3, 100000, 1000000, 7
        self.Ns = torch.randint(min_N, max_N, (self.B,))
        self.voxel_size = 0.025

        self.coords = [torch.rand((N, 3)) for N in self.Ns]
        self.features = [torch.rand((N, self.C)) for N in self.Ns]
        self.pc = Points(self.coords, self.features)

        return super().setUp()

    def test_voxel_down_mapping(self):
        device = "cuda:0"
        pc: Points = self.pc.to(device)

        # Downsample the coordinates
        downsampled_pc = pc.voxel_downsample(self.voxel_size)

        # Find the mapping
        up_map, down_map, valid = voxel_downsample_mapping(
            pc.coordinate_tensor,
            pc.offsets,
            downsampled_pc.coordinate_tensor,
            downsampled_pc.offsets,
            self.voxel_size,
        )

        # Check the mapping
        up_coords = torch.floor(pc.coordinate_tensor[up_map] / self.voxel_size)
        down_coords = torch.floor(downsampled_pc.coordinate_tensor[down_map] / self.voxel_size)
        self.assertTrue(torch.allclose(up_coords, down_coords))

    def test_voxel_down_mapping_sparse(self):
        device = "cuda:0"
        voxel_size = 0.025
        pc = self.pc.to(device)
        st: Voxels = pc.to_voxels(voxel_size)

        # Find the mapping
        up_map, down_map, valid = voxel_downsample_mapping(
            pc.coordinate_tensor,
            pc.offsets,
            st.coordinates * voxel_size,
            st.offsets,
            self.voxel_size,
        )

        # Check the mapping
        up_coords = torch.floor(pc.coordinates[up_map] / self.voxel_size).int()
        down_coords = torch.floor(st.coordinates[down_map])
        self.assertTrue(torch.allclose(up_coords, down_coords))

        # Add points that do not have a corresponding downsample point
        new_coords = 2 * torch.randn(100, 3).to(device)
        random_indices = torch.randint(0, pc.offsets[-1], (100,)).to(device)
        coordinates = pc.coordinates
        coordinates[random_indices] = new_coords

        new_pc = pc.replace(batched_coordinates=RealCoords(coordinates, offsets=pc.offsets))
        up_coords = torch.floor(new_pc.coordinate_tensor / voxel_size).int()
        # Find the mapping
        up_map, down_map, valid = voxel_downsample_mapping(
            up_coords,
            new_pc.offsets,
            st.coordinates,
            st.offsets,
            find_nearest_for_invalid=True,
        )

        for b in range(self.B):
            up_start, up_end = new_pc.offsets[b], new_pc.offsets[b + 1]
            down_start, down_end = st.offsets[b], st.offsets[b + 1]
            curr_up_coords = up_coords[up_start:up_end]
            curr_down_coords = st.coordinates[down_start:down_end]
            curr_up_map = up_map[up_start:up_end]
            curr_down_map = down_map[up_start:up_end]

            # knn
            knn_indices = knn_search(curr_down_coords.float(), curr_up_coords.float(), k=1).view(
                -1
            )
            knn_indices += down_start
            knn_neq = curr_down_map != knn_indices
            self.assertTrue(not knn_neq.any())


if __name__ == "__main__":
    unittest.main()
