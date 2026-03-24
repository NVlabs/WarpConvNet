# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
from typing import Literal, Optional, Tuple

import torch
from torch.utils.data import Dataset

from warpconvnet.geometry.coords.ops.voxel import voxel_downsample_np

SCANNET_URL = (
    "https://cvg-data.inf.ethz.ch/openscene/data/scannet_processed/scannet_3d.zip"
)


class ScanNetDataset(Dataset):
    """
    Dataset from the OpenScene project.
    """

    def __init__(
        self,
        root: str = "./data/scannet",
        split: str = "train",
        voxel_size: Optional[float] = None,
        out_type: Literal["point", "voxel"] = "voxel",
        min_coord: Optional[Tuple[float, float, float]] = None,
    ):
        super().__init__()
        self.root = root
        self.split = split
        self.voxel_size = voxel_size
        self.out_type = out_type
        if min_coord is not None:
            min_coord = torch.tensor(min_coord)
        self.min_coord = min_coord
        self.prepare_data()

    def prepare_data(self):
        # If data is not downloaded, download it
        if not os.path.exists(self.root):
            os.makedirs(self.root, exist_ok=True)
            os.system(f"wget {SCANNET_URL} -O {self.root}/scannet_3d.zip")
            os.system(f"unzip {self.root}/scannet_3d.zip -d {self.root}")
            os.system(f"mv {self.root}/scannet_3d/* {self.root}")
            os.system(f"rmdir {self.root}/scannet_3d")

        # Get split txts
        self.files = []
        with open(os.path.join(self.root, f"scannetv2_{self.split}.txt")) as f:
            self.files = sorted(f.readlines())

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        file = self.files[index]
        coords, colors, labels = torch.load(
            os.path.join(self.root, self.split, file.strip() + "_vh_clean_2.pth"),
            weights_only=False,
        )
        if self.min_coord is not None:
            coords -= self.min_coord
        # All to tensor
        if self.voxel_size is not None:
            # Use cpu for downsampling in dataloader. Should use multiple workers.
            unique_coords, to_unique_indices = voxel_downsample_np(
                coords, self.voxel_size
            )
            if self.out_type == "point":
                unique_coords = coords[to_unique_indices]
            return {
                "coords": unique_coords,
                "colors": colors[to_unique_indices],
                "labels": labels[to_unique_indices],
            }
        else:
            return {
                "coords": coords,
                "colors": colors,
                "labels": labels,
            }
