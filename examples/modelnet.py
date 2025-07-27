# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
from typing import Optional

import fire
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import warp as wp
from jaxtyping import Float
from torch import Tensor
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from warpconvnet.dataset.modelnet import ModelNet40Dataset
from warpconvnet.geometry.coords.search.search_configs import (
    RealSearchConfig,
)
from warpconvnet.geometry.types.points import Points
from warpconvnet.geometry.types.voxels import Voxels
from warpconvnet.nn.modules.point_conv import PointConv
from warpconvnet.nn.modules.sequential import Sequential
from warpconvnet.nn.modules.sparse_conv import SparseConv3d
from warpconvnet.ops.reductions import REDUCTIONS


class UseAllConvNet(nn.Module):
    """
    Example network that showcases the use of point conv, sparse conv, and dense conv in one model.
    """

    def __init__(
        self,
        voxel_size: float = 0.05,
    ):
        super().__init__()

        self.point_conv = Sequential(
            PointConv(
                24,
                64,
                neighbor_search_args=RealSearchConfig("knn", knn_k=16),
            ),
            nn.LayerNorm(64),
            nn.ReLU(),
            PointConv(
                64,
                64,
                neighbor_search_args=RealSearchConfig("radius", radius=voxel_size),
            ),
            nn.LayerNorm(64),
            nn.ReLU(),
        )
        # Pooling from point to sparse tensor
        self.voxel_size = voxel_size
        # must use Sequential here to use spatial ops such as SparseConv3d
        self.sparse_conv = Sequential(
            SparseConv3d(64, 64, kernel_size=3, stride=1),
            nn.LayerNorm(64),
            nn.ReLU(),
            SparseConv3d(64, 64, kernel_size=2, stride=2),  # stride
            nn.LayerNorm(64),
            nn.ReLU(),
            SparseConv3d(64, 128, kernel_size=3, stride=1),
            nn.LayerNorm(128),
            nn.ReLU(),
            SparseConv3d(128, 256, kernel_size=2, stride=2),  # stride
            nn.LayerNorm(256),
            nn.ReLU(),
            SparseConv3d(256, 512, kernel_size=3, stride=1),
            nn.LayerNorm(512),
            nn.ReLU(),
        )
        self.dense_conv = nn.Sequential(
            nn.Conv3d(512, 1024, kernel_size=2, stride=2),
            nn.BatchNorm3d(1024),
            nn.ReLU(),
            nn.Conv3d(1024, 1024, kernel_size=2, stride=2),
            nn.BatchNorm3d(1024),
            nn.ReLU(),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(1024 * 2 * 2 * 2, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 40),  # 40 classes in ModelNet40
        )

    def forward(self, x: Float[Tensor, "B N 3"]) -> Float[Tensor, "B 40"]:
        pc: Points = Points.from_list_of_coordinates(x, encoding_channels=8, encoding_range=1)
        pc = self.point_conv(pc)
        st: Voxels = pc.to_voxels(reduction=REDUCTIONS.MEAN, voxel_size=self.voxel_size)
        st = self.sparse_conv(st)
        dt: Tensor = st.to_dense(channel_dim=1, min_coords=(-5, -5, -5), max_coords=(4, 4, 4))
        return self.dense_conv(dt)


def train(model, device, train_loader, optimizer, epoch, scheduler):
    model.train()
    bar = tqdm(train_loader)
    for batch_idx, data_dict in enumerate(bar):
        data, target = data_dict["coords"].to(device), data_dict["labels"].to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        bar.set_description(f"Loss: {loss.item():.6f}, LR: {scheduler.get_last_lr()}")


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data_dict in tqdm(test_loader):
            data, target = data_dict["coords"].to(device), data_dict["labels"].to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction="sum").item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100.0 * correct / len(test_loader.dataset)
    print(
        f"\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n"
    )
    return accuracy


def main(
    root_dir: str = "./data/modelnet40",
    batch_size: int = 32,
    test_batch_size: int = 100,
    epochs: int = 100,
    lr: float = 1e-3,
    scheduler_step_size: int = 10,
    gamma: float = 0.7,
    device: str = "cuda",
):
    wp.init()
    device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")

    train_dataset = ModelNet40Dataset(root_dir, split="train")
    test_dataset = ModelNet40Dataset(root_dir, split="test")

    print(f"Dataset root directory: {root_dir}")
    print(f"Files in root directory: {os.listdir(root_dir)}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)

    model = UseAllConvNet().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=scheduler_step_size, gamma=gamma)

    for epoch in range(1, epochs + 1):
        train(model, device, train_loader, optimizer, epoch, scheduler)
        accuracy = test(model, device, test_loader)
        scheduler.step()

    print(f"Final accuracy: {accuracy:.2f}%")


if __name__ == "__main__":
    fire.Fire(main)
