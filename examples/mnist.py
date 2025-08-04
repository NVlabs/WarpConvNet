# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import fire
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import warp as wp
from torch import Tensor
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, transforms

import warpconvnet.nn.functional.transforms as T
from warpconvnet.geometry.types.voxels import Voxels
from warpconvnet.nn.functional.sparse_pool import sparse_max_pool
from warpconvnet.nn.modules.sequential import Sequential
from warpconvnet.nn.modules.sparse_conv import SparseConv2d


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        # must use Sequential here to use spatial ops such as SparseConv2d
        self.layers = Sequential(
            SparseConv2d(1, 32, kernel_size=3, stride=1),
            nn.ReLU(),
            SparseConv2d(32, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Dropout(0.25),
            nn.Linear(14 * 14 * 64, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 10),
        )

    def forward(self, x: Tensor):
        x = Voxels.from_dense(x)
        x = self.layers(x)
        x = sparse_max_pool(x, kernel_size=(2, 2), stride=(2, 2))
        x = x.to_dense(channel_dim=1, spatial_shape=(14, 14))
        x = torch.flatten(x, 1)
        x = self.head(x)
        output = F.log_softmax(x, dim=1)
        return output


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print(
                f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} "
                f"({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}"
            )


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction="sum").item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100.0 * correct / len(test_loader.dataset)
    print(
        f"\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n"
    )
    return accuracy


def main(
    batch_size: int = 128,
    test_batch_size: int = 1000,
    epochs: int = 100,
    lr: float = 1e-3,
    scheduler_step_size: int = 10,
    gamma: float = 0.7,
    device: str = "cuda",
):
    wp.init()
    device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")

    torch.manual_seed(1)

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            "../data",
            train=True,
            download=True,
            transform=transforms.Compose([transforms.ToTensor()]),
        ),
        batch_size=batch_size,
        shuffle=True,
    )

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            "../data",
            train=False,
            transform=transforms.Compose([transforms.ToTensor()]),
        ),
        batch_size=test_batch_size,
        shuffle=True,
    )

    model = Net().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=scheduler_step_size, gamma=gamma)

    for epoch in range(1, epochs + 1):
        train(model, device, train_loader, optimizer, epoch)
        accuracy = test(model, device, test_loader)
        scheduler.step()

    print(f"Final accuracy: {accuracy:.2f}%")


if __name__ == "__main__":
    fire.Fire(main)
