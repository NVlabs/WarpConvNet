# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from warpconvnet.dataset.scannet import ScanNetInstanceDataset
from warpconvnet.geometry.types.points import Points
from warpconvnet.models.maskformer import MaskFormer
from warpconvnet.models.mink_unet import MinkUNet18
from warpconvnet.nn.modules.sparse_pool import PointToVoxel

SCANNET_HF = "~/datasets/scannet_hf"


def _make_pc(samples, device):
    return Points.from_list_of_coordinates(
        [torch.from_numpy(s["coords"]).float() for s in samples],
        features=[torch.from_numpy(s["colors"]).float() / 255.0 for s in samples],
    ).to(device)


def _build_maskformer(num_classes=200, hidden_dim=96, voxel_size=0.04):
    backbone = PointToVoxel(
        inner_module=MinkUNet18(in_channels=3, out_channels=hidden_dim),
        voxel_size=voxel_size,
        concat_unpooled_pc=False,
    )
    return MaskFormer(
        backbone=backbone,
        hidden_dim=hidden_dim,
        num_queries=64,
        num_heads=4,
        num_decoders=2,
        dim_feedforward=128,
        dropout=0.0,
        num_classes=num_classes,
    )


@pytest.mark.skipif(
    not os.path.isdir(os.path.expanduser(SCANNET_HF)),
    reason="scannet_hf preprocessed data not available",
)
def test_maskformer_train_step():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = ScanNetInstanceDataset(
        root=SCANNET_HF,
        split="val",
        label_set="scannet200",
        voxel_size=0.04,
    )
    B = 2
    samples = [dataset[i] for i in range(B)]
    pc = _make_pc(samples, device)

    num_classes = 200
    model = _build_maskformer(num_classes=num_classes, voxel_size=0.04).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=1e-3)

    losses = []
    for step in range(3):
        optim.zero_grad()
        logits, masks = model(pc)
        # logits: [B, Q, C+1]
        # masks:  list of [Q, N_b] per scene
        assert logits.shape == (B, 64, num_classes + 1)
        assert len(masks) == B

        # Synthetic targets: random class + binary mask supervision per scene.
        target_class = torch.randint(0, num_classes + 1, (B, 64), device=device)
        cls_loss = F.cross_entropy(logits.reshape(-1, num_classes + 1), target_class.reshape(-1))

        mask_loss = 0.0
        for m in masks:
            target_mask = (torch.rand_like(m) > 0.5).float()
            mask_loss = mask_loss + F.binary_cross_entropy_with_logits(m, target_mask)
        mask_loss = mask_loss / B

        loss = cls_loss + mask_loss
        loss.backward()
        optim.step()
        losses.append(loss.item())

    assert all(np.isfinite(v) for v in losses), losses
    print(f"losses: {losses}")
