# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Image-conditioning feature extractor (DinoV3 ViT).

Thin wrapper over `transformers.DINOv3ViTModel`. Mirrors
`trellis2.modules.image_feature_extractor.DinoV3FeatureExtractor` exactly.
"""

from __future__ import annotations

from typing import List, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms


class DinoV3FeatureExtractor:
    """Extract patch features from a PIL image batch.

    Returns ``(B, N_patches, D)`` LayerNorm-normalized hidden states (D=1024
    for ``facebook/dinov3-vitl16-pretrain-lvd1689m``).
    """

    def __init__(
        self,
        model_name: str = "facebook/dinov3-vitl16-pretrain-lvd1689m",
        image_size: int = 512,
    ):
        from transformers import DINOv3ViTModel  # lazy import — heavy

        self.model_name = model_name
        self.model = DINOv3ViTModel.from_pretrained(model_name)
        self.model.eval()
        self.image_size = image_size
        self.transform = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

    def to(self, device: str | torch.device) -> DinoV3FeatureExtractor:
        self.model.to(device)
        return self

    def cuda(self) -> DinoV3FeatureExtractor:
        self.model.cuda()
        return self

    def cpu(self) -> DinoV3FeatureExtractor:
        self.model.cpu()
        return self

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def extract_features(self, image: torch.Tensor) -> torch.Tensor:
        image = image.to(self.model.embeddings.patch_embeddings.weight.dtype)
        hidden = self.model.embeddings(image, bool_masked_pos=None)
        pos = self.model.rope_embeddings(image)
        # transformers <5.0 had ``m.layer``; >=5.0 nests as ``m.model.layer``.
        layers = getattr(self.model, "layer", None)
        if layers is None:
            layers = self.model.model.layer
        for layer in layers:
            hidden = layer(hidden, position_embeddings=pos)
        return F.layer_norm(hidden, hidden.shape[-1:])

    @torch.no_grad()
    def __call__(self, image: torch.Tensor | list[Image.Image]) -> torch.Tensor:
        device = self.device
        if isinstance(image, torch.Tensor):
            assert image.ndim == 4
            x = image.to(device)
        elif isinstance(image, list):
            assert all(isinstance(i, Image.Image) for i in image)
            arrs = [
                np.array(
                    i.resize((self.image_size, self.image_size), Image.LANCZOS).convert("RGB")
                ).astype(np.float32)
                / 255
                for i in image
            ]
            x = torch.stack([torch.from_numpy(a).permute(2, 0, 1).float() for a in arrs]).to(
                device
            )
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")
        x = self.transform(x)
        return self.extract_features(x)


__all__ = ["DinoV3FeatureExtractor"]
