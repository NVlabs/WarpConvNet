# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""
Auto-tune cache pre-warming utility.

Runs a few diverse batches through a model to populate the auto-tune cache
for all (C_in, C_out, kv, log10(N)) combinations before timed training begins.
This eliminates auto-tune spikes during the first epoch.

Usage:
    from warpconvnet.utils.autotune_warmup import pre_autotune

    model = MinkUNet34(3, 20).cuda()
    pre_autotune(model, train_loader, num_batches=10)
    # Now train — no more auto-tune spikes
"""

import torch
from typing import Optional
from torch.utils.data import DataLoader

from warpconvnet.utils.logger import get_logger

logger = get_logger(__name__)


def pre_autotune(
    model: torch.nn.Module,
    dataloader: DataLoader,
    num_batches: int = 10,
    prepare_fn=None,
    amp_dtype: Optional[torch.dtype] = torch.float16,
):
    """Pre-warm the auto-tune cache by running diverse batches through the model.

    This ensures all (C_in, C_out, kv, log10(N)) cache bins are populated
    before timed training begins. After calling this, subsequent forward/backward
    passes will not trigger auto-tuning.

    Args:
        model: The model to pre-warm (e.g., MinkUNet34).
        dataloader: Training dataloader. Should produce diverse N (voxel counts).
        num_batches: Number of batches to run. 10 is usually sufficient to cover
            all log10(N) bins for a typical ScanNet dataset.
        prepare_fn: Optional function to convert a batch to model input.
            Signature: prepare_fn(batch) -> input_for_model.
            If None, assumes batch is a dict with 'coords' and 'colors' keys
            and creates Voxels directly.
        amp_dtype: Autocast dtype. Set to None to disable autocast.
    """
    from warpconvnet.geometry.types.voxels import Voxels

    model.eval()  # Use eval to avoid BN running stats updates
    data_iter = iter(dataloader)

    logger.info(f"Pre-warming auto-tune cache with {num_batches} batches...")

    for i in range(num_batches):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        if prepare_fn is not None:
            inputs = prepare_fn(batch)
        else:
            # Default: assume batch has 'coords' and 'colors' keys
            coords = [c.int() if isinstance(c, torch.Tensor) else torch.tensor(c).int()
                      for c in batch['coords']]
            feats = [f.float() if isinstance(f, torch.Tensor) else torch.tensor(f).float()
                     for f in batch['colors']]
            inputs = Voxels(coords, feats, device='cuda')

        # Forward pass (populates forward auto-tune cache)
        if amp_dtype is not None:
            with torch.autocast('cuda', dtype=amp_dtype):
                output = model(inputs)
        else:
            output = model(inputs)

        # Backward pass (populates backward auto-tune cache)
        if hasattr(output, 'feature_tensor'):
            loss = output.feature_tensor.sum()
        elif isinstance(output, torch.Tensor):
            loss = output.sum()
        else:
            loss = output.features.sum() if hasattr(output, 'features') else None

        if loss is not None:
            loss.backward()

        torch.cuda.synchronize()

    model.train()  # Restore training mode
    # Clear gradients accumulated during warmup
    model.zero_grad(set_to_none=True)

    logger.info("Auto-tune cache pre-warming complete.")
