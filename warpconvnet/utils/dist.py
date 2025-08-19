# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import torch
import torch.distributed as dist


def _get_current_rank() -> int:
    """Get current process rank for distributed training.
    
    This function checks multiple sources for rank information in order of preference:
    1. RANK environment variable (set by torchrun)
    2. LOCAL_RANK + WORLD_SIZE environment variables
    3. SLURM_PROCID environment variable (for SLURM environments)
    4. PyTorch distributed rank (if initialized)
    5. Default to 0
    
    Returns:
        int: The current process rank
    """
    if "RANK" in os.environ:
        return int(os.environ["RANK"])
    elif "LOCAL_RANK" in os.environ and "WORLD_SIZE" in os.environ:
        return int(os.environ["LOCAL_RANK"])
    elif "SLURM_PROCID" in os.environ:
        return int(os.environ["SLURM_PROCID"])
    else:
        # Fallback to PyTorch distributed if available
        if dist.is_initialized():
            return dist.get_rank()
        return 0


def _is_rank_zero() -> bool:
    """Check if current process is rank 0."""
    return _get_current_rank() == 0


def get_world_size() -> int:
    """Get the total number of processes in distributed training.
    
    Returns:
        int: The world size (total number of processes)
    """
    if "WORLD_SIZE" in os.environ:
        return int(os.environ["WORLD_SIZE"])
    elif dist.is_initialized():
        return dist.get_world_size()
    else:
        return 1


def is_distributed() -> bool:
    """Check if we're running in distributed mode.
    
    Returns:
        bool: True if running in distributed mode, False otherwise
    """
    return get_world_size() > 1
