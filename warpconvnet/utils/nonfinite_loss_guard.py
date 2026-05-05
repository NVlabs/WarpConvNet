# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Optional

import torch


class NonFiniteLossGuard:
    """Tolerate transient NaN / inf losses without taking down training.

    Attention models under AMP fp16 (or with small batches feeding BatchNorm)
    occasionally emit a non-finite loss for a single step. Stepping the
    optimizer with such a loss corrupts parameters and follow-up CUDA
    scatter/gather kernels fire async out-of-bounds assertions that are
    hard to triage.

    Use as a one-step-at-a-time gate in the training loop:

    .. code-block:: python

        guard = NonFiniteLossGuard(max_nonfinite=5)
        for step, batch in enumerate(loader):
            loss = compute_loss(...)
            if not guard.check(loss, epoch=epoch, step=step):
                if scaler is not None:
                    scaler.update()  # keep GradScaler state coherent
                continue
            loss.backward()
            optimizer.step()

    ``check()`` returns ``True`` when ``loss`` is finite (caller proceeds
    with backward / step). On a non-finite loss the streak is incremented
    and a warning is logged; ``check()`` returns ``False`` and the caller
    is expected to skip the update. Streak resets on any finite step.

    Once the streak exceeds ``max_nonfinite`` consecutive non-finite
    losses, ``check()`` raises ``RuntimeError`` so a divergent run aborts
    instead of silently producing garbage.
    """

    def __init__(self, max_nonfinite: int = 5):
        self.max_nonfinite = max_nonfinite
        self.streak = 0

    def check(
        self,
        loss: torch.Tensor,
        *,
        epoch: Optional[int] = None,
        step: Optional[int] = None,
    ) -> bool:
        if torch.isfinite(loss):
            self.streak = 0
            return True

        self.streak += 1
        if self.streak > self.max_nonfinite:
            raise RuntimeError(
                f"Training loss has been non-finite for {self.streak} "
                f"consecutive steps (epoch={epoch}, step={step}). Lower the "
                f"learning rate, disable AMP, or check the model config."
            )
        print(
            f"[warn] non-finite loss at epoch {epoch} step {step} "
            f"(streak {self.streak}/{self.max_nonfinite}); skipping update."
        )
        return False
