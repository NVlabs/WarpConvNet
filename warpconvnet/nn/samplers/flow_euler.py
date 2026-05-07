# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Flow-matching ODE sampler (rectified flow) with optional CFG + guidance
interval. Generic — works on any model with signature ``model(x_t, t, cond)``.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Tuple

import numpy as np
import torch
from tqdm import tqdm


__all__ = [
    "FlowEulerCfgSampler",
    "FlowEulerGuidanceIntervalSampler",
    "FlowEulerSampler",
    "Sampler",
]


def _per_sample_std(x):
    """Standard deviation across non-batch dims, returning a per-batch tensor.

    Handles dense ``(B, ...)`` Tensor and warpconvnet `Voxels` (where the
    batch is encoded by ``coords[:, 0]`` / ``offsets``). For Voxels the
    result is a ``(B, C)`` tensor — std over the per-sample voxel rows.
    """
    if isinstance(x, torch.Tensor):
        return x.std(dim=list(range(1, x.ndim)), keepdim=True)
    feats = x.feats  # (N_total, C)
    offsets = x.offsets.tolist()
    return torch.stack(
        [feats[offsets[i] : offsets[i + 1]].std(dim=0) for i in range(len(offsets) - 1)],
        dim=0,
    )


def _scalar_or_per_sample_scale(x, ratio):
    """Multiply ``x`` by a per-batch ``ratio``.

    For dense Tensor ``ratio`` already broadcasts; for Voxels we expand
    ``(B, C)`` to ``(N_total, C)`` via the batch index column.
    """
    if isinstance(x, torch.Tensor):
        return x * ratio
    per_voxel = ratio[x.coords[:, 0].long()]
    return x.replace_features(x.feats * per_voxel)


class Sampler(ABC):
    """Base class for ODE / SDE samplers."""

    @abstractmethod
    def sample(self, model, **kwargs):  # pragma: no cover
        pass


class FlowEulerSampler(Sampler):
    """Euler ODE solver for rectified-flow / flow-matching models.

    The model is assumed to predict velocity ``v = (x_1 - x_0) / 1`` given the
    noisy sample ``x_t = (1 - t) x_0 + (sigma_min + (1 - sigma_min) t) eps``.
    Stepping is plain Euler: ``x_{t-Δ} = x_t - (t - t_prev) * v``.
    """

    def __init__(self, sigma_min: float):
        self.sigma_min = sigma_min

    # -- conversion utilities (linear-interp flow parameterisation) ------------
    def _v_to_xstart_eps(self, x_t: torch.Tensor, t: float, v: torch.Tensor):
        # Use left-multiply on Voxels (no ``__rmul__`` on Geometry).
        eps = v * (1 - t) + x_t
        x_0 = x_t * (1 - self.sigma_min) - v * (self.sigma_min + (1 - self.sigma_min) * t)
        return x_0, eps

    def _pred_to_xstart(self, x_t: torch.Tensor, t: float, pred: torch.Tensor) -> torch.Tensor:
        # Use left-multiply on x_t / pred so sparse Voxels (which have
        # ``__mul__`` but no ``__rmul__``) flow through unchanged.
        return x_t * (1 - self.sigma_min) - pred * (self.sigma_min + (1 - self.sigma_min) * t)

    def _xstart_to_pred(self, x_t: torch.Tensor, t: float, x_0: torch.Tensor) -> torch.Tensor:
        return (x_t * (1 - self.sigma_min) - x_0) / (self.sigma_min + (1 - self.sigma_min) * t)

    # -- model wrapper ---------------------------------------------------------
    def _inference_model(
        self,
        model,
        x_t: torch.Tensor,
        t: float,
        cond: Any | None = None,
        **kwargs,
    ) -> torch.Tensor:
        # Voxels (Geometry) does not implement .shape; fall back to batch_size.
        n = getattr(x_t, "batch_size", None)
        if n is None:
            n = x_t.shape[0]
        t_in = torch.tensor([1000 * t] * n, device=x_t.device, dtype=torch.float32)
        return model(x_t, t_in, cond, **kwargs)

    def _get_model_prediction(self, model, x_t, t, cond=None, **kwargs):
        pred_v = self._inference_model(model, x_t, t, cond, **kwargs)
        pred_x_0, pred_eps = self._v_to_xstart_eps(x_t=x_t, t=t, v=pred_v)
        return pred_x_0, pred_eps, pred_v

    # -- single Euler step -----------------------------------------------------
    @torch.no_grad()
    def sample_once(
        self,
        model,
        x_t: torch.Tensor,
        t: float,
        t_prev: float,
        cond: Any | None = None,
        **kwargs,
    ):
        pred_x_0, _pred_eps, pred_v = self._get_model_prediction(model, x_t, t, cond, **kwargs)
        pred_x_prev = x_t - pred_v * (t - t_prev)
        return {"pred_x_prev": pred_x_prev, "pred_x_0": pred_x_0}

    # -- full sampling loop ----------------------------------------------------
    @torch.no_grad()
    def sample(
        self,
        model,
        noise: torch.Tensor,
        cond: Any | None = None,
        steps: int = 50,
        rescale_t: float = 1.0,
        verbose: bool = True,
        tqdm_desc: str = "Sampling",
        **kwargs,
    ):
        sample = noise
        t_seq = np.linspace(1, 0, steps + 1)
        t_seq = rescale_t * t_seq / (1 + (rescale_t - 1) * t_seq)
        t_pairs = list(zip(t_seq[:-1].tolist(), t_seq[1:].tolist()))
        pred_x_t_hist, pred_x_0_hist = [], []
        for t, t_prev in tqdm(t_pairs, desc=tqdm_desc, disable=not verbose):
            out = self.sample_once(model, sample, t, t_prev, cond, **kwargs)
            sample = out["pred_x_prev"]
            pred_x_t_hist.append(out["pred_x_prev"])
            pred_x_0_hist.append(out["pred_x_0"])
        return {"samples": sample, "pred_x_t": pred_x_t_hist, "pred_x_0": pred_x_0_hist}


class _ClassifierFreeGuidanceMixin:
    def _inference_model(
        self,
        model,
        x_t,
        t,
        cond,
        neg_cond,
        guidance_strength,
        guidance_rescale: float = 0.0,
        **kwargs,
    ):
        if guidance_strength == 1:
            return super()._inference_model(model, x_t, t, cond, **kwargs)
        if guidance_strength == 0:
            return super()._inference_model(model, x_t, t, neg_cond, **kwargs)
        pred_pos = super()._inference_model(model, x_t, t, cond, **kwargs)
        pred_neg = super()._inference_model(model, x_t, t, neg_cond, **kwargs)
        # Use ``Voxels.__mul__`` (left operand) to support sparse pred_v.
        pred = pred_pos * guidance_strength + pred_neg * (1 - guidance_strength)
        if guidance_rescale > 0:
            x_0_pos = self._pred_to_xstart(x_t, t, pred_pos)
            x_0_cfg = self._pred_to_xstart(x_t, t, pred)
            std_pos = _per_sample_std(x_0_pos)
            std_cfg = _per_sample_std(x_0_cfg)
            ratio = std_pos / std_cfg
            x_0_rescaled = _scalar_or_per_sample_scale(x_0_cfg, ratio)
            x_0 = x_0_rescaled * guidance_rescale + x_0_cfg * (1 - guidance_rescale)
            pred = self._xstart_to_pred(x_t, t, x_0)
        return pred


class _GuidanceIntervalMixin:
    def _inference_model(
        self,
        model,
        x_t,
        t,
        cond,
        guidance_strength,
        guidance_interval,
        **kwargs,
    ):
        if guidance_interval[0] <= t <= guidance_interval[1]:
            return super()._inference_model(
                model, x_t, t, cond, guidance_strength=guidance_strength, **kwargs
            )
        return super()._inference_model(model, x_t, t, cond, guidance_strength=1, **kwargs)


class FlowEulerCfgSampler(_ClassifierFreeGuidanceMixin, FlowEulerSampler):
    """Flow Euler with classifier-free guidance."""

    @torch.no_grad()
    def sample(
        self,
        model,
        noise: torch.Tensor,
        cond: Any,
        neg_cond: Any,
        steps: int = 50,
        rescale_t: float = 1.0,
        guidance_strength: float = 3.0,
        verbose: bool = True,
        **kwargs,
    ):
        return super().sample(
            model,
            noise,
            cond,
            steps=steps,
            rescale_t=rescale_t,
            verbose=verbose,
            neg_cond=neg_cond,
            guidance_strength=guidance_strength,
            **kwargs,
        )


class FlowEulerGuidanceIntervalSampler(
    _GuidanceIntervalMixin, _ClassifierFreeGuidanceMixin, FlowEulerSampler
):
    """Flow Euler with CFG + guidance interval."""

    @torch.no_grad()
    def sample(
        self,
        model,
        noise: torch.Tensor,
        cond: Any,
        neg_cond: Any,
        steps: int = 50,
        rescale_t: float = 1.0,
        guidance_strength: float = 3.0,
        guidance_interval: tuple[float, float] = (0.0, 1.0),
        verbose: bool = True,
        **kwargs,
    ):
        return super().sample(
            model,
            noise,
            cond,
            steps=steps,
            rescale_t=rescale_t,
            verbose=verbose,
            neg_cond=neg_cond,
            guidance_strength=guidance_strength,
            guidance_interval=guidance_interval,
            **kwargs,
        )
