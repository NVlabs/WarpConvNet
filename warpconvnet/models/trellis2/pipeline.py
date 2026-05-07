# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TRELLIS.2 image→mesh inference pipeline (no texture flow).

Mirrors `trellis2.pipelines.Trellis2ImageTo3DPipeline.run()`. Supports the
four upstream pipeline types (texture flow stripped):

    512            : SS @ 32³ → SLAT(512) → shape decoder → mesh
    1024           : SS @ 64³ → SLAT(1024) → shape decoder → mesh
    1024_cascade   : SS @ 32³ → SLAT(512) (LR) → upsample-coords → SLAT(1024) (HR)
    1536_cascade   : same as 1024_cascade but HR target = 1536

The pipeline takes pre-loaded module instances; loader helpers in
`examples/inference/trellis2_image_to_mesh.py` build them from the published
HF safetensors.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Tuple

import torch

from warpconvnet.geometry.types.voxels import Voxels

from .image_cond import DinoV3FeatureExtractor
from .mesh_extract import MeshOut, extract_meshes
from .samplers import FlowEulerGuidanceIntervalSampler
from .shape_vae import FlexiDualGridVaeDecoder
from .slat_flow import SLatFlowModel
from .sparse_ops import from_feats_coords
from .sparse_structure_flow import SparseStructureFlowModel
from .sparse_structure_vae import SparseStructureDecoder


__all__ = [
    "PipelineConfig",
    "PipelineType",
    "Trellis2ImageTo3DPipeline",
    "pipeline_config_from_json_args",
]


PipelineType = Literal["512", "1024", "1024_cascade", "1536_cascade"]


# Pipeline-type → (ss_resolution, output_mesh_resolution).
_TYPE_TABLE: dict[str, tuple[int, int]] = {
    "512": (32, 512),
    "1024": (64, 1024),
    "1024_cascade": (32, 1024),
    "1536_cascade": (32, 1536),
}


@dataclass
class PipelineConfig:
    """Sampler kwargs from `pipeline.json`."""

    ss_steps: int = 12
    ss_guidance_strength: float = 7.5
    ss_guidance_rescale: float = 0.7
    ss_guidance_interval: tuple = (0.6, 1.0)
    ss_rescale_t: float = 5.0

    slat_steps: int = 12
    slat_guidance_strength: float = 7.5
    slat_guidance_rescale: float = 0.5
    slat_guidance_interval: tuple = (0.6, 1.0)
    slat_rescale_t: float = 3.0

    fill_holes: bool = True

    # Cascade-mode token budget — shrinks HR resolution by 128 until the
    # token count drops below this threshold (matches upstream default).
    cascade_max_tokens: int = 49152


def pipeline_config_from_json_args(
    args: dict[str, Any],
    *,
    steps: int | None = None,
    ss_steps: int | None = None,
    slat_steps: int | None = None,
    ss_guidance_strength: float | None = None,
    slat_guidance_strength: float | None = None,
    fill_holes: bool | None = None,
) -> PipelineConfig:
    """Build ``PipelineConfig`` from the published TRELLIS.2 ``pipeline.json``.

    ``steps`` is a convenience override for both sparse-structure and SLAT
    samplers; ``ss_steps`` / ``slat_steps`` take precedence when provided.
    """
    ss_params = args["sparse_structure_sampler"]["params"]
    slat_params = args["shape_slat_sampler"]["params"]
    ss_steps = ss_steps if ss_steps is not None else steps
    slat_steps = slat_steps if slat_steps is not None else steps
    cfg = PipelineConfig(
        ss_steps=ss_steps if ss_steps is not None else ss_params["steps"],
        ss_guidance_strength=(
            ss_guidance_strength
            if ss_guidance_strength is not None
            else ss_params["guidance_strength"]
        ),
        ss_guidance_rescale=ss_params["guidance_rescale"],
        ss_guidance_interval=tuple(ss_params["guidance_interval"]),
        ss_rescale_t=ss_params["rescale_t"],
        slat_steps=slat_steps if slat_steps is not None else slat_params["steps"],
        slat_guidance_strength=(
            slat_guidance_strength
            if slat_guidance_strength is not None
            else slat_params["guidance_strength"]
        ),
        slat_guidance_rescale=slat_params["guidance_rescale"],
        slat_guidance_interval=tuple(slat_params["guidance_interval"]),
        slat_rescale_t=slat_params["rescale_t"],
    )
    if fill_holes is not None:
        cfg.fill_holes = fill_holes
    return cfg


class Trellis2ImageTo3DPipeline:
    """Image→mesh inference orchestrator with multi-resolution support."""

    def __init__(
        self,
        ss_flow: SparseStructureFlowModel,
        ss_decoder: SparseStructureDecoder,
        slat_flow_512: SLatFlowModel,
        shape_decoder: FlexiDualGridVaeDecoder,
        image_cond: DinoV3FeatureExtractor,
        slat_normalization: dict[str, list[float]],
        slat_flow_1024: SLatFlowModel | None = None,
        ss_sampler: FlowEulerGuidanceIntervalSampler | None = None,
        slat_sampler: FlowEulerGuidanceIntervalSampler | None = None,
        config: PipelineConfig | None = None,
        default_pipeline_type: PipelineType = "512",
    ):
        self.ss_flow = ss_flow
        self.ss_decoder = ss_decoder
        self.slat_flow_512 = slat_flow_512
        self.slat_flow_1024 = slat_flow_1024
        self.shape_decoder = shape_decoder
        self.image_cond = image_cond
        self.slat_normalization = slat_normalization
        self.ss_sampler = ss_sampler or FlowEulerGuidanceIntervalSampler(sigma_min=1e-5)
        self.slat_sampler = slat_sampler or FlowEulerGuidanceIntervalSampler(sigma_min=1e-5)
        self.config = config or PipelineConfig()
        self.default_pipeline_type = default_pipeline_type
        self._device = next(ss_flow.parameters()).device

    # -- conditioning ----------------------------------------------------------
    @torch.no_grad()
    def get_cond(self, image, image_size: int = 512) -> dict[str, torch.Tensor]:
        self.image_cond.image_size = image_size
        cond = self.image_cond([image] if not isinstance(image, list) else image)
        return {"cond": cond, "neg_cond": torch.zeros_like(cond)}

    # -- stage 1: sparse structure --------------------------------------------
    @torch.no_grad()
    def sample_sparse_structure(
        self,
        cond: dict[str, torch.Tensor],
        ss_resolution: int,
        num_samples: int = 1,
        verbose: bool = True,
    ) -> torch.Tensor:
        """Returns ``(N, 4)`` int batch-indexed coords at the requested SS res."""
        cfg = self.config
        flow = self.ss_flow
        R = flow.resolution
        noise = torch.randn(num_samples, flow.in_channels, R, R, R, device=self._device)
        z_s = self.ss_sampler.sample(
            flow,
            noise,
            cond=cond["cond"],
            neg_cond=cond["neg_cond"],
            steps=cfg.ss_steps,
            rescale_t=cfg.ss_rescale_t,
            guidance_strength=cfg.ss_guidance_strength,
            guidance_rescale=cfg.ss_guidance_rescale,
            guidance_interval=cfg.ss_guidance_interval,
            verbose=verbose,
            tqdm_desc="Sampling sparse structure",
        )["samples"]
        decoded = self.ss_decoder(z_s) > 0  # (B, 1, 64, 64, 64)
        if ss_resolution != decoded.shape[2]:
            ratio = decoded.shape[2] // ss_resolution
            decoded = torch.nn.functional.max_pool3d(decoded.float(), ratio, ratio, 0) > 0.5
        return torch.argwhere(decoded)[:, [0, 2, 3, 4]].int()

    # -- stage 2: shape SLAT (single-stage) -----------------------------------
    @torch.no_grad()
    def _sample_slat(
        self,
        cond: dict[str, torch.Tensor],
        flow_model: SLatFlowModel,
        coords: torch.Tensor,
        verbose: bool = True,
        tqdm_desc: str = "Sampling shape SLat",
    ) -> Voxels:
        cfg = self.config
        feats0 = torch.randn(coords.shape[0], flow_model.in_channels, device=self._device)
        noise = from_feats_coords(feats0, coords)
        slat = self.slat_sampler.sample(
            flow_model,
            noise,
            cond=cond["cond"],
            neg_cond=cond["neg_cond"],
            steps=cfg.slat_steps,
            rescale_t=cfg.slat_rescale_t,
            guidance_strength=cfg.slat_guidance_strength,
            guidance_rescale=cfg.slat_guidance_rescale,
            guidance_interval=cfg.slat_guidance_interval,
            verbose=verbose,
            tqdm_desc=tqdm_desc,
        )["samples"]
        std = torch.tensor(self.slat_normalization["std"], device=slat.device)
        mean = torch.tensor(self.slat_normalization["mean"], device=slat.device)
        return slat.replace_features(slat.feats * std + mean)

    sample_shape_slat = _sample_slat  # backwards-compatible alias

    # -- stage 2 (cascade): LR SLAT → upsample-coords → HR SLAT ---------------
    @torch.no_grad()
    def sample_shape_slat_cascade(
        self,
        lr_cond: dict[str, torch.Tensor],
        hr_cond: dict[str, torch.Tensor],
        coords: torch.Tensor,
        lr_resolution: int,
        hr_resolution: int,
        verbose: bool = True,
    ) -> tuple[Voxels, int]:
        """LR(=512) → upsample-coords → HR(=1024 or 1536) cascade.

        Returns ``(hr_slat, achieved_hr_resolution)``. The achieved resolution
        may be lower than requested if the per-token budget is exceeded.
        """
        assert self.slat_flow_1024 is not None, "1024 cascade requires slat_flow_1024"
        cfg = self.config

        # 1. LR SLAT @ 512.
        lr_slat = self._sample_slat(
            lr_cond,
            self.slat_flow_512,
            coords,
            verbose=verbose,
            tqdm_desc="Sampling LR shape SLat (512)",
        )

        # 2. Coordinate upsample via shape decoder's first 4 C2S stages.
        hr_coords = self.shape_decoder.upsample(lr_slat, upsample_times=4)

        # 3. Token-budget loop: shrink HR target by 128 until quantised
        #    coord count fits the budget (matches upstream).
        achieved_hr = hr_resolution
        coords_quant = coords
        while True:
            quant = torch.cat(
                [
                    hr_coords[:, :1],
                    ((hr_coords[:, 1:] + 0.5) / lr_resolution * (achieved_hr // 16)).int(),
                ],
                dim=1,
            )
            coords_quant = quant.unique(dim=0)
            if coords_quant.shape[0] < cfg.cascade_max_tokens or achieved_hr == 1024:
                if achieved_hr != hr_resolution:
                    print(f"[cascade] token budget hit; HR reduced to {achieved_hr}")
                break
            achieved_hr -= 128

        # 4. HR SLAT.
        hr_slat = self._sample_slat(
            hr_cond,
            self.slat_flow_1024,
            coords_quant,
            verbose=verbose,
            tqdm_desc=f"Sampling HR shape SLat ({achieved_hr})",
        )
        return hr_slat, achieved_hr

    # -- stage 3: decode + extract --------------------------------------------
    @torch.no_grad()
    def decode_to_mesh(self, slat: Voxels, output_resolution: int) -> list[MeshOut]:
        self.shape_decoder.set_resolution(output_resolution)
        vertices, intersected, quad_lerp = self.shape_decoder(slat)
        return extract_meshes(
            vertices=vertices,
            intersected=intersected,
            quad_lerp=quad_lerp,
            grid_size=output_resolution,
            fill_holes=self.config.fill_holes,
        )

    # -- end-to-end -----------------------------------------------------------
    @torch.no_grad()
    def run(
        self,
        image,
        num_samples: int = 1,
        seed: int = 42,
        pipeline_type: PipelineType | None = None,
        verbose: bool = True,
    ) -> list[MeshOut]:
        ptype = pipeline_type or self.default_pipeline_type
        if ptype not in _TYPE_TABLE:
            raise ValueError(f"unknown pipeline_type: {ptype}")
        ss_res, mesh_res = _TYPE_TABLE[ptype]

        # Conditioning at 512 always; the 1024 / cascade modes also need a 1024-cond.
        cond_512 = self.get_cond(image, image_size=512)
        cond_1024 = self.get_cond(image, image_size=1024) if ptype != "512" else None

        torch.manual_seed(seed)
        coords = self.sample_sparse_structure(
            cond_512, ss_resolution=ss_res, num_samples=num_samples, verbose=verbose
        )

        if ptype == "512":
            slat = self._sample_slat(cond_512, self.slat_flow_512, coords, verbose=verbose)
            mesh_resolution = mesh_res
        elif ptype == "1024":
            assert self.slat_flow_1024 is not None
            slat = self._sample_slat(cond_1024, self.slat_flow_1024, coords, verbose=verbose)
            mesh_resolution = mesh_res
        else:  # cascade
            slat, achieved_hr = self.sample_shape_slat_cascade(
                lr_cond=cond_512,
                hr_cond=cond_1024,
                coords=coords,
                lr_resolution=512,
                hr_resolution=mesh_res,
                verbose=verbose,
            )
            mesh_resolution = achieved_hr

        return self.decode_to_mesh(slat, output_resolution=mesh_resolution)
