# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TRELLIS.2 stage-1 inference (image → 64³ occupancy → sparse coords).

Loads the published `microsoft/TRELLIS.2-4B` SS flow checkpoint and the
`microsoft/TRELLIS-image-large` SS decoder, runs a fixed-seed sampling
trajectory, and prints the resulting sparse coordinate stats. Image
conditioning is replaced by a zero tensor for an unconditional smoke test
unless ``--image`` is supplied.

This is the gating verification for Phases 1–3 + sampler, on real weights.
Sparse stages (SLat flow, FlexiDualGrid VAE, mesh extraction) are not
exercised here.

Usage:
    python examples/inference/trellis2_stage1.py [--image PATH] \\
        [--steps 12] [--guidance 7.5] [--seed 42]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

# Path to upstream is optional — only needed for DinoV3 wrapper compatibility.
_TRELLIS2_PATH = os.environ.get("TRELLIS2_PATH")
if _TRELLIS2_PATH and os.path.isdir(_TRELLIS2_PATH) and _TRELLIS2_PATH not in sys.path:
    sys.path.insert(0, _TRELLIS2_PATH)

from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from warpconvnet.models.trellis2.samplers import FlowEulerGuidanceIntervalSampler
from warpconvnet.models.trellis2.sparse_structure_flow import SparseStructureFlowModel
from warpconvnet.models.trellis2.sparse_structure_vae import SparseStructureDecoder


def _load_safetensors_into(model: torch.nn.Module, path: str) -> None:
    state = load_file(path)
    missing, unexpected = model.load_state_dict(state, strict=False)
    # rope_phases is recomputed at construction (complex64); ignore.
    missing = [k for k in missing if k != "rope_phases"]
    if missing or unexpected:
        raise RuntimeError(
            f"Unexpected state_dict mismatch: missing={missing}, unexpected={unexpected}"
        )


def _load_ss_flow(device: str = "cuda") -> SparseStructureFlowModel:
    cfg = hf_hub_download("microsoft/TRELLIS.2-4B", "ckpts/ss_flow_img_dit_1_3B_64_bf16.json")
    sft = hf_hub_download(
        "microsoft/TRELLIS.2-4B", "ckpts/ss_flow_img_dit_1_3B_64_bf16.safetensors"
    )
    args = json.load(open(cfg))["args"]
    m = SparseStructureFlowModel(**args).to(device).eval()
    _load_safetensors_into(m, sft)
    return m


def _load_ss_decoder(device: str = "cuda") -> SparseStructureDecoder:
    cfg = hf_hub_download("microsoft/TRELLIS-image-large", "ckpts/ss_dec_conv3d_16l8_fp16.json")
    sft = hf_hub_download(
        "microsoft/TRELLIS-image-large", "ckpts/ss_dec_conv3d_16l8_fp16.safetensors"
    )
    args = json.load(open(cfg))["args"]
    m = SparseStructureDecoder(**args).to(device).eval()
    _load_safetensors_into(m, sft)
    return m


def _make_cond(image_path: str | None, image_size: int = 512) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (cond, neg_cond), each ``(1, N_patches, 1024)``.

    With ``image_path=None`` returns zeros (unconditional smoke test).
    """
    n_patches = (image_size // 16) ** 2 + 5  # rough size for DinoV3-ViT-L/16
    if image_path is None:
        z = torch.zeros(1, n_patches, 1024, device="cuda")
        return z, z
    from warpconvnet.models.trellis2.image_cond import DinoV3FeatureExtractor
    from PIL import Image

    extractor = DinoV3FeatureExtractor(image_size=image_size).cuda()
    img = Image.open(image_path)
    cond = extractor([img])
    neg_cond = torch.zeros_like(cond)
    return cond, neg_cond


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--image",
        type=str,
        default=None,
        help="path to a PIL-readable image; if omitted, runs unconditional",
    )
    p.add_argument("--steps", type=int, default=12)
    p.add_argument("--guidance", type=float, default=7.5)
    p.add_argument("--guidance_rescale", type=float, default=0.7)
    p.add_argument("--guidance_interval", type=float, nargs=2, default=[0.6, 1.0])
    p.add_argument("--rescale_t", type=float, default=5.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--threshold", type=float, default=0.0)
    args = p.parse_args()

    print(f"[load] SS flow + SS decoder from HF cache...")
    flow = _load_ss_flow()
    decoder = _load_ss_decoder()
    n_flow = sum(p.numel() for p in flow.parameters())
    n_dec = sum(p.numel() for p in decoder.parameters())
    print(f"[load] SS flow: {n_flow/1e6:.1f}M params  SS decoder: {n_dec/1e6:.1f}M params")

    print(f"[cond] image={args.image or '<unconditional zeros>'}")
    cond, neg_cond = _make_cond(args.image)

    R = flow.resolution
    C = flow.in_channels
    sampler = FlowEulerGuidanceIntervalSampler(sigma_min=1e-5)
    torch.manual_seed(args.seed)
    noise = torch.randn(1, C, R, R, R, device="cuda")

    print(f"[flow] steps={args.steps}  guidance={args.guidance}  rescale_t={args.rescale_t}")
    t0 = time.time()
    out = sampler.sample(
        flow,
        noise,
        cond=cond,
        neg_cond=neg_cond,
        steps=args.steps,
        rescale_t=args.rescale_t,
        guidance_strength=args.guidance,
        guidance_rescale=args.guidance_rescale,
        guidance_interval=tuple(args.guidance_interval),
        verbose=True,
    )
    z = out["samples"]
    print(f"[flow] sampled in {time.time()-t0:.1f}s, latent shape {tuple(z.shape)}")

    print(f"[decode] 16³×{C} latent → 64³ occupancy logits ...")
    t0 = time.time()
    with torch.no_grad():
        logits = decoder(z)
    print(f"[decode] {time.time()-t0:.1f}s, logits shape {tuple(logits.shape)}")

    occ = logits > args.threshold
    n_voxels = int(occ.sum().item())
    print(
        f"[stats] occupied voxels @ threshold={args.threshold}: {n_voxels} "
        f"({100*n_voxels/occ.numel():.2f}% of {occ.numel()})"
    )
    if n_voxels > 0:
        coords = occ[0, 0].nonzero()
        print(
            f"[stats] coord aabb: min={tuple(coords.min(0).values.tolist())} "
            f"max={tuple(coords.max(0).values.tolist())}"
        )
        print(
            f"[stats] logits stats: min={logits.min().item():.3f} "
            f"mean={logits.mean().item():.3f} max={logits.max().item():.3f}"
        )
    else:
        print("[stats] no voxels above threshold (expected for unconditional zeros).")


if __name__ == "__main__":
    main()
