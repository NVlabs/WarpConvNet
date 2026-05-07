# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end image→mesh inference using the warpconvnet TRELLIS.2 port.

Loads `microsoft/TRELLIS.2-4B` (and the `microsoft/TRELLIS-image-large` SS
decoder it depends on) from the HF Hub cache, builds the four sub-models
plus a DinoV3 feature extractor, and runs the 512-resolution image→mesh
pipeline. Writes ``.obj`` per-batch.

Usage:
    python examples/inference/trellis2_image_to_mesh.py \\
        --image /path/to/image.png \\
        --out /tmp/trellis2_output.obj \\
        [--seed 42] [--samples 1] [--steps 12]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

_TRELLIS2_PATH = os.environ.get("TRELLIS2_PATH")
if _TRELLIS2_PATH and os.path.isdir(_TRELLIS2_PATH) and _TRELLIS2_PATH not in sys.path:
    sys.path.insert(0, _TRELLIS2_PATH)

from huggingface_hub import hf_hub_download
from PIL import Image
from safetensors.torch import load_file

from warpconvnet.models.trellis2.image_cond import DinoV3FeatureExtractor
from warpconvnet.models.trellis2.pipeline import (
    Trellis2ImageTo3DPipeline,
    pipeline_config_from_json_args,
)
from warpconvnet.models.trellis2.shape_vae import FlexiDualGridVaeDecoder
from warpconvnet.models.trellis2.slat_flow import SLatFlowModel
from warpconvnet.models.trellis2.sparse_structure_flow import SparseStructureFlowModel
from warpconvnet.models.trellis2.sparse_structure_vae import SparseStructureDecoder


def _load(repo: str, name: str, device: str = "cuda"):
    cfg = json.load(open(hf_hub_download(repo, f"{name}.json")))["args"]
    state = load_file(hf_hub_download(repo, f"{name}.safetensors"))
    return cfg, state


def _build_ss_flow(device: str) -> SparseStructureFlowModel:
    cfg, state = _load("microsoft/TRELLIS.2-4B", "ckpts/ss_flow_img_dit_1_3B_64_bf16", device)
    m = SparseStructureFlowModel(**cfg).to(device).eval()
    m.load_state_dict(state, strict=False)  # rope_phases is a recomputed buffer
    return m


def _build_ss_decoder(device: str) -> SparseStructureDecoder:
    cfg, state = _load("microsoft/TRELLIS-image-large", "ckpts/ss_dec_conv3d_16l8_fp16", device)
    m = SparseStructureDecoder(**cfg).to(device).eval()
    m.load_state_dict(state)
    return m


def _build_slat_flow(device: str, hr: bool = False) -> SLatFlowModel:
    name = (
        "ckpts/slat_flow_img2shape_dit_1_3B_1024_bf16"
        if hr
        else "ckpts/slat_flow_img2shape_dit_1_3B_512_bf16"
    )
    cfg, state = _load("microsoft/TRELLIS.2-4B", name, device)
    m = SLatFlowModel(**cfg).to(device).eval()
    m.load_state_dict(state, strict=False)
    return m


def _build_shape_decoder(device: str) -> FlexiDualGridVaeDecoder:
    cfg, state = _load("microsoft/TRELLIS.2-4B", "ckpts/shape_dec_next_dc_f16c32_fp16", device)
    m = FlexiDualGridVaeDecoder(**cfg).to(device).eval()
    m.load_trellis2_state_dict(state, strict=False)
    return m


def _load_pipeline_json() -> dict:
    return json.load(open(hf_hub_download("microsoft/TRELLIS.2-4B", "pipeline.json")))


def _save_obj(path: str, vertices: torch.Tensor, faces: torch.Tensor) -> None:
    """Minimal Wavefront OBJ writer (vertices + triangle faces)."""
    v = vertices.detach().cpu().numpy()
    f = faces.detach().cpu().numpy()
    with open(path, "w") as fh:
        for x, y, z in v:
            fh.write(f"v {x:.6f} {y:.6f} {z:.6f}\n")
        for a, b, c in f + 1:  # OBJ is 1-indexed
            fh.write(f"f {a} {b} {c}\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="path to a PIL-readable image")
    ap.add_argument("--out", required=True, help="output .obj path")
    ap.add_argument("--samples", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--steps", type=int, default=None, help="override both SS and SLAT steps")
    ap.add_argument("--ss_steps", type=int, default=None, help="override sparse-structure steps")
    ap.add_argument("--slat_steps", type=int, default=None, help="override SLAT steps")
    ap.add_argument(
        "--pipeline_type",
        default="512",
        choices=("512", "1024", "1024_cascade", "1536_cascade"),
    )
    args = ap.parse_args()

    device = "cuda"
    print("[load] SS flow / SS decoder / SLAT flow / shape decoder ...")
    t0 = time.time()
    ss_flow = _build_ss_flow(device)
    ss_decoder = _build_ss_decoder(device)
    slat_flow = _build_slat_flow(device, hr=False)
    slat_flow_1024 = _build_slat_flow(device, hr=True) if args.pipeline_type != "512" else None
    shape_decoder = _build_shape_decoder(device)
    print(f"[load] models ready in {time.time()-t0:.1f}s")

    print("[load] DinoV3 image cond ...")
    image_cond = DinoV3FeatureExtractor(image_size=512).cuda()

    # Sampler params from the published pipeline.json.
    p = _load_pipeline_json()["args"]
    cfg = pipeline_config_from_json_args(
        p,
        steps=args.steps,
        ss_steps=args.ss_steps,
        slat_steps=args.slat_steps,
    )
    pipe = Trellis2ImageTo3DPipeline(
        ss_flow=ss_flow,
        ss_decoder=ss_decoder,
        slat_flow_512=slat_flow,
        slat_flow_1024=slat_flow_1024,
        shape_decoder=shape_decoder,
        image_cond=image_cond,
        slat_normalization=p["shape_slat_normalization"],
        config=cfg,
        default_pipeline_type=args.pipeline_type,
    )

    print(
        f"[run]  image={args.image}  seed={args.seed}  "
        f"steps={cfg.ss_steps}/{cfg.slat_steps}  "
        f"type={args.pipeline_type}"
    )
    t0 = time.time()
    image = Image.open(args.image)
    meshes = pipe.run(
        image,
        num_samples=args.samples,
        seed=args.seed,
        pipeline_type=args.pipeline_type,
    )
    print(f"[run]  total {time.time()-t0:.1f}s, {len(meshes)} mesh(es)")

    for i, m in enumerate(meshes):
        out_path = args.out if args.samples == 1 else args.out.replace(".obj", f"_{i}.obj")
        _save_obj(out_path, m.vertices, m.faces)
        print(f"[save] {out_path}: V={m.vertices.shape[0]}  F={m.faces.shape[0]}")


if __name__ == "__main__":
    main()
