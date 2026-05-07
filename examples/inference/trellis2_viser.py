# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Interactive viser-based viewer for the warpconvnet TRELLIS.2 port.

Loads the pipeline once, lets the user browse a list of input images, run
inference (with adjustable seed / sampler-step count), and view the resulting
mesh in 3D.

Usage:
    python examples/inference/trellis2_viser.py --images img1.png img2.webp ...
    python examples/inference/trellis2_viser.py --image_dir /path/to/folder
    [--port 8080]

Open the printed URL in a browser.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

_TRELLIS2_PATH = os.environ.get("TRELLIS2_PATH")
if _TRELLIS2_PATH and os.path.isdir(_TRELLIS2_PATH) and _TRELLIS2_PATH not in sys.path:
    sys.path.insert(0, _TRELLIS2_PATH)

import viser
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


# -----------------------------------------------------------------------------
# Pipeline loading (mirrors examples/inference/trellis2_image_to_mesh.py)
# -----------------------------------------------------------------------------
def _load_safetensors(repo: str, name: str):
    cfg = json.load(open(hf_hub_download(repo, f"{name}.json")))["args"]
    state = load_file(hf_hub_download(repo, f"{name}.safetensors"))
    return cfg, state


def _build_pipeline(device: str = "cuda", with_1024: bool = True) -> Trellis2ImageTo3DPipeline:
    print(f"[load] models from HF cache " f"({'5' if with_1024 else '4'} sub-models + DinoV3) ...")
    t0 = time.time()
    c, s = _load_safetensors("microsoft/TRELLIS.2-4B", "ckpts/ss_flow_img_dit_1_3B_64_bf16")
    ssf = SparseStructureFlowModel(**c).to(device).eval()
    ssf.load_state_dict(s, strict=False)
    c, s = _load_safetensors("microsoft/TRELLIS-image-large", "ckpts/ss_dec_conv3d_16l8_fp16")
    ssd = SparseStructureDecoder(**c).to(device).eval()
    ssd.load_state_dict(s)
    c, s = _load_safetensors(
        "microsoft/TRELLIS.2-4B", "ckpts/slat_flow_img2shape_dit_1_3B_512_bf16"
    )
    slat_512 = SLatFlowModel(**c).to(device).eval()
    slat_512.load_state_dict(s, strict=False)

    slat_1024 = None
    if with_1024:
        c, s = _load_safetensors(
            "microsoft/TRELLIS.2-4B", "ckpts/slat_flow_img2shape_dit_1_3B_1024_bf16"
        )
        slat_1024 = SLatFlowModel(**c).to(device).eval()
        slat_1024.load_state_dict(s, strict=False)

    c, s = _load_safetensors("microsoft/TRELLIS.2-4B", "ckpts/shape_dec_next_dc_f16c32_fp16")
    dec = FlexiDualGridVaeDecoder(**c).to(device).eval()
    dec.load_trellis2_state_dict(s, strict=False)
    print(f"[load] core models in {time.time()-t0:.1f}s; loading DinoV3 ...")
    t0 = time.time()
    image_cond = DinoV3FeatureExtractor(image_size=512).cuda()
    print(f"[load] DinoV3 in {time.time()-t0:.1f}s")

    p = json.load(open(hf_hub_download("microsoft/TRELLIS.2-4B", "pipeline.json")))["args"]
    cfg = pipeline_config_from_json_args(p)
    return Trellis2ImageTo3DPipeline(
        ss_flow=ssf,
        ss_decoder=ssd,
        slat_flow_512=slat_512,
        slat_flow_1024=slat_1024,
        shape_decoder=dec,
        image_cond=image_cond,
        slat_normalization=p["shape_slat_normalization"],
        config=cfg,
    )


# -----------------------------------------------------------------------------
# Image discovery
# -----------------------------------------------------------------------------
_IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def _gather_images(paths: list[str], image_dir: str | None) -> list[Path]:
    out: list[Path] = []
    for p in paths or []:
        out.append(Path(p))
    if image_dir:
        for f in sorted(Path(image_dir).iterdir()):
            if f.suffix.lower() in _IMG_EXTS:
                out.append(f)
    if not out:
        raise SystemExit("No input images. Use --images or --image_dir.")
    return out


def _load_thumb(path: Path, size: int = 256) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    img.thumbnail((size, size))
    return np.array(img)


# -----------------------------------------------------------------------------
# viser app
# -----------------------------------------------------------------------------
class ViserApp:
    def __init__(
        self,
        pipe: Trellis2ImageTo3DPipeline,
        images: list[Path],
        port: int,
    ):
        self.pipe = pipe
        self.images = images
        self.server = viser.ViserServer(port=port)
        self._mesh_handle = None
        self._cache: dict[tuple, tuple] = {}  # (path, seed, steps) -> (verts, faces)
        self._build_gui()

    def _build_gui(self) -> None:
        self.server.scene.set_up_direction("+y")
        gui = self.server.gui

        with gui.add_folder("Input"):
            self._image_dropdown = gui.add_dropdown(
                "Image",
                options=tuple(p.name for p in self.images),
                initial_value=self.images[0].name,
            )
            self._image_panel = gui.add_image(
                _load_thumb(self.images[0]),
                label="Preview",
            )

            @self._image_dropdown.on_update
            def _(_):
                p = self._path_for_name(self._image_dropdown.value)
                self._image_panel.image = _load_thumb(p)
                self._status.value = "image switched (no inference yet)"

        with gui.add_folder("Sampling"):
            self._pipeline_type = gui.add_dropdown(
                "pipeline type",
                options=("512", "1024", "1024_cascade", "1536_cascade"),
                initial_value="512",
            )
            self._seed = gui.add_number("seed", initial_value=42, step=1)
            self._steps_ss = gui.add_slider(
                "SS steps",
                min=4,
                max=50,
                step=1,
                initial_value=self.pipe.config.ss_steps,
            )
            self._steps_slat = gui.add_slider(
                "SLAT steps",
                min=4,
                max=50,
                step=1,
                initial_value=self.pipe.config.slat_steps,
            )
            self._guidance_ss = gui.add_slider(
                "SS guidance",
                min=1.0,
                max=15.0,
                step=0.5,
                initial_value=self.pipe.config.ss_guidance_strength,
            )
            self._guidance_slat = gui.add_slider(
                "SLAT guidance",
                min=1.0,
                max=15.0,
                step=0.5,
                initial_value=self.pipe.config.slat_guidance_strength,
            )
            self._fill_holes = gui.add_checkbox("fill holes", initial_value=True)

        self._status = gui.add_markdown("**Idle**")
        self._stats = gui.add_markdown("")
        self._run_btn = gui.add_button("Run inference")
        self._clear_btn = gui.add_button("Clear mesh")

        @self._run_btn.on_click
        def _(_):
            self._run_inference()

        @self._clear_btn.on_click
        def _(_):
            if self._mesh_handle is not None:
                self._mesh_handle.remove()
                self._mesh_handle = None
            self._status.content = "**Cleared.**"

    def _path_for_name(self, name: str) -> Path:
        for p in self.images:
            if p.name == name:
                return p
        raise KeyError(name)

    def _run_inference(self) -> None:
        try:
            self._run_inference_inner()
        except Exception as e:  # noqa: BLE001
            import traceback

            tb = traceback.format_exc()
            print(tb, flush=True)
            self._status.content = f"**ERROR**: `{type(e).__name__}: {e}`"

    def _run_inference_inner(self) -> None:
        path = self._path_for_name(self._image_dropdown.value)
        seed = int(self._seed.value)
        steps_ss = int(self._steps_ss.value)
        steps_slat = int(self._steps_slat.value)
        gs = float(self._guidance_ss.value)
        gl = float(self._guidance_slat.value)
        fh = bool(self._fill_holes.value)
        ptype = str(self._pipeline_type.value)

        if ptype != "512" and self.pipe.slat_flow_1024 is None:
            self._status.content = (
                f"**ERROR**: pipeline_type={ptype} requires the 1024 SLAT flow "
                "(start the viewer without --no_1024)."
            )
            return

        cfg = self.pipe.config
        cfg.ss_steps = steps_ss
        cfg.slat_steps = steps_slat
        cfg.ss_guidance_strength = gs
        cfg.slat_guidance_strength = gl
        cfg.fill_holes = fh

        key = (str(path), seed, steps_ss, steps_slat, gs, gl, fh, ptype)
        cached = self._cache.get(key)
        if cached is not None:
            verts, faces = cached
            self._draw_mesh(verts, faces)
            self._status.content = f"**Cached** result for {path.name} ({ptype})"
            return

        self._status.content = (
            f"**Running** {path.name} ({ptype}, seed={seed}, "
            f"steps={steps_ss}/{steps_slat}) ..."
        )
        t0 = time.time()
        img = Image.open(path)
        meshes = self.pipe.run(img, num_samples=1, seed=seed, pipeline_type=ptype, verbose=False)
        m = meshes[0]
        verts = m.vertices.detach().cpu().numpy()
        faces = m.faces.detach().cpu().numpy()
        elapsed = time.time() - t0
        self._cache[key] = (verts, faces)

        self._draw_mesh(verts, faces)
        self._status.content = (
            f"**Done** {path.name} in {elapsed:.1f}s — "
            f"V={verts.shape[0]:,}  F={faces.shape[0]:,}"
        )
        self._stats.content = (
            f"aabb min `{verts.min(0).round(3).tolist()}` "
            f"max `{verts.max(0).round(3).tolist()}`"
        )

    def _draw_mesh(self, verts: np.ndarray, faces: np.ndarray) -> None:
        if self._mesh_handle is not None:
            self._mesh_handle.remove()
        self._mesh_handle = self.server.scene.add_mesh_simple(
            "/mesh",
            vertices=verts,
            faces=faces,
            color=(180, 200, 240),
            flat_shading=False,
            side="double",
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", nargs="*", default=[])
    ap.add_argument("--image_dir", type=str, default=None)
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument(
        "--no_1024",
        action="store_true",
        help="skip loading the 1024 SLAT flow (saves ~2.5GB; disables 1024 / cascade modes)",
    )
    args = ap.parse_args()

    images = _gather_images(args.images, args.image_dir)
    print(f"[init] {len(images)} input image(s).")

    pipe = _build_pipeline(with_1024=not args.no_1024)
    print(f"[init] starting viser on port {args.port} ...")
    app = ViserApp(pipe, images, args.port)
    print(f"[init] open http://localhost:{args.port}")
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\n[exit]")


if __name__ == "__main__":
    main()
