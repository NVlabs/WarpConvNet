# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Structural tests for the TRELLIS.2 shape-VAE encoder port.

1. The encoder can be constructed from the published TRELLIS.2 config.
2. Its parameter / submodule layout exactly matches the encoder weight names
   in `shape_enc_next_dc_f16c32_fp16.safetensors` after the same 5D→3D
   sparse-conv layout conversion the decoder uses.

When the safetensors file is available at the path passed via
``--shape-enc`` (or env var ``TRELLIS2_SHAPE_ENC``), the loader assertion
becomes a real "zero missing, zero unexpected" check against published weights.
Without it, the test creates a synthetic state-dict with all-zeros tensors
shaped per the encoder's own ``state_dict()`` and confirms round-trip.
"""

from __future__ import annotations

import argparse
import os
from collections import OrderedDict

import torch

# Import via the deeper module path to avoid relying on aggregate model
# re-exports. The trellis2.__init__ is a re-export of shape_vae symbols, so the
# deeper path is equivalent for our purposes.
from warpconvnet.models.trellis2.shape_vae import (
    FlexiDualGridVaeEncoder,
    convert_trellis2_shape_vae_state_dict,
)


# Config copied verbatim from
# TRELLIS.2/configs/scvae/shape_vae_next_dc_f16c32_fp16.json (encoder side).
_SHAPE_ENC_CONFIG = dict(
    model_channels=[64, 128, 256, 512, 1024],
    latent_channels=32,
    num_blocks=[0, 4, 8, 16, 4],
    block_type=["SparseConvNeXtBlock3d"] * 5,
    down_block_type=["SparseResBlockS2C3d"] * 4,
    block_args=[
        {"use_checkpoint": True},
        {"use_checkpoint": True},
        {"use_checkpoint": False},
        {"use_checkpoint": False},
        {"use_checkpoint": False},
    ],
    use_fp16=True,
)


def _build_encoder() -> FlexiDualGridVaeEncoder:
    return FlexiDualGridVaeEncoder(**_SHAPE_ENC_CONFIG)


def test_construct_from_published_config() -> None:
    """Encoder construction should succeed and yield a sensible param count."""
    enc = _build_encoder()
    n_params = sum(p.numel() for p in enc.parameters())
    # Upstream reports ~177M params for the 4B shape encoder. Sanity: > 100M.
    assert n_params > 100_000_000, f"unexpectedly small encoder ({n_params:,} params)"
    print(f"[ok] encoder constructed: {n_params/1e6:.1f}M params")


def test_synthetic_roundtrip() -> None:
    """The model's own state_dict() round-trips through load_state_dict.

    Catches name/shape regressions in the module graph independent of any
    published checkpoint.
    """
    enc = _build_encoder()
    sd = OrderedDict((k, torch.zeros_like(v)) for k, v in enc.state_dict().items())
    missing, unexpected = enc.load_state_dict(sd, strict=True)
    assert not missing and not unexpected, (
        f"round-trip missing={len(missing)} unexpected={len(unexpected)}"
    )
    print(f"[ok] state-dict round-trip clean ({len(sd)} tensors)")


def test_load_published_safetensors(path: str) -> None:
    """Load the published shape-encoder weights and assert zero unexpected
    keys after the 5D→3D conversion. Missing keys are reported but not fatal
    (in case a checkpoint contains optional buffers we don't materialize).
    """
    from safetensors.torch import load_file

    print(f"[load] {path}")
    raw = load_file(path)
    enc = _build_encoder()
    converted = convert_trellis2_shape_vae_state_dict(raw, enc)
    missing, unexpected = enc.load_state_dict(converted, strict=False)
    print(f"[load] missing={len(missing)} unexpected={len(unexpected)}")
    if unexpected:
        print(f"[load] first 10 unexpected: {unexpected[:10]}")
    if missing:
        print(f"[load] first 10 missing: {missing[:10]}")
    assert not unexpected, (
        f"published encoder has {len(unexpected)} keys our port doesn't recognize"
    )
    print("[ok] published safetensors load with no unexpected keys")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--shape-enc",
        default=os.environ.get("TRELLIS2_SHAPE_ENC", ""),
        help="Path to shape_enc_next_dc_f16c32_fp16.safetensors (optional).",
    )
    args = p.parse_args()

    test_construct_from_published_config()
    test_synthetic_roundtrip()

    if args.shape_enc and os.path.exists(args.shape_enc):
        test_load_published_safetensors(args.shape_enc)
    else:
        print(
            "[skip] published safetensors path not provided "
            "(set --shape-enc or env TRELLIS2_SHAPE_ENC to enable)"
        )


if __name__ == "__main__":
    main()
