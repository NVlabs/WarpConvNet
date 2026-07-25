# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""End-to-end training-step timing for a MinkUNet-shaped sparse encoder.

python scripts/bench_unet_gb300.py                 # real device arch
python scripts/bench_unet_gb300.py --spoof-arch 100  # mask_gemm gate open
"""

import argparse
import sys


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=250_000)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--extent", type=int, default=512)
    p.add_argument("--dtype", default="float16")
    p.add_argument("--spoof-arch", type=int, default=0)
    p.add_argument("--iters", type=int, default=20)
    args = p.parse_args()

    if args.spoof_arch:
        import warpconvnet.nn.functional.sparse_conv.detail.tile_metadata as tm

        tm._DEVICE_ARCH = args.spoof_arch

    import torch
    import torch.nn as nn
    import warpconvnet  # noqa: F401
    from warpconvnet.geometry.types.voxels import Voxels
    from warpconvnet.nn.functional.sparse_conv.detail import tile_metadata as tile_meta
    from warpconvnet.nn.modules.sparse_conv import SparseConv3d

    dev, dtype = "cuda", getattr(torch, args.dtype)
    torch.manual_seed(0)
    coords, feats = [], []
    for _ in range(args.batch):
        c = torch.randint(0, args.extent, (int(args.n * 1.3), 3), device=dev, dtype=torch.int32)
        c = torch.unique(c, dim=0)[: args.n]
        coords.append(c)
        feats.append(torch.randn(c.shape[0], 32, device=dev, dtype=dtype))
    vox = Voxels(coords, feats).unique()

    # MinkUNet-ish encoder: 4 downsampling stages, 2 residual convs each.
    class Stage(nn.Module):
        def __init__(self, cin, cout, stride):
            super().__init__()
            self.down = SparseConv3d(cin, cout, kernel_size=3, stride=stride, bias=False)
            self.a = SparseConv3d(cout, cout, kernel_size=3, bias=False)
            self.b = SparseConv3d(cout, cout, kernel_size=3, bias=False)

        def forward(self, x):
            x = self.down(x)
            y = self.a(x)
            y = y.replace(batched_features=torch.relu(y.feature_tensor))
            y = self.b(y)
            return y.replace(batched_features=torch.relu(y.feature_tensor + x.feature_tensor))

    net = (
        nn.Sequential(Stage(32, 64, 1), Stage(64, 128, 2), Stage(128, 256, 2), Stage(256, 256, 2))
        .to(dev)
        .to(dtype)
    )
    opt = torch.optim.SGD(net.parameters(), lr=1e-3)

    def step():
        opt.zero_grad(set_to_none=True)
        v = vox.replace(batched_features=vox.feature_tensor.detach().requires_grad_(True))
        out = net(v)
        out.feature_tensor.float().pow(2).mean().backward()
        opt.step()

    for _ in range(8):  # warmup + autotune
        step()
    torch.cuda.synchronize()
    s, e = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    s.record()
    for _ in range(args.iters):
        step()
    e.record()
    torch.cuda.synchronize()
    ms = s.elapsed_time(e) / args.iters
    major, minor = torch.cuda.get_device_capability()
    n_tiles = len(tile_meta._get_tiles("forward", filter_arch=True))
    label = (
        f"sm_{args.spoof_arch} (spoofed)"
        if args.spoof_arch
        else f"sm_{major}{minor} ({n_tiles} fwd tiles)"
    )
    print(
        f"{label:<26} training step: {ms:8.2f} ms   "
        f"({vox.feature_tensor.shape[0]} input voxels, "
        f"peak {torch.cuda.max_memory_allocated()/2**30:.1f} GiB)",
        file=sys.stderr,
    )
    print(f"MS\t{ms:.3f}")


if __name__ == "__main__":
    main()
