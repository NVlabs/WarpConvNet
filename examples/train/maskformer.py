# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Minimal MaskFormer training loop for ScanNet instance segmentation.
#
# Works with either label set (scannet20 or scannet200) — the choice only
# affects the size of the per-mask classification head; the instance-mask
# task itself is identical.
#
# Usage:
#   python examples/train/maskformer.py paths.data_dir=/path/to/scannet_preprocessed
#
# Backbone selection. The default is MinkUNet18; any
# `warpconvnet.models.*` class with the same Voxels->Voxels (or Points->Points)
# interface drops in via the `model.backbone._target_` Hydra override. For
# example, to swap in SpaCeFormer:
#
#   python examples/train/maskformer.py \
#       model.backbone._target_=warpconvnet.models.SpaCeFormer \
#       model.backbone.in_channels=3 \
#       model.backbone.out_channels=96 \
#       +model.backbone.enc_attn_types=ssccc \
#       +model.backbone.dec_attn_types=ssca \
#       +model.backbone.use_rope=true
#
# The data layout follows the Mask3D ScanNet preprocessing:
#   root/{train,val}/sceneXXXX_YY/{coord,color,normal,segmentXX,instance}.npy
# See `warpconvnet.dataset.scannet.ScanNetInstanceDataset` for details.

from typing import Dict, List, Tuple

try:
    import hydra
    from hydra.core.config_store import ConfigStore
    from omegaconf import DictConfig, OmegaConf
except ImportError:
    print("Hydra and OmegaConf not installed, pip install hydra-core omegaconf")
    exit(1)

import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import warp as wp
from scipy.optimize import linear_sum_assignment
from torch import Tensor
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from warpconvnet.dataset.scannet import ScanNetInstanceDataset
from warpconvnet.geometry.types.points import Points
from warpconvnet.models import MaskFormer  # backbone instantiated via Hydra
from warpconvnet.nn.modules.sparse_pool import PointToVoxel

CONFIG_YAML = """
paths:
  data_dir: ~/datasets/scannet_hf
  output_dir: ./results/maskformer
  ckpt_path: null

train:
  batch_size: 2
  lr: 0.001
  epochs: 100
  step_size: 25
  gamma: 0.5
  num_workers: 4
  log_every: 20

data:
  label_set: scannet20       # scannet20 (20 classes, faster) or scannet200 (198 fine-grained)
  voxel_size: 0.01
  ignore_index: -1
  # Train-time transform applied per scene. Override the whole node with
  # `data.train_transform=null` to disable augs, or replace with a custom
  # Compose. See warpconvnet.dataset.transforms for available ops.
  train_transform:
    _target_: warpconvnet.dataset.transforms.default_train_augmentations
    crop_size: 6.0            # metric box crop side length (m); null disables cropping
    min_points: 5000          # skip the crop if it would leave fewer than this

model:
  hidden_dim: 96
  num_queries: 100
  num_heads: 8
  num_decoders: 6
  dim_feedforward: 256
  dropout: 0.1
  backbone_voxel_size: 0.02   # null => skip PointToVoxel (backbone takes Points directly)
  backbone:
    _target_: warpconvnet.models.MinkUNet18
    in_channels: 3
    out_channels: 96          # must equal model.hidden_dim above
    # Alternative: SpaCeFormer.
    # _target_: warpconvnet.models.SpaCeFormer
    # in_channels: 3
    # out_channels: 96
    # enc_attn_types: ssccc   # space at shallow, curve deeper (rule of thumb)
    # dec_attn_types: ssca    # 'a' = full-sequence attention at the bottleneck
    # use_rope: true

loss:
  cls_weight: 1.0
  bce_weight: 5.0
  dice_weight: 5.0
  no_object_weight: 0.1       # down-weight bg class in CE

viz:
  enabled: false              # set true to launch live viser viewer
  port: 8080
  interval_seconds: 10.0
  voxel_size: 0.02            # display voxel size (independent of training voxel)
  score_thresh: 0.5           # min predicted-class probability to keep a query
  mask_thresh: 0.0            # min mask logit to assign a point to a query

device: cuda
seed: 42
"""


def collate_fn(batch: List[Dict]) -> List[Dict]:
    return batch


def build_pc(samples: List[Dict], device: str) -> Points:
    return Points.from_list_of_coordinates(
        [torch.from_numpy(s["coords"]).float() for s in samples],
        features=[torch.from_numpy(s["colors"]).float() / 255.0 for s in samples],
    ).to(device)


def build_targets(samples: List[Dict], num_classes: int, ignore_index: int = -1):
    """Per-scene list of dicts with `labels (M,)` and `masks (M, N)` for M
    ground-truth instances. Skips instance id == ignore_index."""
    targets = []
    for s in samples:
        instance = torch.from_numpy(s["instance"]).long()
        segment = torch.from_numpy(s["segment"]).long()
        valid_inst = torch.unique(instance)
        valid_inst = valid_inst[valid_inst != ignore_index]
        if valid_inst.numel() == 0:
            targets.append(
                {
                    "labels": torch.zeros(0, dtype=torch.long),
                    "masks": torch.zeros(0, instance.shape[0], dtype=torch.bool),
                }
            )
            continue
        masks = []
        labels = []
        for inst_id in valid_inst:
            m = instance == inst_id
            sem = segment[m]
            sem = sem[sem != ignore_index]
            if sem.numel() == 0:
                continue
            cls = int(sem.mode().values.item())
            if cls < 0 or cls >= num_classes:
                continue
            masks.append(m)
            labels.append(cls)
        if len(masks) == 0:
            targets.append(
                {
                    "labels": torch.zeros(0, dtype=torch.long),
                    "masks": torch.zeros(0, instance.shape[0], dtype=torch.bool),
                }
            )
        else:
            targets.append(
                {
                    "labels": torch.tensor(labels, dtype=torch.long),
                    "masks": torch.stack(masks, dim=0),
                }
            )
    return targets


def dice_loss(pred_logits: Tensor, target: Tensor, eps: float = 1.0) -> Tensor:
    pred = pred_logits.sigmoid()
    intersection = (pred * target).sum(-1)
    denom = pred.sum(-1) + target.sum(-1)
    return 1.0 - (2.0 * intersection + eps) / (denom + eps)


@torch.no_grad()
def hungarian_match(
    cls_logits: Tensor,  # (Q, C+1)
    mask_logits: Tensor,  # (Q, N)
    target_labels: Tensor,  # (M,)
    target_masks: Tensor,  # (M, N)
    cls_weight: float,
    bce_weight: float,
    dice_weight: float,
):
    """Returns (q_idx, t_idx) tensors."""
    Q = cls_logits.shape[0]
    M = target_labels.shape[0]
    if M == 0:
        return (torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long))

    # Cost: -cls_prob[t] for each (q, t) + mask BCE + mask dice
    cls_prob = cls_logits.softmax(-1)  # (Q, C+1)
    cls_cost = -cls_prob[:, target_labels]  # (Q, M)

    pred = mask_logits.float()  # (Q, N)
    tgt = target_masks.float()  # (M, N)
    bce_cost = F.binary_cross_entropy_with_logits(
        pred[:, None, :].expand(-1, M, -1),
        tgt[None, :, :].expand(Q, -1, -1),
        reduction="none",
    ).mean(
        -1
    )  # (Q, M)
    pred_sig = pred.sigmoid()
    intersection = pred_sig @ tgt.t()  # (Q, M)
    denom = pred_sig.sum(-1, keepdim=True) + tgt.sum(-1)[None, :]
    dice_cost = 1.0 - (2.0 * intersection + 1.0) / (denom + 1.0)

    cost = cls_weight * cls_cost + bce_weight * bce_cost + dice_weight * dice_cost
    cost_np = cost.detach().cpu().numpy()
    q_idx, t_idx = linear_sum_assignment(cost_np)
    return (
        torch.as_tensor(q_idx, dtype=torch.long),
        torch.as_tensor(t_idx, dtype=torch.long),
    )


def maskformer_loss(
    logits: Tensor,  # (B, Q, C+1)
    masks: List[Tensor],  # list of (Q, N_b)
    targets: List[Dict],
    num_classes: int,
    cfg_loss,
    device: str,
) -> Tuple[Tensor, Dict[str, float]]:
    B, Q, _ = logits.shape
    bg_class = num_classes  # last index = "no object"

    cls_weights = torch.ones(num_classes + 1, device=device)
    cls_weights[bg_class] = cfg_loss.no_object_weight

    cls_loss = logits.new_zeros(())
    bce_loss = logits.new_zeros(())
    dice_total = logits.new_zeros(())
    n_matched = 0

    for b in range(B):
        cls_b = logits[b]  # (Q, C+1)
        mask_b = masks[b]  # (Q, N_b)
        tgt = targets[b]
        tgt_labels = tgt["labels"].to(device)
        tgt_masks = tgt["masks"].to(device)

        q_idx, t_idx = hungarian_match(
            cls_b,
            mask_b,
            tgt_labels,
            tgt_masks,
            cfg_loss.cls_weight,
            cfg_loss.bce_weight,
            cfg_loss.dice_weight,
        )

        target_classes = torch.full((Q,), bg_class, dtype=torch.long, device=device)
        if q_idx.numel() > 0:
            target_classes[q_idx] = tgt_labels[t_idx]
            matched_mask_logits = mask_b[q_idx]  # (M, N_b)
            matched_target_masks = tgt_masks[t_idx].float()  # (M, N_b)
            bce_loss = bce_loss + F.binary_cross_entropy_with_logits(
                matched_mask_logits, matched_target_masks
            )
            dice_total = dice_total + dice_loss(matched_mask_logits, matched_target_masks).mean()
            n_matched += 1

        cls_loss = cls_loss + F.cross_entropy(cls_b, target_classes, weight=cls_weights)

    cls_loss = cls_loss / B
    if n_matched > 0:
        bce_loss = bce_loss / n_matched
        dice_total = dice_total / n_matched

    total = (
        cfg_loss.cls_weight * cls_loss
        + cfg_loss.bce_weight * bce_loss
        + cfg_loss.dice_weight * dice_total
    )
    return total, {
        "cls": float(cls_loss.detach()),
        "bce": float(bce_loss.detach()),
        "dice": float(dice_total.detach()),
        "total": float(total.detach()),
    }


def build_model(cfg, num_classes: int, device: str) -> MaskFormer:
    """Instantiate `cfg.model.backbone` via Hydra and wrap it in MaskFormer.

    The `backbone._target_` config field selects any
    `warpconvnet.models.*` (or other `nn.Module`) class. The resulting
    module is wrapped with `PointToVoxel` when
    `cfg.model.backbone_voxel_size` is non-null so it consumes `Points`
    directly; set the voxel size to `null` to feed the backbone raw
    points (e.g. a model that already operates on `Points`).
    """
    backbone_cfg = OmegaConf.to_container(cfg.model.backbone, resolve=True)
    inner = hydra.utils.instantiate(backbone_cfg)

    voxel_size = cfg.model.get("backbone_voxel_size", None)
    if voxel_size is not None:
        inner = PointToVoxel(inner_module=inner, voxel_size=voxel_size, concat_unpooled_pc=False)

    return MaskFormer(
        backbone=inner,
        hidden_dim=cfg.model.hidden_dim,
        num_queries=cfg.model.num_queries,
        num_heads=cfg.model.num_heads,
        num_decoders=cfg.model.num_decoders,
        dim_feedforward=cfg.model.dim_feedforward,
        dropout=cfg.model.dropout,
        num_classes=num_classes,
    ).to(device)


def train_epoch(model, loader, optimizer, cfg, num_classes, device, epoch, visualizer=None):
    model.train()
    pbar = tqdm(loader, desc=f"epoch {epoch}")
    for step, batch in enumerate(pbar):
        pc = build_pc(batch, device)
        targets = build_targets(batch, num_classes, cfg.data.ignore_index)

        optimizer.zero_grad()
        logits, masks = model(pc)
        loss, parts = maskformer_loss(logits, masks, targets, num_classes, cfg.loss, device)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if step % cfg.train.log_every == 0:
            pbar.set_postfix(parts)

        if visualizer is not None:
            sample = batch[0]
            visualizer.maybe_update(
                coords=torch.from_numpy(sample["coords"]),
                colors=torch.from_numpy(sample["colors"]),
                gt_instance=torch.from_numpy(sample["instance"]),
                pred_logits=logits[0],
                pred_masks=masks[0],
                epoch=epoch,
                step=step,
            )


@torch.no_grad()
def validate(model, loader, cfg, num_classes, device):
    model.eval()
    losses = []
    for batch in tqdm(loader, desc="val"):
        pc = build_pc(batch, device)
        targets = build_targets(batch, num_classes, cfg.data.ignore_index)
        logits, masks = model(pc)
        _, parts = maskformer_loss(logits, masks, targets, num_classes, cfg.loss, device)
        losses.append(parts["total"])
    print(f"val loss: {np.mean(losses):.4f}")


def register_config():
    cfg = OmegaConf.create(yaml.safe_load(CONFIG_YAML))
    cs = ConfigStore.instance()
    cs.store(name="maskformer_config", node=cfg)


@hydra.main(version_base=None, config_path=None, config_name="maskformer_config")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    wp.init()
    device = cfg.device
    num_classes = ScanNetInstanceDataset.NUM_CLASSES[cfg.data.label_set]

    train_transform = (
        hydra.utils.instantiate(cfg.data.train_transform)
        if cfg.data.get("train_transform")
        else None
    )

    train_ds = ScanNetInstanceDataset(
        root=cfg.paths.data_dir,
        split="train",
        label_set=cfg.data.label_set,
        voxel_size=cfg.data.voxel_size,
        transform=train_transform,
    )
    val_ds = ScanNetInstanceDataset(
        root=cfg.paths.data_dir,
        split="val",
        label_set=cfg.data.label_set,
        voxel_size=cfg.data.voxel_size,
    )
    print(f"train: {len(train_ds)}  val: {len(val_ds)}  classes: {num_classes}")

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.train.num_workers,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.train.num_workers,
        collate_fn=collate_fn,
    )

    model = build_model(cfg, num_classes, device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"MaskFormer params: {n_params/1e6:.1f}M")

    optimizer = optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=1e-4)
    scheduler = StepLR(optimizer, step_size=cfg.train.step_size, gamma=cfg.train.gamma)

    visualizer = None
    if cfg.get("viz", None) is not None and cfg.viz.enabled:
        # Hydra changes cwd, so add the repo root (two levels up) to sys.path
        # before importing the sibling utils package.
        import os
        import sys

        sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
        from examples.utils.viser_maskformer_visualizer import MaskFormerViserVisualizer

        visualizer = MaskFormerViserVisualizer(
            voxel_size=cfg.viz.voxel_size,
            port=cfg.viz.port,
            interval_seconds=cfg.viz.interval_seconds,
            score_thresh=cfg.viz.score_thresh,
            mask_thresh=cfg.viz.mask_thresh,
            color_range=(0.0, 255.0),
        )
        print(
            f"[viz] visualizer at http://localhost:{cfg.viz.port} "
            f"(refresh every {cfg.viz.interval_seconds}s)"
        )

    for epoch in range(cfg.train.epochs):
        train_epoch(model, train_loader, optimizer, cfg, num_classes, device, epoch, visualizer)
        validate(model, val_loader, cfg, num_classes, device)
        scheduler.step()


if __name__ == "__main__":
    register_config()
    main()
