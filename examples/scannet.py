# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Use hydra config to run the script
# python scannet.py model._target=mink_unet.MinkUNet34 train.batch_size=12

# WARNING: This is a simple example of how to use the warpconvnet library.
# The data loader does NOT apply any augmentation to the point cloud.
# Please create your own data augmentation pipeline for high-quality training.
from typing import Dict, List, Optional, Tuple
import yaml

try:
    import hydra
    from hydra.core.config_store import ConfigStore
    from omegaconf import DictConfig, OmegaConf
except ImportError:
    print("Hydra and OmegaConf not installed, pip install hydra-core omegaconf")
    exit(1)

try:
    import wandb
except ImportError:
    wandb = None

import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import warp as wp
from torch import Tensor
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torchmetrics.classification import MulticlassConfusionMatrix
from tqdm import tqdm
from warpconvnet.dataset.scannet import ScanNetDataset
from warpconvnet.geometry.base.geometry import Geometry
from warpconvnet.geometry.types.points import Points
from warpconvnet.nn.modules.sparse_pool import PointToSparseWrapper

# Embedded YAML configuration
CONFIG_YAML = """
# Path configuration
paths:
  data_dir: ./data/scannet_3d
  output_dir: ./results/
  ckpt_path: null

# Training configuration.
train:
  batch_size: 12
  lr: 0.001
  epochs: 100
  step_size: 20
  gamma: 0.7
  num_workers: 8

# Testing configuration
test:
  batch_size: 12
  num_workers: 4

# Dataset configuration
data:
  num_classes: 20
  voxel_size: 0.02
  ignore_index: 255

# Model configuration
model:
  _target_: mink_unet.MinkUNet18
  in_channels: 3
  out_channels: 20
  in_type: "voxel"

# General configuration
device: "cuda"
use_wandb: false
seed: 42
"""


def collate_fn(batch: List[Dict[str, Tensor]]):
    """
    Return dict of list of tensors
    """
    keys = batch[0].keys()
    return {key: [torch.tensor(item[key]) for item in batch] for key in keys}


class DataToTensor:
    def __init__(
        self,
        device: str = "cuda",
    ):
        self.device = device

    def __call__(
        self, batch_dict: Dict[str, Tensor]
    ) -> Tuple[Geometry, Dict[str, Tensor]]:
        # cat all features into a single tensor
        cat_batch_dict = {
            k: torch.cat(v, dim=0).to(self.device) for k, v in batch_dict.items()
        }
        return (
            Points.from_list_of_coordinates(
                batch_dict["coords"],
                features=batch_dict["colors"],
            ).to(self.device),
            cat_batch_dict,
        )


def confusion_matrix_to_metrics(conf_matrix: Tensor) -> Dict[str, float]:
    """
    Return accuracy, miou, class_iou, class_accuracy

    Rows are ground truth, columns are predictions.
    """
    conf_matrix = conf_matrix.cpu()
    accuracy = (conf_matrix.diag().sum() / conf_matrix.sum()).item() * 100
    class_accuracy = (conf_matrix.diag() / conf_matrix.sum(dim=1)) * 100
    class_iou = conf_matrix.diag() / (
        conf_matrix.sum(dim=1) + conf_matrix.sum(dim=0) - conf_matrix.diag()
    )
    miou = class_iou.mean().item() * 100
    return {
        "accuracy": accuracy,
        "miou": miou,
        "class_iou": class_iou,
        "class_accuracy": class_accuracy,
    }


@torch.amp.autocast(device_type="cuda", enabled=True)
def train(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    epoch: int,
    cfg: DictConfig,
):
    model.train()
    bar = tqdm(train_loader)
    dict_to_data = DataToTensor(device=cfg.device)
    for batch_dict in bar:
        start_time = time.time()
        optimizer.zero_grad()
        st, batch_dict = dict_to_data(batch_dict)
        output = model(st.to(cfg.device))
        loss = F.cross_entropy(
            output.features,
            batch_dict["labels"].long().to(cfg.device),
            reduction="mean",
            ignore_index=cfg.data.ignore_index,
        )
        loss.backward()
        optimizer.step()
        bar.set_description(f"Train Epoch: {epoch} Loss: {loss.item(): .3f}")
        if cfg.use_wandb:
            wandb.log(
                {
                    "train/loss": loss.item(),
                    "train/lr": optimizer.param_groups[0]["lr"],
                    "train/time": time.time() - start_time,
                    "epoch": epoch,
                }
            )


@torch.amp.autocast(device_type="cuda", enabled=True)
@torch.inference_mode()
def test(
    model: nn.Module,
    test_loader: DataLoader,
    cfg: DictConfig,
    num_test_batches: Optional[int] = None,
):
    model.eval()
    torch.cuda.empty_cache()
    confusion_matrix = MulticlassConfusionMatrix(
        num_classes=cfg.data.num_classes, ignore_index=cfg.data.ignore_index
    ).to(cfg.device)
    test_loss = 0
    num_batches = 0
    dict_to_data = DataToTensor(device=cfg.device)
    for batch_dict in tqdm(test_loader):
        st, batch_dict = dict_to_data(batch_dict)
        output = model(st.to(cfg.device))
        labels = batch_dict["labels"].long().to(cfg.device)
        test_loss += F.cross_entropy(
            output.features,
            labels,
            reduction="mean",
            ignore_index=cfg.data.ignore_index,
        ).item()
        pred = output.features.argmax(dim=1)
        confusion_matrix.update(pred, labels)
        num_batches += 1
        if num_test_batches is not None and num_batches >= num_test_batches:
            break

    metrics = confusion_matrix_to_metrics(confusion_matrix.compute())

    if cfg.use_wandb:
        wandb.log(
            {
                "test/loss": test_loss / num_batches,
                "test/accuracy": metrics["accuracy"],
                "test/miou": metrics["miou"],
                "test/class_iou": metrics["class_iou"],
                "test/class_accuracy": metrics["class_accuracy"],
            }
        )

    print(
        f"\nTest set: Average loss: {test_loss / num_batches: .4f}, Accuracy: {metrics['accuracy']: .2f}%, mIoU: {metrics['miou']: .2f}%\n"
    )
    return metrics


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def register_config():
    """Register the embedded configuration with Hydra's ConfigStore."""
    cfg_dict = yaml.safe_load(CONFIG_YAML)
    cfg = OmegaConf.create(cfg_dict)

    cs = ConfigStore.instance()
    cs.store(name="scannet_config", node=cfg)


@hydra.main(version_base=None, config_path=None, config_name="scannet_config")
def main(cfg: DictConfig):
    # Initialize seed
    set_seed(cfg.seed)

    # Print config
    print(OmegaConf.to_yaml(cfg))

    if cfg.use_wandb:
        wandb.init(
            project="scannet-segmentation",
            config=OmegaConf.to_container(cfg),
        )

    device = torch.device(cfg.device)

    train_loader = DataLoader(
        ScanNetDataset(
            cfg.paths.data_dir,
            split="train",
        ),
        batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers,
        shuffle=True,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        ScanNetDataset(
            cfg.paths.data_dir,
            split="val",
        ),
        batch_size=cfg.test.batch_size,
        num_workers=cfg.test.num_workers,
        shuffle=False,
        collate_fn=collate_fn,
    )

    # Model initialization
    model = hydra.utils.instantiate(cfg.model).to(device)
    if cfg.use_wandb:
        wandb.watch(model)

    if hasattr(cfg.model, "in_type") and cfg.model.in_type == "voxel":
        model = PointToSparseWrapper(
            inner_module=model,
            voxel_size=cfg.data.voxel_size,
            concat_unpooled_pc=False,
        )

    optimizer = optim.AdamW(model.parameters(), lr=cfg.train.lr)
    scheduler = StepLR(optimizer, step_size=cfg.train.step_size, gamma=cfg.train.gamma)

    # Test before training
    metrics = test(model, test_loader, cfg, num_test_batches=5)
    for epoch in range(1, cfg.train.epochs + 1):
        train(
            model,
            train_loader,
            optimizer,
            epoch,
            cfg,
        )
        metrics = test(model, test_loader, cfg)
        scheduler.step()

    print(f"Final mIoU: {metrics['miou']: .2f}%")


if __name__ == "__main__":
    register_config()
    main()
