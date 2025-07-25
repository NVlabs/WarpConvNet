# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import ssl
import urllib.request
import zipfile
from typing import Dict

import h5py
import torch
from torch import Tensor
from torch.utils.data import Dataset

ssl._create_default_https_context = ssl._create_unverified_context

# Constants
_URL = "https://huggingface.co/datasets/Msun/modelnet40/resolve/main/modelnet40_ply_hdf5_2048.zip"
_LABELS = [
    "airplane",
    "bathtub",
    "bed",
    "bench",
    "bookshelf",
    "bottle",
    "bowl",
    "car",
    "chair",
    "cone",
    "cup",
    "curtain",
    "desk",
    "door",
    "dresser",
    "flower_pot",
    "glass_box",
    "guitar",
    "keyboard",
    "lamp",
    "laptop",
    "mantel",
    "monitor",
    "night_stand",
    "person",
    "piano",
    "plant",
    "radio",
    "range_hood",
    "sink",
    "sofa",
    "stairs",
    "stool",
    "table",
    "tent",
    "toilet",
    "tv_stand",
    "vase",
    "wardrobe",
    "xbox",
]


class ModelNet40Dataset(Dataset):
    def __init__(self, root_dir: str = "./data/modelnet40", split: str = "train"):
        self.root_dir = root_dir
        self.split = split

        # Download and extract dataset if it doesn't exist
        if not os.path.exists(root_dir):
            self.download_and_extract()

        # Load file list
        file_list_path = os.path.join(root_dir, f"modelnet40_ply_hdf5_2048/{split}_files.txt")
        with open(file_list_path, "r") as f:
            self.file_list = [
                os.path.join(root_dir, line.strip().replace("data/", "")) for line in f.readlines()
            ]

        self.data = []
        self.labels = []
        self.load_data()

    def download_and_extract(self):
        print("Downloading and extracting dataset...")
        os.makedirs(self.root_dir, exist_ok=True)
        zip_path = os.path.join(self.root_dir, "modelnet40.zip")
        urllib.request.urlretrieve(_URL, zip_path)
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(self.root_dir)
        os.remove(zip_path)
        print("Dataset downloaded and extracted.")

    def load_data(self):
        for filepath in self.file_list:
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"File not found: {filepath}")
            with h5py.File(filepath, "r") as h5file:
                self.data.append(torch.from_numpy(h5file["data"][:]))
                self.labels.append(torch.from_numpy(h5file["label"][:]))

        if not self.data:
            raise FileNotFoundError("No valid data files were found.")

        self.data = torch.cat(self.data, dim=0)
        self.labels = torch.cat(self.labels, dim=0).squeeze()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx) -> Dict[str, Tensor]:
        points = self.data[idx]
        label = self.labels[idx]

        return {"coords": points, "labels": label}
