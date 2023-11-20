import logging
import multiprocessing as mp
import pathlib

import datasets
import torch
import torchvision
from rich import print
from torch.utils.data import DataLoader

logging.getLogger("datasets").setLevel(logging.DEBUG)
# cache_path = pathlib.Path("/disk/scratch_fast0/tali/")
# print(f"Cache path: {cache_path}")
# dataset = datasets.load_dataset(
#     "Antreas/TALI",
#     split="train",
#     data_dir=cache_path / "data",
#     cache_dir=cache_path,
# )

import datasets

data_files = {
    "train": "/disk/scratch_fast0/tali/data/train-*.parquet",
    "test": "/disk/scratch_fast0/tali/data/test-*.parquet",
    "val": "/disk/scratch_fast0/tali/data/val-*.parquet",
}

# The data will be loaded and concatenated automatically from the shards
dataset = datasets.load_dataset("parquet", data_files=data_files)

loader = DataLoader(dataset, batch_size=1, num_workers=16, pin_memory=False)

for item in loader:
    print(item)
    break
