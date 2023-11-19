import logging
import multiprocessing as mp
from pathlib import Path

import datasets
import torch
from rich import print

dataset_cache = Path("/disk/scratch_fast1/data/")
logging.getLogger("datasets").setLevel(logging.DEBUG)
dataset = datasets.load_dataset(
    "Antreas/TALI",
    revision="refs/convert/parquet",
    split="train",
    streaming=True,
    cache_dir=dataset_cache,
)


for sample in dataset:
    print(sample)
    break
