import logging
import multiprocessing as mp
from pathlib import Path

import datasets
import torch
from rich import print
from tqdm.auto import tqdm

dataset_cache = Path("/disk/scratch_fast1/data/")
logging.getLogger("datasets").setLevel(logging.DEBUG)
dataset = datasets.load_dataset(
    "Antreas/TALI",
    split="train",
    streaming=True,
    cache_dir=dataset_cache,
)

sample_idx = 0
for sample in tqdm(dataset):
    print(sample)
    sample_idx += 1
    if sample_idx > 1000:
        break
