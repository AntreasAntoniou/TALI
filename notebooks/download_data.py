import multiprocessing as mp

import datasets
from rich import print

dataset = datasets.load_dataset(
    "Antreas/TALI",
    split="val",
    # streaming=True,
    cache_dir="/data-fast1/datasets/tali",
    num_proc=mp.cpu_count(),
)
