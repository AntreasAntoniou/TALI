import hashlib
import math
import multiprocessing as mp
import os
import pathlib
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from math import ceil
from typing import Optional

import datasets
import fire
import numpy as np
import yaml
from datasets import logging as datasets_logging
from rich import print, traceback
from rich.console import Console
from tqdm.auto import tqdm

from tali.utils import get_logger, load_json

traceback.install()

tali_dataset_dir = "/data/"

logger = get_logger(__name__, set_rich=True)

import huggingface_hub as hf_hub


def main(
    dataset_name: str = "Antreas/TALI",  # Name of the dataset to be uploaded to the Hub
    train_data_percentage: float = 1.0,  # Percentage of training data to use
    num_data_samples: Optional[int] = None,  # Number of data samples to use
    max_shard_size: str = "10GB",  # Maximum size of each dataset shard
    num_workers: Optional[
        int
    ] = None,  # Number of worker processes to use for loading the dataset
):
    """
    Prepares and uploads a dataset to the Hugging Face Hub.

    Args:
        dataset_name (str, optional): Name of the dataset to be uploaded to the Hub. Defaults to "Antreas/TALI".
        train_percentage (float, optional): Percentage of training data to use. Defaults to 1.0.
        max_shard_size (str, optional): Maximum size of each dataset shard. Defaults to "10GB".
        num_workers (int, optional): Number of worker processes to use for loading the dataset. Defaults to None.
    """
    print(
        f"Starting preparation and upload with arguments dataset_name: {dataset_name}, data_percentage: {train_data_percentage}, num_data_samples: {num_data_samples}, max_shard_size: {max_shard_size}, num_workers: {num_workers}"
    )

    dataset_dir = pathlib.Path(f"{tali_dataset_dir}/{dataset_name}")

    hf_hub.upload_file(
        repo_id=dataset_name,
        folder_path=dataset_dir / "test.parquet",
        path_in_repo="data/",
        use_auth_token=True,
    )
    hf_hub.upload_file(
        repo_id=dataset_name,
        folder_path=dataset_dir / "val.parquet",
        path_in_repo="data/",
        use_auth_token=True,
    )
    hf_hub.upload_file(
        repo_id=dataset_name,
        folder_path=dataset_dir / "train.parquet",
        path_in_repo="data/",
        use_auth_token=True,
    )
    # dataset = datasets.load_from_disk(dataset_dir)

    # dataset["train"].to_parquet(f"{dataset_dir}/train.parquet")
    # dataset["val"].to_parquet(f"{dataset_dir}/val.parquet")
    # dataset["test"].to_parquet(f"{dataset_dir}/test.parquet")


if __name__ == "__main__":
    fire.Fire(main)
