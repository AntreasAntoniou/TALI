import multiprocessing as mp
import pathlib
from math import ceil
from typing import Optional

import datasets
import fire
import huggingface_hub as hub
import numpy as np
import rich
import yaml
from datasets import logging as datasets_logging
from huggingface_hub import logging as hf_hub_logging
from rich import print, traceback
from rich.console import Console
from rich.traceback import Traceback
from tqdm.auto import tqdm

from tali.utils import get_logger, load_json

traceback.install()

tali_dataset_dir = "/data/"

np.random.seed(42)


console = Console()
datasets_logging.disable_progress_bar()

logger = get_logger(__name__, set_rich=True)

import os


def get_file_size(file_path):
    return os.path.getsize(file_path)


from collections import Counter


def get_byte_histogram(file_path):
    with open(file_path, "rb") as f:
        byte_content = f.read()
    return Counter(byte_content)


import hashlib


def get_file_hash(file_path):
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


import math


def calculate_entropy(file_path):
    byte_histogram = get_byte_histogram(file_path)
    entropy = 0
    total_bytes = sum(byte_histogram.values())
    for count in byte_histogram.values():
        p_x = count / total_bytes
        entropy += -p_x * math.log2(p_x)
    return entropy


from collections import defaultdict


def get_byte_pair_frequency(file_path):
    pair_freq = defaultdict(int)
    with open(file_path, "rb") as f:
        prev_byte = f.read(1)
        while byte := f.read(1):
            pair_freq[(prev_byte, byte)] += 1
            prev_byte = byte
    return pair_freq


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
    full_dataset = datasets.load_dataset(
        "Antreas/TALI",
        num_proc=mp.cpu_count() if num_workers is None else int(num_workers),
        cache_dir=tali_dataset_dir,
    )

    def data_generator(
        set_name, train_data_percentage: float = 1.0, num_data_samples=None
    ):
        dataset = full_dataset[set_name]
        if num_data_samples is None:
            num_data_samples = len(dataset)
        for idx, item in enumerate(tqdm(dataset)):
            if idx >= num_data_samples:
                break
            video_list = item["youtube_content_video"]
            video_list = video_list[
                : int(ceil(len(video_list) * train_data_percentage))
            ]
            video_list = sorted(video_list)
            if len(video_list) == 0:
                return None
            captions = load_json(item["youtube_subtitle_text"])

            new_captions = {}
            for key, value in captions.items():
                new_captions[str(key)] = "".join(value)
            captions = yaml.dump(new_captions)

            for video_path in video_list:
                temp_path = video_path.replace("/data/", tali_dataset_dir)
                video_path_actual: pathlib.Path = pathlib.Path(temp_path)

                if video_path_actual.exists():
                    logger.info(video_path_actual)
                    with open(video_path_actual, "rb") as f:
                        video_bytes = f.read()
                    video_starting_time = (
                        video_path.split("/")[-1].split("_")[1].split(".")[0]
                    )
                    youtube_subtitles = captions

                    sample = {
                        key: value
                        for key, value in item.items()
                        if key
                        not in [
                            "youtube_content_video",
                            "youtube_subtitle_text",
                        ]
                    }

                    sample["youtube_video_content"] = video_bytes
                    sample["youtube_video_starting_time"] = video_starting_time
                    sample["youtube_subtitle_text"] = youtube_subtitles

                    print(f"Summary statistics for {video_path_actual}")
                    print(f"File size: {get_file_size(video_path_actual)}")
                    print(f"File hash: {get_file_hash(video_path_actual)}")
                    print(f"Entropy: {calculate_entropy(video_path_actual)}")
                    print(
                        f"Byte histogram: {get_byte_histogram(video_path_actual)}"
                    )
                    print(
                        f"Byte pair frequency: {get_byte_pair_frequency(video_path_actual)}"
                    )

                    yield sample

    # print(data_generator("train", percentage=data_percentage).__next__())

    train_generator = lambda: data_generator(
        "train",
        train_data_percentage=train_data_percentage,
        num_data_samples=num_data_samples,
    )
    val_generator = lambda: data_generator(
        "val", num_data_samples=num_data_samples
    )
    test_generator = lambda: data_generator(
        "test", num_data_samples=num_data_samples
    )

    train_data = datasets.Dataset.from_generator(
        train_generator,
        num_proc=mp.cpu_count(),
        writer_batch_size=5000,
        cache_dir=tali_dataset_dir,
    )

    val_data = datasets.Dataset.from_generator(
        val_generator,
        writer_batch_size=5000,
        num_proc=mp.cpu_count(),
        cache_dir=tali_dataset_dir,
    )

    test_data = datasets.Dataset.from_generator(
        test_generator,
        writer_batch_size=5000,
        num_proc=mp.cpu_count(),
        cache_dir=tali_dataset_dir,
    )

    print(f"Pushing {dataset_name} to hub")

    dataset = datasets.DatasetDict(
        {"train": train_data, "val": val_data, "test": test_data}
    )
    # dataset_path = pathlib.Path(tali_dataset_dir) / dataset_name
    # dataset.save_to_disk(
    #     dataset_path,
    #     num_proc=mp.cpu_count(),
    #     max_shard_size="10GB",
    # )

    # dataset = datasets.load_from_disk(dataset_path)

    succesful_competion = False

    while not succesful_competion:
        try:
            dataset.push_to_hub(
                repo_id=f"{dataset_name}",
                num_shards={"train": 100, "val": 1, "test": 1},
            )
            succesful_competion = True

        except Exception as e:
            print("🚨 Full traceback of the exception:")
            console.print_exception(show_locals=True)
            print("Push to hub failed. Retrying...")


if __name__ == "__main__":
    fire.Fire(main)
