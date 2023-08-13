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

np.random.seed(42)


console = Console()
datasets_logging.disable_progress_bar()

logger = get_logger(__name__, set_rich=True)


def get_file_size(file_path):
    return os.path.getsize(file_path)


def get_byte_histogram(byte_content):
    return Counter(byte_content)


def get_file_hash(byte_content: bytes):
    sha256_hash = hashlib.sha256()

    sha256_hash.update(byte_content)
    return sha256_hash.hexdigest()


def calculate_entropy(byte_content):
    byte_histogram = get_byte_histogram(byte_content)
    entropy = 0
    total_bytes = sum(byte_histogram.values())
    for count in byte_histogram.values():
        p_x = count / total_bytes
        entropy += -p_x * math.log2(p_x)
    return entropy


def get_byte_pair_frequency(file_path):
    pair_freq = defaultdict(int)
    with open(file_path, "rb") as f:
        prev_byte = f.read(1)
        while byte := f.read(1):
            pair_freq[(prev_byte, byte)] += 1
            prev_byte = byte
    return pair_freq


def process_video(video_path, youtube_subtitles, item):
    temp_path = video_path.replace("/data/", tali_dataset_dir)
    video_path_actual: pathlib.Path = pathlib.Path(temp_path)

    if video_path_actual.exists():
        logger.info(video_path_actual)
        with open(video_path_actual, "rb") as f:
            video_bytes = f.read()
        video_starting_time = (
            video_path.split("/")[-1].split("_")[1].split(".")[0]
        )

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

        sample["youtube_video_size"] = get_file_size(video_path_actual)
        sample["youtube_video_file_path"] = video_path_actual.as_posix()

        return sample


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
    # full_dataset: datasets.DatasetDict = datasets.load_dataset(
    #     "Antreas/TALI",
    #     num_proc=mp.cpu_count() if num_workers is None else int(num_workers),
    #     cache_dir=tali_dataset_dir,
    # )  # type: ignore

    # def data_generator(
    #     set_name, train_data_percentage: float = 1.0, num_data_samples=None
    # ):
    #     dataset = full_dataset[set_name]
    #     if num_data_samples is None:
    #         num_data_samples = len(dataset)
    #     for idx, item in enumerate(tqdm(dataset)):
    #         if idx >= num_data_samples:
    #             break
    #         video_list = item["youtube_content_video"]  # type: ignore
    #         video_list = video_list[
    #             : int(ceil(len(video_list) * train_data_percentage))
    #         ]
    #         video_list = sorted(video_list)
    #         if len(video_list) == 0:
    #             return None
    #         captions = load_json(item["youtube_subtitle_text"])  # type: ignore

    #         new_captions = {}
    #         for key, value in captions.items():
    #             new_captions[str(key)] = "".join(value)
    #         captions = yaml.dump(new_captions)

    #         with ThreadPoolExecutor(max_workers=mp.cpu_count()) as executor:
    #             for sample in executor.map(
    #                 process_video,
    #                 video_list,
    #                 [captions] * len(video_list),
    #                 [item] * len(video_list),
    #             ):
    #                 if sample is not None:
    #                     yield sample

    #     # print(data_generator("train", percentage=data_percentage).__next__())

    # train_generator = lambda: data_generator(
    #     "train",
    #     train_data_percentage=train_data_percentage,
    #     num_data_samples=num_data_samples,
    # )
    # val_generator = lambda: data_generator(
    #     "val", num_data_samples=num_data_samples
    # )
    # test_generator = lambda: data_generator(
    #     "test", num_data_samples=num_data_samples
    # )

    # train_data = datasets.Dataset.from_generator(
    #     train_generator,
    #     num_proc=mp.cpu_count(),
    #     writer_batch_size=5000,
    #     cache_dir=tali_dataset_dir,
    # )

    # val_data = datasets.Dataset.from_generator(
    #     val_generator,
    #     writer_batch_size=5000,
    #     num_proc=mp.cpu_count(),
    #     cache_dir=tali_dataset_dir,
    # )

    # test_data = datasets.Dataset.from_generator(
    #     test_generator,
    #     writer_batch_size=5000,
    #     num_proc=mp.cpu_count(),
    #     cache_dir=tali_dataset_dir,
    # )
    train_data_dir = "/data/generator/default-e3d897e3cfea555e/0.0.0/generator-train-00052-of-00211.arrow"
    val_data_dir = "/data/generator/default-344765592abf12ec/0.0.0/generator-train-00052-of-00211.arrow"
    test_data_dir = "/data/generator/default-eb76b39d4609a487/0.0.0/generator-train-00052-of-00211.arrow"

    # train_data = datasets.Dataset.load_from_disk(train_data_dir)
    # val_data = datasets.Dataset.load_from_disk(val_data_dir)
    # test_data = datasets.Dataset.load_from_disk(test_data_dir)

    train_data = datasets.load_from_disk(train_data_dir)
    val_data = datasets.load_from_disk(val_data_dir)
    test_data = datasets.load_from_disk(test_data_dir)

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
                num_shards={"train": 400, "val": 1, "test": 1},
            )
            succesful_competion = True

        except Exception as e:
            print(f"🚨 Full traceback of the exception: {e}")
            console.print_exception(show_locals=True)
            print("Push to hub failed. Retrying...")


if __name__ == "__main__":
    fire.Fire(main)
