import argparse
import os
import sys
os.environ["FFREPORT"] = "loglevel=error"
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "loglevel;error"
# null_device = open(os.devnull, 'w')
# sys.stderr = null_device

import pathlib
import shutil
import datasets

from tali_wit.dataset_cache_generator import tali_cache_generator
from rich import print
from rich.traceback import install

install()

datasets.enable_caching()


def train_tali_generator():
    return tali_cache_generator(set_name="train", num_samples=10_000_000, root_path="/data/", num_workers=128)


def val_tali_generator():
    return tali_cache_generator(set_name="val", num_samples=10_050, root_path="/data/", num_workers=128)


def test_tali_generator():
    return tali_cache_generator(set_name="test", num_samples=10_050, root_path="/data/", num_workers=128)


if __name__ == "__main__":
    # create an argument parser that takes in the set name and number of samples
    parser = argparse.ArgumentParser()
    parser.add_argument("--set_name", type=str, required=True)
    args = parser.parse_args()

    dataset_generator = {"train": train_tali_generator, "val": val_tali_generator, "test": test_tali_generator}
    # for item in dataset_generator[args.set_name]():
    #     pass
    # create the dataset generator and save to the cache directory
    dataset = datasets.Dataset.from_generator(dataset_generator[args.set_name], keep_in_memory=False, cache_dir=f"/home/jupyter/tali_cache/{args.set_name}/", writer_batch_size=10000)

    # save the dataset to a file
    # dataset.save_to_disk(f"/home/jupyter/tali_cache/{args.set_name}/dataset")
