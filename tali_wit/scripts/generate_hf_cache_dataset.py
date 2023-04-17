import argparse
import os
import sys
from numpy import False_

from responses import start

os.environ["TMPDIR"] = "/data/tmp"
os.environ["FFREPORT"] = "loglevel=error"
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "loglevel;error"
# null_device = open(os.devnull, 'w')
# sys.stderr = null_device

import pathlib
import shutil
import datasets

from tali_wit.dataset_cache_generator import TALICacheGenerator
from rich import print
from rich.traceback import install
import os
import tempfile
import shutil


install()

datasets.enable_caching()

local_path = "/data/datasets/tali-wit-2-1-buckets/"
remote_path = "/data/"
remote_path = "/home/jupyter/"

current_path = remote_path


def empty_tmp_dirs():
    tmp_dir = tempfile.gettempdir()

    # Iterate through the contents of the temporary directory
    for entry in os.listdir(tmp_dir):
        entry_path = os.path.join(tmp_dir, entry)

        # Remove files and directories
        try:
            if os.path.isfile(entry_path):
                os.remove(entry_path)
            elif os.path.isdir(entry_path):
                shutil.rmtree(entry_path)
            else:
                print(f"Skipping unknown entry: {entry_path}")
        except Exception as e:
            print(f"Failed to remove {entry_path}: {e}")


if __name__ == "__main__":
    # create an argument parser that takes in the set name and number of samples
    parser = argparse.ArgumentParser()
    parser.add_argument("--set_name", type=str, default="train")
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=1000)
    parser.add_argument("--num_workers", type=int, default=16)
    args = parser.parse_args()

    dataset_cache_generator = TALICacheGenerator(
        set_name=args.set_name,
        root_path=current_path,
        num_workers=args.num_workers,
        start_idx=args.start_idx,
        end_idx=args.end_idx,
    )
    dataset = datasets.Dataset.from_generator(
        dataset_cache_generator,
        keep_in_memory=False,
        cache_dir=f"/data/tali_cache/{args.set_name}/f{args.start_idx}_t{args.end_idx}",
        writer_batch_size=1000,
        num_proc=1,
    )

    # save the dataset to a file
    dataset.save_to_disk(
        f"/data/tali_cache/{args.set_name}/f{args.start_idx}_t{args.end_idx}"
    )
