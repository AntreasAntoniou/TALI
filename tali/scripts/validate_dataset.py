import multiprocessing as mp
import pathlib
from math import ceil

import datasets
import numpy as np
from tqdm.auto import tqdm

from tali.data.data import select_subtitles_between_timestamps
from tali.utils import load_json

tali_dataset_dir = "/data/"

if __name__ == "__main__":
    full_dataset = datasets.load_dataset(
        "Antreas/TALI", num_proc=mp.cpu_count(), cache_dir=tali_dataset_dir
    )

    def data_generator(set_name, percentage: float = 1.0):
        dataset = full_dataset[set_name]

        for item in tqdm(dataset):
            video_list = item["youtube_content_video"]
            video_list = np.random.choice(
                video_list, int(ceil(len(video_list) * percentage))
            )
            if len(video_list) == 0:
                continue
            captions = item["youtube_subtitle_text"]
            captions = select_subtitles_between_timestamps(
                subtitle_dict=load_json(
                    captions.replace(
                        "/data/",
                        tali_dataset_dir,
                    )
                ),
                starting_timestamp=0,
                ending_timestamp=100000000,
            )

            for video_path in video_list:
                temp_path = video_path.replace("/data/", tali_dataset_dir)
                video_path_actual: pathlib.Path = pathlib.Path(temp_path)

                if video_path_actual.exists():
                    item["youtube_content_video"] = open(video_path_actual, "rb").read()
                    item["youtube_subtitle_text"] = captions
                    yield item

    train_generator = lambda: data_generator("train", percentage=0.1)
    val_generator = lambda: data_generator("val")
    test_generator = lambda: data_generator("test")

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

    dataset = datasets.DatasetDict(
        {
            "train": train_data,
            "val": val_data,
            "test": test_data,
        }
    )
    succesful_competion = False
    while not succesful_competion:
        try:
            dataset.push_to_hub(repo_id="Antreas/TALI-small", max_shard_size="5GB")
            succesful_competion = True
        except Exception as e:
            print(e)

    train_generator = lambda: data_generator("train", percentage=0.5)

    train_data = datasets.Dataset.from_generator(
        train_generator,
        num_proc=mp.cpu_count(),
        writer_batch_size=5000,
        cache_dir=tali_dataset_dir,
    )

    dataset = datasets.DatasetDict(
        {"test": test_data, "train": train_data, "val": val_data}
    )
    succesful_competion = False
    while not succesful_competion:
        try:
            dataset.push_to_hub(repo_id="Antreas/TALI-base", max_shard_size="5GB")
            succesful_competion = True
        except Exception as e:
            print(e)

    train_generator = lambda: data_generator("train", percentage=1.0)

    train_data = datasets.Dataset.from_generator(
        train_generator,
        num_proc=mp.cpu_count(),
        writer_batch_size=5000,
        cache_dir=tali_dataset_dir,
    )

    dataset = datasets.DatasetDict(
        {"test": test_data, "train": train_data, "val": val_data}
    )
    succesful_competion = False
    while not succesful_competion:
        try:
            dataset.push_to_hub(repo_id="Antreas/TALI-large", max_shard_size="5GB")
            succesful_competion = True
        except Exception as e:
            print(e)
