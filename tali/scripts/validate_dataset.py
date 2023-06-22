import multiprocessing as mp
import pathlib
from math import ceil

import datasets
import numpy as np
from tqdm.auto import tqdm
from rich import print
import fire

from tali.utils import load_json

tali_dataset_dir = "/data/"


def main(dataset_name="Antreas/TALI", train_percentage=1.0, max_shard_size="10GB"):
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
            captions = load_json(item["youtube_subtitle_text"])

            for video_path in video_list:
                temp_path = video_path.replace("/data/", tali_dataset_dir)
                video_path_actual: pathlib.Path = pathlib.Path(temp_path)

                if video_path_actual.exists():
                    # video actual looks like this /data/video_data.parquet/10/10000/LfjW3emXfMU/360p_2220.mp4
                    item["youtube_content_video"] = open(video_path_actual, "rb").read()
                    item["youtube_content_video_start_time"] = (
                        video_path.split("/")[-1].split("_")[1].split(".")[0]
                    )
                    item["youtube_subtitle_text"] = captions
                    yield item

    train_generator = lambda: data_generator("train", percentage=train_percentage)
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

    print(f"Pushing {dataset_name} to hub")

    dataset = datasets.DatasetDict(
        {"train": train_data, "val": val_data, "test": test_data}
    )
    succesful_competion = False

    while not succesful_competion:
        try:
            dataset.push_to_hub(
                repo_id=f"{dataset_name}", max_shard_size=max_shard_size
            )
            succesful_competion = True
        except Exception as e:
            print(e)


if __name__ == "__main__":
    fire.Fire(main)
