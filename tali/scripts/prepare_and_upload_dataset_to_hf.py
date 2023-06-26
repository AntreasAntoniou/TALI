import multiprocessing as mp
import os
import pathlib
from math import ceil

import datasets
import huggingface_hub as hub
import numpy as np
from tqdm.auto import tqdm
from rich import print
import fire
import yaml

from tali.utils import load_json

tali_dataset_dir = "/data/"

np.random.seed(42)


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
                    item["youtube_content_video"] = open(video_path_actual, "rb").read()
                    item["youtube_content_video_start_time"] = (
                        video_path.split("/")[-1].split("_")[1].split(".")[0]
                    )
                    item["youtube_subtitle_text"] = captions
                    yield item

    print(data_generator("train", percentage=train_percentage).__next__())

    # train_generator = lambda: data_generator("train", percentage=train_percentage)
    # val_generator = lambda: data_generator("val")
    # test_generator = lambda: data_generator("test")

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

    # print(f"Pushing {dataset_name} to hub")

    # dataset = datasets.DatasetDict(
    #     {"train": train_data, "val": val_data, "test": test_data}
    # )
    dataset_path = pathlib.Path(tali_dataset_dir) / dataset_name
    # dataset.save_to_disk(
    #     dataset_path,
    #     num_proc=mp.cpu_count(),
    #     max_shard_size="10GB",
    # )

    print(f"Creating repo {dataset_name}")
    hub.create_repo(
        dataset_name,
        token=os.environ.get("HF_TOKEN"),
        repo_type="dataset",
        exist_ok=True,
    )
    print(f"Uploading {dataset_path} to {dataset_name}")
    hub.upload_folder(
        folder_path=dataset_path,
        path_in_repo="data/",
        repo_id=dataset_name,
        repo_type="dataset",
        token=os.environ.get("HF_TOKEN"),
        multi_commits=True,
        multi_commits_verbose=True,
    )

    # succesful_competion = False

    # while not succesful_competion:
    #     try:
    #         print("Attempting to push to hub")
    #         dataset.push_to_hub(
    #             repo_id=f"{dataset_name}", max_shard_size=max_shard_size
    #         )
    #         succesful_competion = True
    #     except Exception as e:
    #         print(e)
    #         print("Push failed, retrying")


if __name__ == "__main__":
    fire.Fire(main)
