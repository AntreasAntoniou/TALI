import pathlib
import datasets
import tqdm
import shutil
from rich import print
import fire
from datasets import Dataset


def gen_mod_data(
    tali_dataset_dir: str = "/data_large/datasets/tali-wit-2-1-buckets/",
    set_name: str = "train",
):
    dataset = datasets.load_from_disk(
        pathlib.Path(tali_dataset_dir) / f"{set_name}-set", keep_in_memory=True  # type: ignore
    )

    with tqdm.tqdm(total=len(dataset)) as pbar:
        for example in dataset:
            example["youtube_content_video"] = [
                item.replace("/data/datasets/tali-wit-2-1-buckets/", "/data/")
                for item in example["youtube_content_video"]
            ]
            example["youtube_subtitle_text"] = example["youtube_subtitle_text"].replace(
                "/data/datasets/tali-wit-2-1-buckets/", "/data/"
            )
            pbar.update(1)
            yield example


def gen_test():
    return gen_mod_data(set_name="test", tali_dataset_dir="/data_fast/tali-v-3-4/")


def gen_val():
    return gen_mod_data(
        set_name="val", tali_dataset_dir="/data_large/datasets/tali-wit-2-1-buckets/"
    )


def gen_train():
    return gen_mod_data(set_name="train", tali_dataset_dir="/data_fast/tali-v-3-4/")


if __name__ == "__main__":
    ds = Dataset.from_generator(gen_val, cache_dir="/data_fast/tali-v-3-4/val/cache")
    ds.save_to_disk("/data_fast/tali-v-3-4/val-set")

    # ds = Dataset.from_generator(
    #     gen_train, cache_dir="/data_fast/tali-v-3-4/train/cache"
    # )
    # ds.save_to_disk("/data_fast/tali-v-3-4/train-set")
