import pathlib
import datasets
import tqdm
import shutil
from rich import print
import fire


def main(
    tali_dataset_dir: str = "/data_large/datasets/tali-wit-2-1-buckets/",
    set_name: str = "train",
    source_dataset_dir: str = "/data_large/datasets/tali-wit-2-1-buckets/",
    target_dataset_dir: str = "/data_fast/tali-v-3-4/",
):
    dataset = datasets.load_from_disk(
        pathlib.Path(tali_dataset_dir) / f"{set_name}-set", keep_in_memory=True  # type: ignore
    )

    with tqdm.tqdm(total=len(dataset)) as pbar:
        for example in dataset:
            # source_path = pathlib.Path(source_dataset_dir) / example["path"]
            # target_path = pathlib.Path(target_dataset_dir) / example["path"]
            # target_path.parent.mkdir(parents=True, exist_ok=True)
            # shutil.copyfile(source_path, target_path)
            # print(list(example.keys()))  # type: ignore
            # print(f"{example['youtube_subtitle_text']}, {example['youtube_title_text']}")  # type: ignore
            # for video_path in example["youtube_content_video"][10:]:
            #     video_path: pathlib.Path = pathlib.Path(source_dataset_dir) / video_path
            #     source_path = video_path.as_posix().replace(
            #         "/data/datasets/tali-wit-2-1-buckets/", source_dataset_dir
            #     )
            #     target_path = video_path.as_posix().replace(
            #         "/data/datasets/tali-wit-2-1-buckets/", target_dataset_dir
            #     )

            subtitle_path: pathlib.Path = (
                pathlib.Path(source_dataset_dir)
                / example["youtube_subtitle_text"]
            )
            source_path = subtitle_path.as_posix().replace(
                "/data/datasets/tali-wit-2-1-buckets/", source_dataset_dir
            )
            target_path = subtitle_path.as_posix().replace(
                "/data/datasets/tali-wit-2-1-buckets/", target_dataset_dir
            )

            # copy source path to target path
            target_path = pathlib.Path(target_path)
            target_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source_path, target_path.as_posix())
            pbar.update(1)


if __name__ == "__main__":
    fire.Fire(main)
