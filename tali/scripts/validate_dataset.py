import datasets
import pathlib
from tqdm.auto import tqdm

tali_dataset_dir = "/tali-data/"

if __name__ == "__main__":
    # def train_generator():
    #     set_name = "train"
    #     dataset = datasets.load_from_disk(
    #         pathlib.Path(tali_dataset_dir) / f"{set_name}-set",
    #         keep_in_memory=True,
    #     )

    #     for item in tqdm(dataset):
    #         video_list = item["youtube_content_video"]
    #         updated_video_list = []
    #         for video_path in video_list:
    #             temp_path = video_path.replace(
    #                 "/data/", tali_dataset_dir + "TALI/"
    #             )
    #             video_path_actual: pathlib.Path = pathlib.Path(temp_path)

    #             if video_path_actual.exists():
    #                 updated_video_list.append(video_path)

    #         if len(updated_video_list) == 0:
    #             continue

    #         item["youtube_content_video"] = updated_video_list
    #         yield item

    # def val_generator():
    #     set_name = "val"
    #     dataset = datasets.load_from_disk(
    #         pathlib.Path(tali_dataset_dir) / f"{set_name}-set",
    #         keep_in_memory=True,
    #     )

    #     for item in tqdm(dataset):
    #         video_list = item["youtube_content_video"]
    #         updated_video_list = []
    #         for video_path in video_list:
    #             temp_path = video_path.replace(
    #                 "/data/", tali_dataset_dir + "TALI/"
    #             )
    #             video_path_actual: pathlib.Path = pathlib.Path(temp_path)

    #             if video_path_actual.exists():
    #                 updated_video_list.append(video_path)

    #         if len(updated_video_list) == 0:
    #             continue

    #         item["youtube_content_video"] = updated_video_list
    #         yield item

    # def test_generator():
    #     set_name = "test"
    #     dataset = datasets.load_from_disk(
    #         pathlib.Path(tali_dataset_dir) / f"{set_name}-set",
    #         keep_in_memory=True,
    #     )

    #     for item in tqdm(dataset):
    #         video_list = item["youtube_content_video"]
    #         updated_video_list = []
    #         for video_path in video_list:
    #             temp_path = video_path.replace(
    #                 "/data/", tali_dataset_dir + "TALI/"
    #             )
    #             video_path_actual: pathlib.Path = pathlib.Path(temp_path)

    #             if video_path_actual.exists():
    #                 updated_video_list.append(video_path)

    #         if len(updated_video_list) == 0:
    #             continue

    #         item["youtube_content_video"] = updated_video_list
    #         yield item

    # train_data = datasets.Dataset.from_generator(
    #     train_generator,
    #     num_proc=64,
    #     keep_in_memory=True,
    #     writer_batch_size=10000,
    # )
    # train_data.save_to_disk(pathlib.Path(tali_dataset_dir) / f"train-set")
    # val_data = datasets.Dataset.from_generator(
    #     val_generator, writer_batch_size=1000, num_proc=64, keep_in_memory=True
    # )
    # val_data.save_to_disk(pathlib.Path(tali_dataset_dir) / f"val-set")
    # test_data = datasets.Dataset.from_generator(
    #     test_generator,
    #     writer_batch_size=10000,
    #     num_proc=64,
    #     keep_in_memory=True,
    # )
    # test_data.save_to_disk(pathlib.Path(tali_dataset_dir) / f"test-set")

    train_data = datasets.load_from_disk(
        pathlib.Path(tali_dataset_dir) / "train-set"
    )
    val_data = datasets.load_from_disk(
        pathlib.Path(tali_dataset_dir) / "val-set"
    )
    test_data = datasets.load_from_disk(
        pathlib.Path(tali_dataset_dir) / "test-set"
    )

    dataset = datasets.DatasetDict(
        {"train": train_data, "val": val_data, "test": test_data}
    )
    dataset.push_to_hub(repo_id="Antreas/TALI")
