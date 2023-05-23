import datasets
import pathlib
from tqdm.auto import tqdm

tali_dataset_dir = "/tali-data/TALI"

if __name__ == "__main__":

    def train_generator():
        set_name = "train"
        dataset = datasets.load_from_disk(
            pathlib.Path(tali_dataset_dir) / f"{set_name}-set",
            keep_in_memory=True,
        )
        updated_video_list = []
        for item in tqdm(dataset):
            video_list = item["youtube_content_video"]
            for video_path in video_list:
                video_path: pathlib.Path = (
                    pathlib.Path(
                        tali_dataset_dir.replace("/data/", tali_dataset_dir)
                    )
                    / video_path
                )
                if video_path.exists():
                    updated_video_list.append(video_path.as_posix())
            if len(updated_video_list) == 0:
                continue
            item["youtube_content_video"] = updated_video_list
            yield item

    def val_generator():
        set_name = "val"
        dataset = datasets.load_from_disk(
            pathlib.Path(tali_dataset_dir) / f"{set_name}-set",
            keep_in_memory=True,
        )
        updated_video_list = []
        for item in tqdm(dataset):
            video_list = item["youtube_content_video"]

            for video_path in video_list:
                video_path: pathlib.Path = (
                    pathlib.Path(
                        tali_dataset_dir.replace("/data/", tali_dataset_dir)
                    )
                    / video_path
                )
                if video_path.exists():
                    updated_video_list.append(video_path.as_posix())
            if len(updated_video_list) == 0:
                continue
            item["youtube_content_video"] = updated_video_list
            yield item

    def test_generator():
        set_name = "test"
        dataset = datasets.load_from_disk(
            pathlib.Path(tali_dataset_dir) / f"{set_name}-set",
            keep_in_memory=True,
        )
        updated_video_list = []
        for item in tqdm(dataset):
            video_list = item["youtube_content_video"]
            for video_path in video_list:
                video_path: pathlib.Path = (
                    pathlib.Path(
                        tali_dataset_dir.replace("/data/", tali_dataset_dir)
                    )
                    / video_path
                )
                if video_path.exists():
                    updated_video_list.append(
                        video_path.as_posix().replace(
                            tali_dataset_dir, "/data/"
                        )
                    )
            if len(updated_video_list) == 0:
                continue
            item["youtube_content_video"] = updated_video_list
            yield item

    train_data = datasets.Dataset.from_generator(train_generator, num_proc=64)
    train_data.save_to_disk(pathlib.Path(tali_dataset_dir) / f"train-set")
    val_data = datasets.Dataset.from_generator(
        val_generator, writer_batch_size=1000
    )
    val_data.save_to_disk(pathlib.Path(tali_dataset_dir) / f"val-set")
    test_data = datasets.Dataset.from_generator(
        test_generator, writer_batch_size=1000
    )
    test_data.save_to_disk(pathlib.Path(tali_dataset_dir) / f"test-set")
