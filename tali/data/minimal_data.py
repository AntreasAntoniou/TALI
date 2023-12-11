import multiprocessing as mp
import pathlib
from typing import Optional

import datasets
from rich import print
from tqdm import tqdm


def download_dataset_via_hub(
    dataset_name: str,
    dataset_download_path: pathlib.Path,
    num_download_workers: int = mp.cpu_count(),
):
    import huggingface_hub as hf_hub

    download_folder = hf_hub.snapshot_download(
        repo_id=dataset_name,
        repo_type="dataset",
        cache_dir=dataset_download_path,
        resume_download=True,
        max_workers=num_download_workers,
        ignore_patterns=[],
    )

    return pathlib.Path(download_folder) / "data"


def load_dataset_via_hub(
    dataset_download_path: pathlib.Path,
    num_download_workers: int = mp.cpu_count(),
    dataset_name: Optional[str] = None,
):
    pass

    from datasets import Features, Image, Sequence, Value

    dataset_path = download_dataset_via_hub(
        dataset_download_path=dataset_download_path,
        num_download_workers=num_download_workers,
        dataset_name=dataset_name,
    )
    # Building a list of file paths for validation set

    train_files = [
        file.as_posix()
        for file in pathlib.Path(dataset_path).glob("*.parquet")
        if "train" in file.as_posix()
    ]
    val_files = [
        file.as_posix()
        for file in pathlib.Path(dataset_path).glob("*.parquet")
        if "val" in file.as_posix()
    ]
    test_files = [
        file.as_posix()
        for file in pathlib.Path(dataset_path).glob("*.parquet")
        if "test" in file.as_posix()
    ]
    print(
        f"Found {len(test_files)} files for testing set, {len(train_files)} for training set and {len(val_files)} for validation set"
    )
    data_files = {
        "test": test_files,
        "val": val_files,
        "train": train_files,
    }

    features = Features(
        {
            "image": Image(
                decode=True
            ),  # Set `decode=True` if you want to decode the images, otherwise `decode=False`
            "image_url": Value("string"),
            "item_idx": Value("int64"),
            "wit_features": Sequence(
                {
                    "attribution_passes_lang_id": Value("bool"),
                    "caption_alt_text_description": Value("string"),
                    "caption_reference_description": Value("string"),
                    "caption_title_and_reference_description": Value("string"),
                    "context_page_description": Value("string"),
                    "context_section_description": Value("string"),
                    "hierarchical_section_title": Value("string"),
                    "is_main_image": Value("bool"),
                    "language": Value("string"),
                    "page_changed_recently": Value("bool"),
                    "page_title": Value("string"),
                    "page_url": Value("string"),
                    "section_title": Value("string"),
                }
            ),
            "wit_idx": Value("int64"),
            "youtube_title_text": Value("string"),
            "youtube_description_text": Value("string"),
            "youtube_video_content": Value("binary"),
            "youtube_video_starting_time": Value("string"),
            "youtube_subtitle_text": Value("string"),
            "youtube_video_size": Value("int64"),
            "youtube_video_file_path": Value("string"),
        }
    )

    dataset = datasets.load_dataset(
        "parquet" if dataset_name is None else dataset_name,
        data_files=data_files,
        features=features,
        num_proc=1,
        cache_dir=dataset_download_path / "cache",
    )
    return dataset


if __name__ == "__main__":
    dataset_cache = pathlib.Path("/disk/scratch_fast0/tali/")
    dataset = load_dataset_via_hub(dataset_cache, dataset_name="Antreas/TALI")[
        "test"
    ]

    for sample in tqdm(dataset):
        print(list(sample.keys()))
