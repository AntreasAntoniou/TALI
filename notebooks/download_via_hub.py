import glob
import logging
import multiprocessing as mp
from pathlib import Path

import datasets
import huggingface_hub as hf_hub
import torch
from rich import print
from tqdm.auto import tqdm

dataset_cache = Path("/disk/scratch_fast1/data/")
logging.getLogger("datasets").setLevel(logging.DEBUG)
hf_hub.snapshot_download(
    repo_id="Antreas/TALI",
    repo_type="dataset",
    cache_dir=dataset_cache,
    resume_download=True,
    max_workers=mp.cpu_count(),
    ignore_patterns=["data/train-*", "data/val-*"],
)

from datasets import Array2D, ClassLabel, Features, Image, Sequence, Value

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

dataset_path = Path(
    "/disk/scratch_fast1/data/datasets--Antreas--TALI/snapshots/8a5a0e83a3afb64580c896c375af36d39bf1b42d/data/"
)
# Building a list of file paths for validation set
test_files = [file.as_posix() for file in Path(dataset_path).glob("*.parquet")]
print(f"Found {len(test_files)} files for testing set")
data_files = {
    "test": test_files[0],
    "val": test_files[1],
    "train": test_files[2],
}

dataset = datasets.load_dataset(
    "parquet",
    data_files=data_files,
    features=features,
    num_proc=1,
    cache_dir=dataset_cache,
)


# dataloader = torch.utils.data.DataLoader(
#     dataset["test"],
#     batch_size=1,
#     num_workers=mp.cpu_count(),
#     prefetch_factor=1,
# )


for sample in tqdm(dataset["test"]):
    print(list(sample.keys()))
    # pass
