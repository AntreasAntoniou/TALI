import multiprocessing as mp
import pathlib
import time
from collections import defaultdict
from dataclasses import dataclass
from functools import cache
from typing import Any, Dict, List, Optional, Union

import datasets
import numpy as np
from rich import print
from tqdm import tqdm

from tali.data.data import (AnyModalSample, ModalityTypes,
                            default_image_transforms,
                            select_subtitles_between_timestamps)
from tali.data.data_plus import (TALIBaseTransformConfig, convert_to_pil,
                                 get_submodality_name, get_video_tensors,
                                 videoclip_to_video_audio_tensors)
from tali.frame_extractor import FrameSelectionMethod, extract_frames_pyav
from tali.utils import get_logger, load_json

logger = get_logger(__name__)


@dataclass
class TALIBaseTransformConfig:
    root_filepath: Union[str, pathlib.Path]
    modality_list: List
    rng_seed: int = 42
    image_size: int = 224
    num_video_frames: int = 30
    num_audio_frames: int = 44100
    clip_duration_in_seconds: float = 3
    priority_caption_language: Optional[str] = "en"


class TALIBaseDemoTransform:
    def __init__(
        self,
        cache_dir: pathlib.Path,
    ):
        self.cache_dir = cache_dir
        self.select_subtitles_between_timestamps = select_subtitles_between_timestamps

    def _apply_transform(self, input_dict: Dict[str, Any]):
        output_dict = input_dict.copy()

        wit_sample = input_dict["wit_features"]

        output_dict["wit_idx"] = [input_dict["wit_idx"]]
        output_dict["captions"] = {}

        for language in wit_sample["language"]:
            language_idx = wit_sample["language"].index(language)
            wit_text = {
                key: wit_sample[key][language_idx]
                for key in [
                    "caption_alt_text_description",
                    "caption_reference_description",
                    "caption_title_and_reference_description",
                    "context_page_description",
                    "context_section_description",
                    "hierarchical_section_title",
                    "page_title",
                    "section_title",
                ]
                if wit_sample[key][language_idx] is not None
            }
            output_dict["captions"][language] = wit_text

        output_dict[get_submodality_name(ModalityTypes.youtube_description.value)] = input_dict[
            "youtube_description_text"
        ]

        output_dict[get_submodality_name(ModalityTypes.youtube_subtitles.value)] = (
            "<ysub> "
            + select_subtitles_between_timestamps(
                subtitle_dict=input_dict["youtube_subtitle_text"],
                starting_timestamp=int(input_dict["youtube_video_starting_time"]),
                ending_timestamp=int(input_dict["youtube_video_starting_time"]) + 30,
            )
            + " </ysub>"
        )
        return output_dict

    def __call__(self, input_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Wrapper function for the transform function.

        This function is used to make the transform function configurable.

        Args:
            input_dict (Dict[str, Any]): The input dictionary.
            dict_keys([
                'image', 'image_url', 'item_idx',
                'wit_features', 'wit_idx', 'youtube_content_video',
                'youtube_subtitle_text', 'youtube_title_text',
                'youtube_description_text'])

        Returns:
            Dict[str, Any]: The transformed dictionary.
        """

        if isinstance(input_dict["item_idx"], list):
            output_dict = defaultdict(list)
            for idx in range(len(input_dict["item_idx"])):
                input_dict_ = {key: input_dict[key][idx] for key in input_dict.keys()}
                output_dict_ = self._apply_transform(input_dict_)
                for key in output_dict_.keys():
                    output_dict[key].append(output_dict_[key])
        else:
            output_dict = self._apply_transform(input_dict)

        return output_dict


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
    dataset_cache_path: pathlib.Path,
    num_download_workers: int = mp.cpu_count(),
    dataset_name: Optional[str] = None,
):
    from dataclasses import dataclass, field

    from datasets import ClassLabel, Features, Image, Sequence, Value

    dataset_path = download_dataset_via_hub(
        dataset_download_path=dataset_download_path,
        num_download_workers=num_download_workers,
        dataset_name=dataset_name,
    )
    # Building a list of file paths for validation set
    test_files = [
        file.as_posix()
        for file in pathlib.Path(dataset_path).glob("*.parquet")
        if "test" in file.as_posix()
    ]
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
    import multiprocessing as mp

    dataset = datasets.load_dataset(
        "parquet" if dataset_name is None else dataset_name,
        data_files=data_files,
        features=features,
        num_proc=mp.cpu_count() * 2,
        cache_dir=dataset_cache_path,
    )
    return dataset


if __name__ == "__main__":
    dataset_cache = pathlib.Path("/disk/scratch_fast0/tali/")
    dataset = load_dataset_via_hub(dataset_cache, dataset_name="Antreas/TALI")["test"]
    demo_transform = TALIBaseDemoTransform(cache_dir=dataset_cache / "cache")

    for sample in tqdm(dataset):
        sample = demo_transform(sample)
        print(list(sample.keys()))
