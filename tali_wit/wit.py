from calendar import c
import copy
import json
import os
import pathlib
import random
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum
from math import floor
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import datasets

import numpy as np
import pandas as pd
import PIL
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import torch
import torch.nn as nn
import tqdm
from hydra_zen import builds, instantiate
from PIL import Image
from pytorchvideo.data.encoded_video import EncodedVideo
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    ShortSideScale,
    UniformTemporalSubsample,
)
from rich import print
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataloader import default_collate
from torch.utils.data import Dataset, Subset, DataLoader
from torchvision.transforms import Compose, RandomCrop, Resize, ToTensor
from torchvision.transforms._transforms_video import CenterCropVideo
from traitlets import default
from transformers import CLIPModel, CLIPProcessor
import datasets
from tali_wit.data import (
    AnyModalSample,
    dataclass_collate,
    default_image_transforms,
    ModalityTypes,
    select_subtitles_between_timestamps,
    TALIDataset,
)

from tali_wit.data_plus import get_submodality_name

from tali_wit.decorators import configurable
from tali_wit.utils import get_logger, load_json, save_json
from tali_wit.models import ModalityConfig

logger = get_logger(__name__)


@configurable
class WITBase(Dataset):
    def __init__(
        self,
        cache_dir: str,
        tali_dataset_dir: str,
        image_size: int,
        set_name: str,
        deterministic_sampling: bool = False,
        infinite_sampling: bool = False,
        priority_caption_language: Optional[str] = None,
        dummy_batch_mode: bool = False,
    ):
        super().__init__()
        self.cache_dir = cache_dir
        self.image_size = image_size
        self.wit_transform = WITBaseTransform(
            image_size=image_size,
            priority_caption_language=priority_caption_language,
            deterministic_sampling=deterministic_sampling,
        )
        self.dataset = datasets.load_dataset(
            "wikimedia/wit_base",
            split="train",
            cache_dir=cache_dir,
        )
        self.indices_filepath = pathlib.Path(cache_dir) / "wit_indices.json"

        if not self.indices_filepath.exists():
            tali_val_dataset = datasets.load_from_disk(
                pathlib.Path(tali_dataset_dir) / "val-set"
            )
            tali_val_indices = [sample["wit_idx"] for sample in tali_val_dataset]
            # print(len(tali_val_indices))

            tali_test_dataset = datasets.load_from_disk(
                pathlib.Path(tali_dataset_dir) / "test-set"
            )
            tali_test_indices = [sample["wit_idx"] for sample in tali_test_dataset]
            # print(len(tali_test_indices))

            train_wit_indices = []
            with tqdm.tqdm(total=len(self.dataset)) as pbar:
                for i in range(len(self.dataset)):
                    if i not in tali_val_indices and i not in tali_test_indices:
                        train_wit_indices.append(i)
                    pbar.update(1)

            # print(len(train_wit_indices))

            self.indices = {
                "train": train_wit_indices,
                "val": tali_val_indices,
                "test": tali_test_indices,
            }
            save_json(
                filepath=os.path.join(self.cache_dir, "wit_indices.json"),
                dict_to_store=self.indices,
            )
        else:
            self.indices = load_json(self.indices_filepath)

        self.dataset = Subset(self.dataset, self.indices[set_name])

        self.infinite_sampling = infinite_sampling
        if infinite_sampling:
            self.num_samples = 10**8
        else:
            self.num_samples = len(self.dataset)

    def __getitem__(self, idx):
        if self.infinite_sampling:
            idx = idx % len(self.dataset)

        sample = self.dataset[idx] | {"wit_idx": idx}
        sample = self.wit_transform(sample)

        return sample

    def __len__(self):
        return self.num_samples


class WITBaseTransform:
    def __init__(
        self,
        image_size,
        deterministic_sampling: bool = False,
        priority_caption_language: Optional[str] = None,
    ):
        self.image_size = image_size
        self.image_transform = default_image_transforms(self.image_size)
        self.deterministic_sampling = deterministic_sampling
        self.priority_caption_language = priority_caption_language

    def __call__(self, input_dict: Dict[str, Any]) -> Any:
        input_dict = {
            "image": input_dict["image"],
            "image_url": input_dict["image_url"],
            "wit_idx": input_dict["wit_idx"],
            "wit_features": input_dict["wit_features"].copy(),
            "language": input_dict["wit_features"]["language"].copy(),
        }

        if self.deterministic_sampling:
            rng = np.random.RandomState(input_dict["wit_idx"])
        else:
            seconds_rng = int(time.time()) % 1000000
            rng = np.random.RandomState(input_dict["wit_idx"] + seconds_rng)

        output_dict = {}
        wit_sample = input_dict["wit_features"]
        output_dict["wit_idx"] = input_dict["wit_idx"]

        output_dict[
            get_submodality_name(ModalityTypes.wit_image.value)
        ] = self.image_transform(input_dict["image"])

        if self.priority_caption_language is None:
            choose_language = rng.choice(wit_sample["language"])
        elif self.priority_caption_language in wit_sample["language"]:
            choose_language = self.priority_caption_language
        else:
            choose_language = rng.choice(wit_sample["language"])
        choose_language = rng.choice(wit_sample["language"])
        language_idx = wit_sample["language"].index(choose_language)
        wit_text = [
            f"<{key}> <{choose_language}>"
            + wit_sample[key][language_idx]
            + f"</{choose_language}> </{key}>"
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
        ]
        output_dict[get_submodality_name(ModalityTypes.wit_caption.value)] = rng.choice(
            wit_text
        )

        return output_dict


if __name__ == "__main__":
    import tqdm
    from rich import print
    from rich.traceback import install
    import cProfile
    import pstats
    import datasets

    install()
    os.environ["HYDRA_FULL_ERROR"] = "1"

    def sample():
        dataset = WITBase(
            cache_dir="/data/datasets/tali-wit-2-1-buckets/wit_cache",
            tali_dataset_dir="/home/evolvingfungus/forge/workspaces/tali-2-2/",
            image_size=224,
            deterministic_sampling=True,
            infinite_sampling=False,  # True,
            priority_caption_language="en",
            set_name="train",
        )
        dataloader = DataLoader(
            dataset,
            batch_size=128,
            num_workers=8,
            shuffle=True,
            collate_fn=dataclass_collate,
        )
        num_samples = 100
        with tqdm.tqdm(total=len(dataloader)) as pbar:
            for i, example in enumerate(dataloader):
                pbar.set_description(f"Processing {i}th example")
                pbar.update(1)

    pr = cProfile.Profile()
    pr.runcall(sample)

    ps = pstats.Stats(pr).sort_stats("tottime")
    ps.print_stats()
# write a transform for the wit dataset, and, add an option for a youtube image sampling process
