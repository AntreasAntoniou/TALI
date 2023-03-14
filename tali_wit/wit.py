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


class WITBaseTransform:
    def __init__(self, image_size):
        self.image_size = image_size
        self.image_transform = default_image_transforms(self.image_size)

    def __call__(self, input_dict: Dict[str, Any]) -> Any:
        output_dict = {}
        choose_language = np.random.choice(input_dict["language"])
        language_idx = input_dict["language"].index(choose_language)
        wit_text = [
            f"<{key}> " + input_dict[key][language_idx] + f" </{key}>"
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
            if input_dict[key][language_idx] is not None
        ]
        output_dict["wikipedia_text"] = np.random.choice(wit_text)
        output_dict[
            get_submodality_name(ModalityTypes.wit_image.value)
        ] = self.image_transform(input_dict["image"])
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
        dataset = datasets.load_dataset("wikimedia/wit_base")
        dataset = dataset.with_transform(WITBaseTransform(image_size=224))
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
