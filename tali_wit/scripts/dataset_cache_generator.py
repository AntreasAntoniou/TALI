from concurrent import futures
import pathlib
from collections import defaultdict
from math import floor
import torch.utils.data
from typing import Any, List, Union
import datasets

import numpy as np
import pyarrow as pa
import pyarrow.dataset as ds
import tqdm
from pytorchvideo.data.encoded_video import EncodedVideo
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    ShortSideScale,
    UniformTemporalSubsample,
)
from torchvision.transforms import Compose
from torchvision.transforms._transforms_video import CenterCropVideo
import datasets
from tali_wit.data import (
    AnyModalSample,
    ModalityDataSample,
    ModalityTypes,
    TALISchema,
    VideoCLIPScoreSchema,
)
from tali_wit.data_plus import TALIBase

from tali_wit.utils import get_logger, load_json, set_seed

logger = get_logger(__name__)

import torch


class TALICacheGenerator:
    def __init__(
        self,
        set_name: str,
        start_idx: int,
        end_idx: int,
        root_path="/data/datasets/tali-wit-2-1-buckets/",
        num_workers=8,
    ):
        self.num_workers = num_workers
        self.set_name = set_name
        self.root_path = root_path
        self.start_idx = start_idx
        self.end_idx = end_idx

        set_seed(42)

        self.dataset = TALIBase(
            set_name=self.set_name,
            tali_root_filepath=root_path,
            hf_tali_root_filepath=root_path,
            modality_list=[
                ModalityTypes.wit_image.value,
                ModalityTypes.wit_caption.value,
                ModalityTypes.wit_title.value,
                ModalityTypes.wit_main_body.value,
                ModalityTypes.youtube_image.value,
                ModalityTypes.youtube_video.value,
                ModalityTypes.youtube_subtitles.value,
                ModalityTypes.youtube_audio.value,
                ModalityTypes.youtube_description.value,
            ],
            rng_seed=42,
            top_k_tali=10,
            image_size=224,
            num_video_frames=10,
            num_audio_frames=16000 * 2,
            clip_duration_in_seconds=10,
            deterministic_sampling=False,
            infinite_sampling=True,
            dummy_batch_mode=False,
            image_text_model_name="openai/clip-vit-base-patch16",
            audio_model_name="openai/whisper-base",
            use_model_preprocessing=False,
        )
        self.start_idx = start_idx
        self.end_idx = end_idx

    def get_thread_based_generator(self):
        fn_argument_list = list(range(self.start_idx, self.end_idx))
        sample_idx = 0
        with tqdm.tqdm(
            total=self.end_idx - self.start_idx, smoothing=0.0
        ) as pbar:
            with futures.ThreadPoolExecutor(
                max_workers=self.num_workers
            ) as executor:
                for sample in executor.map(
                    self.dataset.__getitem__, fn_argument_list
                ):
                    if sample_idx >= (self.end_idx - self.start_idx):
                        break

                    sample_idx += 1
                    pbar.update(1)
                    yield sample

    def get_dataloader_based_generator(self):
        dataset = torch.utils.data.Subset(
            self.dataset, range(self.start_idx, self.end_idx)
        )
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
            drop_last=True,
            prefetch_factor=4,
            persistent_workers=False,
        )
        sample_idx = 0
        with tqdm.tqdm(
            total=self.end_idx - self.start_idx, smoothing=0.0
        ) as pbar:
            for batch in dataloader:
                try:
                    batch = [
                        {key: value[idx] for key, value in batch.items()}
                        for idx in range(len(batch["wit_idx"]))
                    ]
                    for idx, sample in enumerate(batch):
                        if sample_idx >= (self.end_idx - self.start_idx):
                            break

                        sample_idx += 1
                        pbar.update(1)
                        yield sample
                except Exception as e:
                    print(e)
                    continue

    def __call__(self) -> Any:
        return self.get_dataloader_based_generator()
