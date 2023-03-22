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

from tali_wit.utils import get_logger, load_json

logger = get_logger(__name__)

root_path = "/data/"

train_dataset = TALIBase(
        set_name="train",
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

val_dataset = TALIBase(
        set_name="val",
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

test_dataset = TALIBase(
        set_name="test",
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

dataset_dict = {"train": train_dataset, "val": val_dataset, "test": test_dataset}

def get_sample(input_dict):
    dataset_instance = input_dict["dataset_instance"]
    set_name = input_dict["set_name"]
    idx = input_dict["idx"]
    return dataset_instance[set_name][idx]

import torch

def tali_cache_generator(
    set_name, num_samples, root_path="/data/datasets/tali-wit-2-1-buckets/", num_workers=8
):
    failed_samples = 0
    dataloader = torch.utils.data.DataLoader(dataset=dataset_dict[set_name], batch_size=1, num_workers=num_workers, pin_memory=False, shuffle=True, prefetch_factor=8)
    sample_idx = 0
    fn_argument_list = []
    
    # with tqdm.tqdm(total=num_samples, smoothing=0.0) as pbar:
    #     for idx in range(num_samples):
    #         fn_argument_list.append(dict(dataset_instance=dataset_dict[set_name], set_name=set_name, idx=idx))
    #         pbar.update(1)
    
    with tqdm.tqdm(total=num_samples, smoothing=0.0) as pbar:
        # with futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
            for idx, batch in enumerate(
                dataloader,
            ):
                
                
                try:
                    # for key, value in item.items():
                    #     print(f"{key}: {value.shape if hasattr(value, 'shape') else value}")
                    # print(f"{list(batch.keys())}: {batch['wit_idx']}")
                    batch_as_a_list = [{key: value[i] for key, value in batch.items()} for i in range(len(batch["wit_idx"]))]
                    for item in batch_as_a_list:
                        pbar.update(1)
                        sample_idx += 1
                        yield item
                except Exception as e:
                    print(e)
                
                if sample_idx >= num_samples:
                    return
