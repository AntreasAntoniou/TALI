from concurrent import futures
import pathlib
from collections import defaultdict
from math import floor
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


def tali_cache_generator(
    set_name, num_samples, root_path="/data/datasets/tali-wit-2-1-buckets/"
):
    failed_samples = 0
    dataset = TALIBase(
        set_name=set_name,
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
    sample_idx = 0
    with tqdm.tqdm(total=num_samples, smoothing=0.0) as pbar:
        with futures.ThreadPoolExecutor(max_workers=8) as executor:
            for idx, item in enumerate(
                executor.map(dataset.__getitem__, range(num_samples))
            ):
                pbar.update(1)

                if idx >= num_samples:
                    break

                if item is False:
                    failed_samples += 1
                    continue

                pbar.set_description(
                    f"idx: {idx} Failed samples: {failed_samples}, total length: {len(dataset)}"
                )
                sample_idx += 1
                # for key, value in item.items():
                #     print(f"{key}: {value.shape if hasattr(value, 'shape') else value}")

                yield item
