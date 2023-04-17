from math import floor
import os
from dataclasses import MISSING, dataclass
import pathlib
from typing import Any, Optional

import torch
from accelerate import Accelerator
from hydra.core.config_store import ConfigStore
from hydra_zen import (
    MISSING,
    ZenField,
    builds,
    make_config,
)
from timm.scheduler import CosineLRScheduler
from torch.utils.data import DataLoader

from tali_wit.boilerplate import Learner
from tali_wit.callbacks import UploadCheckpointsToHuggingFace
from tali_wit.data import ModalityTypes
from tali_wit.data_plus import *
from tali_wit.utils import get_hydra_config, get_logger
from tali_wit.wit import WITBase

from .models import ModalityConfig, MultiModalityConfig, TALIModel

CHECKPOINT_DIR = "${hf_cache_dir}"
NUM_WORKERS = "${num_workers}"
HF_USERNAME = "${hf_username}"
CODE_DIR = "${code_dir}"
TALI_DATASET_DIR = "${tali_dataset_dir}"
WIT_DATASET_DIR = "${wit_dataset_dir}"
EXPERIMENT_NAME = "${exp_name}"
EXPERIMENTS_ROOT_DIR = "${root_experiment_dir}"
TRAIN_BATCH_SIZE = "${train_batch_size}"
CURRENT_EXPERIMENT_DIR = "${current_experiment_dir}"
TRAIN_ITERS = "${learner.train_iters}"
REPO_PATH = "${repo_path}"
EXP_NAME = "${exp_name}"
SEED = "${seed}"
RESUME = "${resume}"
LOGGER_LEVEL = "${logger_level}"
GPU_MEMORY = 24  # in GB
DUMMY_BATCH_MODE = "${dummy_batch_mode}"
PREFETCH_FACTOR = "${prefetch_factor}"
PERSISTENT_WORKERS = "${persistent_workers}"
PIN_MEMORY = "${pin_memory}"
IMAGE_TEXT_MODEL_NAME = "${model.image_text_model_name}"
AUDIO_MODEL_NAME = "${model.audio_model_name}"


hydra_logger = get_logger("hydra")

HFModelUploadConfig = builds(
    UploadCheckpointsToHuggingFace, populate_full_signature=True
)

hf_upload = HFModelUploadConfig(
    repo_name=EXPERIMENT_NAME, repo_owner=HF_USERNAME
)

adamw_optimizer_config = builds(
    torch.optim.AdamW,
    populate_full_signature=True,
    zen_partial=True,
)


cosine_learning_rate_scheduler_config = builds(
    CosineLRScheduler,
    populate_full_signature=True,
    zen_partial=True,
)

accelerator_config = builds(Accelerator, populate_full_signature=True)

cosine_learning_rate_scheduler_config = cosine_learning_rate_scheduler_config()

model_config = TALIModel.build_config(populate_full_signature=True)

tali_dataset_config = TALIBase.build_config(
    populate_full_signature=True,
    set_name="train",
    tali_dataset_dir=TALI_DATASET_DIR,
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
    num_samples_per_episode=32,
    num_video_frames=10,
    num_audio_frames=2 * 16000,
    clip_duration_in_seconds=10,
    deterministic_sampling=False,
    infinite_sampling=False,
    dummy_batch_mode=DUMMY_BATCH_MODE,
    image_text_model_name=IMAGE_TEXT_MODEL_NAME,
    audio_model_name=AUDIO_MODEL_NAME,
)


wit_dataset_config = WITBase.build_config(
    populate_full_signature=True,
    set_name="train",
    wit_dataset_dir=pathlib.Path(WIT_DATASET_DIR),
    tali_dataset_dir=pathlib.Path(TALI_DATASET_DIR),
    image_size=224,
    num_samples_per_episode=32,
    deterministic_sampling=False,
    infinite_sampling=False,  # True,
    priority_caption_language="en",
    dummy_batch_mode=DUMMY_BATCH_MODE,
    image_text_model_name=IMAGE_TEXT_MODEL_NAME,
    audio_model_name=AUDIO_MODEL_NAME,
)

dataloader_config = builds(
    DataLoader, dataset=None, populate_full_signature=True
)

learner_config = builds(Learner, populate_full_signature=True)

learner_config = learner_config(
    model=None,
    experiment_name=EXPERIMENT_NAME,
    experiment_dir=CHECKPOINT_DIR,
    resume=RESUME,
    evaluate_every_n_steps=1000,
    checkpoint_after_validation=True,
    checkpoint_every_n_steps=500,
    train_iters=100000,
    limit_val_iters=250,
    dummy_batch_mode=DUMMY_BATCH_MODE,
    print_model_parameters=False,
)

default_callbacks = dict(hf_uploader=hf_upload)


def compute_batch_size_given_gpu_memory(reference_batch_size, gpu_memory):
    """Compute the batch size given the GPU memory and the reference batch size."""
    return int(floor(reference_batch_size * gpu_memory / 24))


@dataclass
class BaseConfig:
    # Must be passed at command line -- neccesary arguments

    exp_name: str = MISSING

    # Defaults for these are provided in the collect_config_store method,
    # but will be often overridden at command line

    model: Any = MISSING
    dataset: Any = MISSING
    dataloader: Any = MISSING
    optimizer: Any = MISSING
    scheduler: Any = MISSING
    learner: Any = MISSING
    callbacks: Any = MISSING

    hf_username: str = (
        os.environ["HF_USERNAME"] if "HF_USERNAME" in os.environ else MISSING
    )

    seed: int = 42

    freeze_backbone: bool = False
    resume: bool = False
    resume_from_checkpoint: Optional[int] = None
    print_config: bool = True
    # Dataloader config
    train_num_samples_per_episode: int = 96
    eval_num_samples_per_episode: int = 96
    num_workers: int = 2
    prefetch_factor: int = 1
    persistent_workers: bool = True
    pin_memory: bool = True

    train: bool = True
    test: bool = False
    dummy_batch_mode: bool = False
    download_latest: bool = True
    download_checkpoint_with_name: Optional[str] = None
    logger_level: str = "INFO"

    root_experiment_dir: str = (
        os.environ["EXPERIMENTS_DIR"]
        if "EXPERIMENTS_DIR" in os.environ
        else "/experiments"
    )

    tali_dataset_dir: str = (
        os.environ["TALI_DATASET_DIR"]
        if "TALI_DATASET_DIR" in os.environ
        else "/tali-data"
    )
    wit_dataset_dir: str = (
        os.environ["WIT_DATASET_DIR"]
        if "WIT_DATASET_DIR" in os.environ
        else "/wit-data"
    )

    current_experiment_dir: str = "${root_experiment_dir}/${exp_name}"
    hf_repo_path: str = "${hf_username}/${exp_name}"
    hf_cache_dir: str = "${current_experiment_dir}/repo"
    code_dir: str = (
        os.environ["CODE_DIR"]
        if "CODE_DIR" in os.environ
        else "${hydra:runtime.cwd}"
    )


# Using hydra might look a bit more verbose but it saves having to manually define
# future args, and makes it a lot easier to add whatever we need from the command line


def collect_config_store():
    config_store = ConfigStore.instance()
    ###################################################################################
    tali_vit_image_text_model_config = model_config(
        image_text_model_name="openai/clip-vit-base-patch16",
        audio_model_name="openai/whisper-base",
        multi_modality_config=MultiModalityConfig(
            image=ModalityConfig(support=True, pretrained=True),
            text=ModalityConfig(support=True, pretrained=True),
            audio=ModalityConfig(support=False, pretrained=False),
            video=ModalityConfig(support=False, pretrained=False),
        ),
    )

    tali_vit_image_text_model_scratch_config = model_config(
        image_text_model_name="openai/clip-vit-base-patch16",
        audio_model_name="openai/whisper-base",
        multi_modality_config=MultiModalityConfig(
            image=ModalityConfig(support=True, pretrained=False),
            text=ModalityConfig(support=True, pretrained=False),
            audio=ModalityConfig(support=False, pretrained=False),
            video=ModalityConfig(support=False, pretrained=False),
        ),
    )

    tali_vit_image_audio_model_config = model_config(
        image_text_model_name="openai/clip-vit-base-patch16",
        audio_model_name="openai/whisper-base",
        multi_modality_config=MultiModalityConfig(
            image=ModalityConfig(support=True, pretrained=True),
            text=ModalityConfig(support=False, pretrained=False),
            audio=ModalityConfig(support=True, pretrained=True),
            video=ModalityConfig(support=False, pretrained=False),
        ),
    )

    tali_vit_image_audio_model_scratch_config = model_config(
        image_text_model_name="openai/clip-vit-base-patch16",
        audio_model_name="openai/whisper-base",
        multi_modality_config=MultiModalityConfig(
            image=ModalityConfig(support=True, pretrained=False),
            text=ModalityConfig(support=False, pretrained=False),
            audio=ModalityConfig(support=True, pretrained=False),
            video=ModalityConfig(support=False, pretrained=False),
        ),
    )

    tali_vit_image_text_audio_model_config = model_config(
        image_text_model_name="openai/clip-vit-base-patch16",
        audio_model_name="openai/whisper-base",
        multi_modality_config=MultiModalityConfig(
            image=ModalityConfig(support=True, pretrained=True),
            text=ModalityConfig(support=True, pretrained=True),
            audio=ModalityConfig(support=True, pretrained=True),
            video=ModalityConfig(support=False, pretrained=False),
        ),
    )

    tali_vit_image_text_audio_model_scratch_config = model_config(
        image_text_model_name="openai/clip-vit-base-patch16",
        audio_model_name="openai/whisper-base",
        multi_modality_config=MultiModalityConfig(
            image=ModalityConfig(support=True, pretrained=False),
            text=ModalityConfig(support=True, pretrained=False),
            audio=ModalityConfig(support=True, pretrained=False),
            video=ModalityConfig(support=False, pretrained=False),
        ),
    )

    tali_vit_image_text_video_model_config = model_config(
        image_text_model_name="openai/clip-vit-base-patch16",
        audio_model_name="openai/whisper-base",
        multi_modality_config=MultiModalityConfig(
            image=ModalityConfig(support=True, pretrained=True),
            text=ModalityConfig(support=True, pretrained=True),
            audio=ModalityConfig(support=False, pretrained=False),
            video=ModalityConfig(support=True, pretrained=True),
        ),
    )

    tali_vit_image_text_video_model_scratch_config = model_config(
        image_text_model_name="openai/clip-vit-base-patch16",
        audio_model_name="openai/whisper-base",
        multi_modality_config=MultiModalityConfig(
            image=ModalityConfig(support=True, pretrained=False),
            text=ModalityConfig(support=True, pretrained=False),
            audio=ModalityConfig(support=False, pretrained=False),
            video=ModalityConfig(support=True, pretrained=False),
        ),
    )

    tali_vit_image_video_model_config = model_config(
        image_text_model_name="openai/clip-vit-base-patch16",
        audio_model_name="openai/whisper-base",
        multi_modality_config=MultiModalityConfig(
            image=ModalityConfig(support=True, pretrained=True),
            text=ModalityConfig(support=False, pretrained=False),
            audio=ModalityConfig(support=False, pretrained=False),
            video=ModalityConfig(support=True, pretrained=True),
        ),
    )

    tali_vit_image_video_model_scratch_config = model_config(
        image_text_model_name="openai/clip-vit-base-patch16",
        audio_model_name="openai/whisper-base",
        multi_modality_config=MultiModalityConfig(
            image=ModalityConfig(support=True, pretrained=False),
            text=ModalityConfig(support=False, pretrained=False),
            audio=ModalityConfig(support=False, pretrained=False),
            video=ModalityConfig(support=True, pretrained=False),
        ),
    )

    tali_vit_model_config = model_config(
        image_text_model_name="openai/clip-vit-base-patch16",
        audio_model_name="openai/whisper-base",
        multi_modality_config=MultiModalityConfig(
            image=ModalityConfig(support=True, pretrained=True),
            text=ModalityConfig(support=True, pretrained=True),
            audio=ModalityConfig(support=True, pretrained=True),
            video=ModalityConfig(support=True, pretrained=True),
        ),
    )

    tali_vit_model_scratch_config = model_config(
        image_text_model_name="openai/clip-vit-base-patch16",
        audio_model_name="openai/whisper-base",
        multi_modality_config=MultiModalityConfig(
            image=ModalityConfig(support=True, pretrained=False),
            text=ModalityConfig(support=True, pretrained=False),
            audio=ModalityConfig(support=True, pretrained=False),
            video=ModalityConfig(support=True, pretrained=False),
        ),
    )

    wit_dataset_image_text_config = wit_dataset_config(
        tali_dataset_dir=TALI_DATASET_DIR
    )

    tali_dataset_image_text_config = tali_dataset_config(
        set_name="train",
        modality_list=[
            ModalityTypes.wit_image.value,
            ModalityTypes.wit_caption.value,
            ModalityTypes.wit_title.value,
            ModalityTypes.wit_main_body.value,
            # ModalityTypes.youtube_video.value,
            ModalityTypes.youtube_image.value,
            ModalityTypes.youtube_subtitles.value,
            # ModalityTypes.youtube_audio.value,
            ModalityTypes.youtube_description.value,
        ],
    )

    tali_dataset_image_video_config = tali_dataset_config(
        set_name="train",
        modality_list=[
            ModalityTypes.wit_image.value,
            # ModalityTypes.wit_caption.value,
            # ModalityTypes.wit_title.value,
            # ModalityTypes.wit_main_body.value,
            ModalityTypes.youtube_video.value,
            # ModalityTypes.youtube_subtitles.value,
            # ModalityTypes.youtube_audio.value,
            # ModalityTypes.youtube_description.value,
        ],
    )

    tali_dataset_wit_image_audio_config = tali_dataset_config(
        set_name="train",
        modality_list=[
            ModalityTypes.wit_image.value,
            # ModalityTypes.wit_caption.value,
            # ModalityTypes.wit_title.value,
            # ModalityTypes.wit_main_body.value,
            # ModalityTypes.youtube_video.value,
            # ModalityTypes.youtube_subtitles.value,
            ModalityTypes.youtube_audio.value,
            # ModalityTypes.youtube_description.value,
        ],
    )

    tali_dataset_image_audio_config = tali_dataset_config(
        set_name="train",
        modality_list=[
            ModalityTypes.wit_image.value,
            # ModalityTypes.wit_caption.value,
            # ModalityTypes.wit_title.value,
            # ModalityTypes.wit_main_body.value,
            ModalityTypes.youtube_image.value,
            # ModalityTypes.youtube_video.value,
            # ModalityTypes.youtube_subtitles.value,
            ModalityTypes.youtube_audio.value,
            # ModalityTypes.youtube_description.value,
        ],
    )

    tali_dataset_youtube_image_audio_config = tali_dataset_config(
        set_name="train",
        modality_list=[
            # ModalityTypes.wit_image.value,
            # ModalityTypes.wit_caption.value,
            # ModalityTypes.wit_title.value,
            # ModalityTypes.wit_main_body.value,
            ModalityTypes.youtube_image.value,
            # ModalityTypes.youtube_video.value,
            # ModalityTypes.youtube_subtitles.value,
            ModalityTypes.youtube_audio.value,
            # ModalityTypes.youtube_description.value,
        ],
    )

    tali_dataset_text_audio_config = tali_dataset_config(
        set_name="train",
        modality_list=[
            # ModalityTypes.wit_image.value,
            ModalityTypes.wit_caption.value,
            ModalityTypes.wit_title.value,
            ModalityTypes.wit_main_body.value,
            # ModalityTypes.youtube_image.value,
            # ModalityTypes.youtube_video.value,
            ModalityTypes.youtube_subtitles.value,
            ModalityTypes.youtube_audio.value,
            # ModalityTypes.youtube_description.value,
        ],
    )

    tali_dataset_text_video_config = tali_dataset_config(
        set_name="train",
        modality_list=[
            # ModalityTypes.wit_image.value,
            ModalityTypes.wit_caption.value,
            ModalityTypes.wit_title.value,
            ModalityTypes.wit_main_body.value,
            # ModalityTypes.youtube_image.value,
            ModalityTypes.youtube_video.value,
            ModalityTypes.youtube_subtitles.value,
            # ModalityTypes.youtube_audio.value,
            # ModalityTypes.youtube_description.value,
        ],
    )

    tali_dataset_audio_video_config = tali_dataset_config(
        set_name="train",
        modality_list=[
            # ModalityTypes.wit_image.value,
            # ModalityTypes.wit_caption.value,
            # ModalityTypes.wit_title.value,
            # ModalityTypes.wit_main_body.value,
            ModalityTypes.youtube_video.value,
            # ModalityTypes.youtube_subtitles.value,
            ModalityTypes.youtube_audio.value,
            # ModalityTypes.youtube_description.value,
        ],
    )

    tali_dataset_omni_minus_audio_config = tali_dataset_config(
        set_name="train",
        modality_list=[
            ModalityTypes.wit_image.value,
            ModalityTypes.wit_caption.value,
            ModalityTypes.wit_title.value,
            ModalityTypes.wit_main_body.value,
            ModalityTypes.youtube_video.value,
            ModalityTypes.youtube_subtitles.value,
            # ModalityTypes.youtube_audio.value,
            ModalityTypes.youtube_description.value,
        ],
    )

    tali_dataset_omni_minus_video_config = tali_dataset_config(
        set_name="train",
        modality_list=[
            ModalityTypes.wit_image.value,
            ModalityTypes.wit_caption.value,
            ModalityTypes.wit_title.value,
            ModalityTypes.wit_main_body.value,
            # ModalityTypes.youtube_video.value,
            ModalityTypes.youtube_subtitles.value,
            ModalityTypes.youtube_audio.value,
            ModalityTypes.youtube_description.value,
        ],
    )

    tali_dataset_omni_config = tali_dataset_config(
        set_name="train",
        modality_list=[
            ModalityTypes.wit_image.value,
            ModalityTypes.wit_caption.value,
            ModalityTypes.wit_title.value,
            ModalityTypes.wit_main_body.value,
            ModalityTypes.youtube_video.value,
            ModalityTypes.youtube_subtitles.value,
            ModalityTypes.youtube_audio.value,
            ModalityTypes.youtube_description.value,
        ],
    )

    ###################################################################################

    config_store.store(
        group="model",
        name="tali_omni_base_patch16_224",
        node=tali_vit_model_config,
    )

    config_store.store(
        group="model",
        name="tali_omni_base_patch16_224_scratch",
        node=tali_vit_model_scratch_config,
    )

    config_store.store(
        group="model",
        name="tali_image_text_base_patch16_224",
        node=tali_vit_image_text_model_config,
    )

    config_store.store(
        group="model",
        name="tali_image_text_base_patch16_224_scratch",
        node=tali_vit_image_text_model_scratch_config,
    )

    config_store.store(
        group="model",
        name="tali_image_audio_base_patch16_224",
        node=tali_vit_image_audio_model_config,
    )

    config_store.store(
        group="model",
        name="tali_image_audio_base_patch16_224_scratch",
        node=tali_vit_image_audio_model_scratch_config,
    )

    config_store.store(
        group="model",
        name="tali_image_video_base_patch16_224",
        node=tali_vit_image_video_model_config,
    )

    config_store.store(
        group="model",
        name="tali_image_video_base_patch16_224_scratch",
        node=tali_vit_image_video_model_scratch_config,
    )

    config_store.store(
        group="model",
        name="tali_image_text_audio_base_patch16_224",
        node=tali_vit_image_text_audio_model_config,
    )

    config_store.store(
        group="model",
        name="tali_image_text_audio_base_patch16_224_scratch",
        node=tali_vit_image_text_audio_model_scratch_config,
    )

    config_store.store(
        group="model",
        name="tali_image_text_video_base_patch16_224",
        node=tali_vit_image_text_video_model_config,
    )

    config_store.store(
        group="model",
        name="tali_image_text_video_base_patch16_224_scratch",
        node=tali_vit_image_text_video_model_scratch_config,
    )

    ###################################################################################

    config_store.store(
        group="dataset",
        name="wit_image_text_dataset",
        node={
            "wit_dataset_image_text": (
                240,
                wit_dataset_image_text_config,
            ),
        },
    )

    config_store.store(
        group="dataset",
        name="tali_image_text_dataset",
        node={
            "tali_dataset_image_text": (
                128,
                tali_dataset_image_text_config,
            ),
        },
    )

    config_store.store(
        group="dataset",
        name="wit_tali_image_text_dataset",
        node={
            "wit_dataset_image_text": (
                240,
                wit_dataset_image_text_config,
            ),
            "tali_dataset_image_text": (
                128,
                tali_dataset_image_text_config,
            ),
        },
    )

    config_store.store(
        group="dataset",
        name="wit_tali_image_audio_dataset",
        node={
            "wit_dataset_image_text": (
                240,
                wit_dataset_image_text_config,
            ),
            "tali_dataset_image_audio": (
                8,
                tali_dataset_image_audio_config,
            ),
        },
    )

    config_store.store(
        group="dataset",
        name="wit_tali_image_video_dataset",
        node={
            "wit_dataset_image_text": (
                240,
                wit_dataset_image_text_config,
            ),
            "tali_dataset_image_video": (
                8,
                tali_dataset_image_video_config,
            ),
        },
    )

    config_store.store(
        group="dataset",
        name="wit_tali_image_text_audio_dataset",
        node={
            "wit_dataset_image_text": (
                240,
                wit_dataset_image_text_config,
            ),
            "tali_dataset_image_text": (
                128,
                tali_dataset_image_text_config,
            ),
            "tali_dataset_image_audio": (
                8,
                tali_dataset_image_audio_config,
            ),
            "tali_dataset_text_audio": (
                8,
                tali_dataset_text_audio_config,
            ),
        },
    )

    config_store.store(
        group="dataset",
        name="wit_tali_image_text_audio_video_dataset",
        node={
            "wit_dataset_image_text": (
                240,
                wit_dataset_image_text_config,
            ),
            "tali_dataset_image_text": (
                128,
                tali_dataset_image_text_config,
            ),
            "tali_dataset_omni": (
                4,
                tali_dataset_omni_config,
            ),
            # "tali_dataset_image_audio": (
            #     str(
            #         compute_batch_size_given_gpu_memory(
            #             reference_batch_size=18, gpu_memory=GPU_MEMORY
            #         )
            #     ),
            #     tali_dataset_image_audio_config,
            # ),
            # "tali_dataset_image_video": (
            #     str(
            #         compute_batch_size_given_gpu_memory(
            #             reference_batch_size=12, gpu_memory=GPU_MEMORY
            #         )
            #     ),
            #     tali_dataset_image_video_config,
            # ),
            # "tali_dataset_text_audio": (
            #     str(
            #         compute_batch_size_given_gpu_memory(
            #             reference_batch_size=18, gpu_memory=GPU_MEMORY
            #         )
            #     ),
            #     tali_dataset_text_audio_config,
            # ),
            # "tali_dataset_text_video": (
            #     str(
            #         compute_batch_size_given_gpu_memory(
            #             reference_batch_size=12, gpu_memory=GPU_MEMORY
            #         )
            #     ),
            #     tali_dataset_text_video_config,
            # ),
            # "tali_dataset_audio_video": (
            #     str(
            #         compute_batch_size_given_gpu_memory(
            #             reference_batch_size=12, gpu_memory=GPU_MEMORY
            #         )
            #     ),
            #     tali_dataset_audio_video_config,
            # ),
        },
    )

    config_store.store(
        group="dataloader",
        name="default",
        node=dataloader_config(
            batch_size=1,
            num_workers=NUM_WORKERS,
            pin_memory=PIN_MEMORY,
            shuffle=True,
            prefetch_factor=PREFETCH_FACTOR,
            persistent_workers=PERSISTENT_WORKERS,
        ),
    )

    config_store.store(
        group="optimizer",
        name="adamw",
        node=adamw_optimizer_config(lr=1e-5, weight_decay=0.0),
    )

    config_store.store(
        group="scheduler",
        name="cosine-annealing",
        node=cosine_learning_rate_scheduler_config,
    )

    ###################################################################################
    config_store.store(
        group="learner",
        name="default",
        node=learner_config,
    )

    config_store.store(
        group="callbacks", name="default", node=default_callbacks
    )

    config_store.store(
        group="hydra",
        name="default",
        node=get_hydra_config(logger_level=LOGGER_LEVEL),
    )

    zen_config = []

    for value in BaseConfig.__dataclass_fields__.values():
        item = (
            ZenField(name=value.name, hint=value.type, default=value.default)
            if value.default is not MISSING
            else ZenField(name=value.name, hint=value.type)
        )
        zen_config.append(item)

    config = make_config(
        *zen_config,
        hydra_defaults=[
            "_self_",
            dict(learner="default"),
            dict(optimizer="adamw"),
            dict(scheduler="cosine-annealing"),
            dict(model="tali_image_text_base_patch16_224"),
            dict(dataset="tali_image_text_dataset"),
            dict(dataloader="default"),
            dict(hydra="default"),
            dict(callbacks="default"),
        ],
    )
    # Config
    config_store.store(name="config", node=config)

    return config_store
