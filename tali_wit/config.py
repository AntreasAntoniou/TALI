import multiprocessing
import os
from dataclasses import MISSING, dataclass, field
from datetime import timedelta
from typing import Any, Dict, List, Optional, Union

import timm
import torch
import wandb
from accelerate import Accelerator
from hydra.core.config_store import ConfigStore
from hydra_zen import (
    MISSING,
    ZenField,
    builds,
    hydrated_dataclass,
    make_config,
)
from omegaconf import OmegaConf
from timm.scheduler import CosineLRScheduler
from torch.utils.data import DataLoader

from tali_wit.boilerplate import Learner
from tali_wit.callbacks import UploadCheckpointsToHuggingFace

from .data import ModalityTypes, TALIDataset
from .models import ModalityConfig, MultiModalityConfig, TALIModel

CHECKPOINT_DIR = "${hf_repo_dir}"
NUM_WORKERS = "${num_workers}"
HF_USERNAME = "${hf_username}"
CODE_DIR = "${code_dir}"
DATASET_DIR = "${data_dir}"
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


wandb_args_config = builds(wandb.init, populate_full_signature=True)

wandb_args_default = wandb_args_config(
    project=os.environ.get("WANDB_PROJECT", "mlproject"),
    resume="allow",  # allow, True, False, must
    dir=CURRENT_EXPERIMENT_DIR,
    save_code=True,
)


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

dataset_config = TALIDataset.build_config(populate_full_signature=True)

dataloader_config = builds(
    DataLoader, dataset=None, populate_full_signature=True
)

learner_config = builds(Learner, populate_full_signature=True)

learner_config = learner_config(
    model=None,
    experiment_name=EXPERIMENT_NAME,
    experiment_dir=CHECKPOINT_DIR,
    resume=RESUME,
    evaluate_every_n_steps=500,
    checkpoint_after_validation=True,
    checkpoint_every_n_steps=500,
    train_iters=10000,
)

default_callbacks = dict(hf_uploader=hf_upload)


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

    wandb_args: Any = MISSING

    hf_username: str = (
        os.environ["HF_USERNAME"] if "HF_USERNAME" in os.environ else MISSING
    )

    seed: int = 42

    freeze_backbone: bool = False
    resume: bool = False
    resume_from_checkpoint: Optional[int] = None
    print_config: bool = False
    train_batch_size: int = 96
    eval_batch_size: int = 96
    num_workers: int = 1
    train: bool = True
    test: bool = False
    download_latest: bool = True
    download_checkpoint_with_name: Optional[str] = None
    logger_level: str = "INFO"

    root_experiment_dir: str = (
        os.environ["EXPERIMENTS_DIR"]
        if "EXPERIMENTS_DIR" in os.environ
        else "/experiments"
    )

    data_dir: str = (
        os.environ["DATASET_DIR"] if "DATASET_DIR" in os.environ else "/data"
    )

    current_experiment_dir: str = "${root_experiment_dir}/${exp_name}"
    repo_path: str = "${hf_username}/${exp_name}"
    hf_repo_dir: str = "${current_experiment_dir}/repo"
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
        num_video_frames=8,
        num_audio_frames=8,
        audio_sampling_rate=16000,
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
        num_video_frames=8,
        num_audio_frames=8,
        audio_sampling_rate=16000,
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
        num_video_frames=8,
        num_audio_frames=8,
        audio_sampling_rate=16000,
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
        num_video_frames=8,
        num_audio_frames=8,
        audio_sampling_rate=16000,
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
        num_video_frames=8,
        num_audio_frames=8,
        audio_sampling_rate=16000,
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
        num_video_frames=8,
        num_audio_frames=8,
        audio_sampling_rate=16000,
    )

    wit_dataset_image_text_config = dataset_config(
        set_name="train",
        root_filepath=DATASET_DIR,
        modality_list=[
            ModalityTypes.wit_image.value,
            ModalityTypes.wit_caption.value,
            ModalityTypes.wit_title.value,
            ModalityTypes.wit_main_body.value,
            # ModalityTypes.youtube_video.value,
            # ModalityTypes.youtube_subtitles.value,
            # ModalityTypes.youtube_audio.value,
            # ModalityTypes.youtube_description.value,
        ],
        language_id="en",
        rng_seed=42,
        top_k_tali=10,
        image_size=224,
        transforms=None,
        num_video_frames=5,
        num_audio_frames=1 * 16000,
        clip_duration_in_seconds=1.5,
    )

    tali_dataset_image_text_config = dataset_config(
        set_name="train",
        root_filepath=DATASET_DIR,
        modality_list=[
            ModalityTypes.wit_image.value,
            ModalityTypes.wit_caption.value,
            ModalityTypes.wit_title.value,
            ModalityTypes.wit_main_body.value,
            # ModalityTypes.youtube_video.value,
            ModalityTypes.youtube_subtitles.value,
            # ModalityTypes.youtube_audio.value,
            ModalityTypes.youtube_description.value,
        ],
        language_id="en",
        rng_seed=42,
        top_k_tali=10,
        image_size=224,
        transforms=None,
        num_video_frames=5,
        num_audio_frames=1 * 16000,
        clip_duration_in_seconds=1.5,
    )

    tali_dataset_image_video_config = dataset_config(
        set_name="train",
        root_filepath=DATASET_DIR,
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
        language_id="en",
        rng_seed=42,
        top_k_tali=10,
        image_size=224,
        transforms=None,
        num_video_frames=5,
        num_audio_frames=1 * 16000,
        clip_duration_in_seconds=1.5,
    )

    tali_dataset_wit_image_audio_config = dataset_config(
        set_name="train",
        root_filepath=DATASET_DIR,
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
        language_id="en",
        rng_seed=42,
        top_k_tali=10,
        image_size=224,
        transforms=None,
        num_video_frames=5,
        num_audio_frames=1 * 16000,
        clip_duration_in_seconds=1.5,
    )

    tali_dataset_youtube_image_audio_config = dataset_config(
        set_name="train",
        root_filepath=DATASET_DIR,
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
        language_id="en",
        rng_seed=42,
        top_k_tali=10,
        image_size=224,
        transforms=None,
        num_video_frames=5,
        num_audio_frames=1 * 16000,
        clip_duration_in_seconds=1.5,
    )

    tali_dataset_text_audio_config = dataset_config(
        set_name="train",
        root_filepath=DATASET_DIR,
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
        language_id="en",
        rng_seed=42,
        top_k_tali=10,
        image_size=224,
        transforms=None,
        num_video_frames=5,
        num_audio_frames=1 * 16000,
        clip_duration_in_seconds=1.5,
    )

    tali_dataset_text_video_config = dataset_config(
        set_name="train",
        root_filepath=DATASET_DIR,
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
        language_id="en",
        rng_seed=42,
        top_k_tali=10,
        image_size=224,
        transforms=None,
        num_video_frames=5,
        num_audio_frames=1 * 16000,
        clip_duration_in_seconds=1.5,
    )

    tali_dataset_audio_video_config = dataset_config(
        set_name="train",
        root_filepath=DATASET_DIR,
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
        language_id="en",
        rng_seed=42,
        top_k_tali=10,
        image_size=224,
        transforms=None,
        num_video_frames=5,
        num_audio_frames=1 * 16000,
        clip_duration_in_seconds=1.5,
    )

    ###################################################################################

    config_store.store(
        group="model",
        name="tali_omni_base_patch16_224",
        node=tali_vit_model_config,
    )

    config_store.store(
        group="model",
        name="tali_image_text_base_patch16_224",
        node=tali_vit_image_text_model_config,
    )

    config_store.store(
        group="model",
        name="tali_image_audio_base_patch16_224",
        node=tali_vit_image_audio_model_config,
    )

    config_store.store(
        group="model",
        name="tali_image_video_base_patch16_224",
        node=tali_vit_image_video_model_config,
    )

    config_store.store(
        group="model",
        name="tali_image_text_audio_base_patch16_224",
        node=tali_vit_image_text_audio_model_config,
    )

    config_store.store(
        group="model",
        name="tali_image_text_video_base_patch16_224",
        node=tali_vit_image_text_video_model_config,
    )

    ###################################################################################

    config_store.store(
        group="dataset",
        name="wit_image_text_dataset",
        node={"128": wit_dataset_image_text_config},
    )

    config_store.store(
        group="dataset",
        name="tali_image_text_dataset",
        node={"128": tali_dataset_image_text_config},
    )

    config_store.store(
        group="dataset",
        name="tali_image_audio_dataset",
        node={
            "18": tali_dataset_youtube_image_audio_config,
            "18": tali_dataset_wit_image_audio_config,
        },
    )

    config_store.store(
        group="dataset",
        name="tali_image_video_dataset",
        node={
            "18": tali_dataset_image_video_config,
        },
    )

    config_store.store(
        group="dataset",
        name="tali_image_text_audio_dataset",
        node={
            "128": tali_dataset_image_text_config,
            "12": tali_dataset_audio_video_config,
            "16": tali_dataset_text_audio_config,
        },
    )

    config_store.store(
        group="dataset",
        name="tali_image_text_audio_video_dataset",
        node={
            "128": tali_dataset_image_text_config,
            "12": tali_dataset_audio_video_config,
            "16": tali_dataset_text_audio_config,
            "16": tali_dataset_text_video_config,
        },
    )

    config_store.store(
        group="dataset",
        name="tali_everything_dataset",
        node={
            "128": tali_dataset_image_text_config,
            "12": tali_dataset_audio_video_config,
            "20": tali_dataset_text_audio_config,
            "16": tali_dataset_text_video_config,
            "18": tali_dataset_youtube_image_audio_config,
            "18": tali_dataset_wit_image_audio_config,
            "24": tali_dataset_image_video_config,
        },
    )

    config_store.store(
        group="dataloader",
        name="default",
        node=dataloader_config(
            batch_size=TRAIN_BATCH_SIZE,
            num_workers=NUM_WORKERS,
            pin_memory=False,
            shuffle=True,
        ),
    )

    config_store.store(
        group="optimizer", name="adamw", node=adamw_optimizer_config(lr=1e-5)
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
        group="wandb_args", name="default", node=wandb_args_default
    )

    config_store.store(
        group="hydra",
        name="default",
        node=dict(
            job_logging=dict(
                version=1,
                formatters=dict(
                    simple=dict(
                        level=LOGGER_LEVEL,
                        format="%(message)s",
                        datefmt="[%X]",
                    )
                ),
                handlers=dict(
                    rich={
                        "class": "rich.logging.RichHandler",
                        "formatter": "simple",
                    }
                ),
                root={"handlers": ["rich"], "level": LOGGER_LEVEL},
                disable_existing_loggers=False,
            ),
            hydra_logging=dict(
                version=1,
                formatters=dict(
                    simple=dict(
                        level=LOGGER_LEVEL,
                        format="%(message)s",
                        datefmt="[%X]",
                    )
                ),
                handlers={
                    "rich": {
                        "class": "rich.logging.RichHandler",
                        "formatter": "simple",
                    }
                },
                root={"handlers": ["rich"], "level": LOGGER_LEVEL},
                disable_existing_loggers=False,
            ),
            run={
                "dir": "${current_experiment_dir}/hydra-run/${now:%Y-%m-%d_%H-%M-%S}"
            },
            sweep={
                "dir": "${current_experiment_dir}/hydra-multirun/${now:%Y-%m-%d_%H-%M-%S}",
                "subdir": "${hydra.job.num}",
            },
        ),
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
            dict(dataset="wit_image_text_dataset"),
            dict(dataloader="default"),
            dict(hydra="default"),
            dict(wandb_args="default"),
            dict(callbacks="default"),
        ],
    )
    # Config
    config_store.store(name="config", node=config)

    return config_store
