import copy
import dataclasses
import datetime
import itertools
import os
from pathlib import Path
from typing import List

from rich import print


@dataclasses.dataclass
class ExperimentConfig:
    model: str
    dataset: str
    num_workers: int
    seed: int = 42


def get_scripts(exp_configs: List[ExperimentConfig]):
    script_list = []
    for exp_config in exp_configs:
        current_script_text = (
            f"/opt/conda/envs/main/bin/accelerate-launch --mixed_precision=bf16 "
            f"/app/tali_wit/run.py exp_name=tali-2-{exp_config.model}-{exp_config.dataset}-{exp_config.seed} "
            f"dataset={exp_config.dataset} model={exp_config.model} num_workers={exp_config.num_workers} seed={exp_config.seed} learner.train_iters=100000"
        )
        script_list.append(current_script_text)

    return script_list


if __name__ == "__main__":
    from bwatchcompute.kubernetes.job import ExperimentTemplate, Job

    exp_configs = [
        ExperimentConfig(
            model="tali_image_text_base_patch16_224",
            dataset="wit_image_text_dataset",
            num_workers=12,
        ),
        ExperimentConfig(
            model="tali_image_text_base_patch16_224",
            dataset="wit_tali_image_text_dataset",
            num_workers=6,
        ),
        ExperimentConfig(
            model="tali_image_text_audio_base_patch16_224",
            dataset="wit_tali_image_text_audio_dataset",
            num_workers=4,
        ),
        ExperimentConfig(
            model="tali_omni_base_patch16_224",
            dataset="wit_tali_image_text_audio_video_dataset",
            num_workers=3,
        ),
    ]

    script_list = get_scripts(exp_configs=exp_configs)
    # write a one liner that picks up date and time and converts them into a number
    datetime_seed = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

    exp = Job(
        name=f"{os.getenv('EXPERIMENT_NAME_PREFIX')}",
        script_list=script_list,
        docker_image_path=os.getenv("DOCKER_IMAGE_PATH"),
        secret_variables={
            "HF_TOKEN": os.getenv("EXPERIMENT_NAME_PREFIX"),
            "NEPTUNE_API_TOKEN": os.getenv("EXPERIMENT_NAME_PREFIX"),
            "WANDB_API_KEY": os.getenv("EXPERIMENT_NAME_PREFIX"),
        },
        environment_variables={
            "NEPTUNE_PROJECT": os.getenv("NEPTUNE_PROJECT"),
            "NEPTUNE_ALLOW_SELF_SIGNED_CERTIFICATE": os.getenv(
                "NEPTUNE_ALLOW_SELF_SIGNED_CERTIFICATE"
            ),
            "WANDB_PROJECT": os.getenv("WANDB_PROJECT"),
            "WANDB_ENTITY": os.getenv("WANDB_ENTITY"),
            "EXPERIMENT_NAME": os.getenv("EXPERIMENT_NAME"),
            "HF_USERNAME": os.getenv("HF_USERNAME"),
            "TOKENIZERS_PARALLELISM": os.getenv("TOKENIZERS_PARALLELISM"),
            "CODE_DIR": os.getenv("CODE_DIR"),
            "PROJECT_DIR": os.getenv("PROJECT_DIR"),
            "EXPERIMENT_NAME_PREFIX": os.getenv("EXPERIMENT_NAME_PREFIX"),
            "EXPERIMENTS_DIR": os.getenv("EXPERIMENTS_DIR"),
            "EXPERIMENT_DIR": os.getenv("EXPERIMENT_DIR"),
            "TALI_DATASET_DIR": os.getenv("TALI_DATASET_DIR"),
            "WIT_DATASET_DIR": os.getenv("WIT_DATASET_DIR"),
            "MODEL_DIR": os.getenv("MODEL_DIR"),
            "DOCKER_IMAGE_PATH": os.getenv("DOCKER_IMAGE_PATH"),
        },
        num_repeat_experiment=10,
        experiment_template=ExperimentTemplate.standard,
        persistent_disk_claim_names_to_mount_dict={
            "pvc-tali": "/tali-data/",
        },
        multi_persistent_disk_claim_names_to_mount_dict={
            "multi-pvc-wit": {
                "pvc-wit0": "/wit-data/",
                "pvc-wit1": "/wit-data/",
                "pvc-wit2": "/wit-data/",
                "pvc-wit2": "/wit-data/",
            },
        },
        num_gpus=1,
        image_pull_policy="Always",
        shm_size="167Gi",
    )

    exp.generate_spec_files()
    # output = exp.run_jobs()
    # print(output)
