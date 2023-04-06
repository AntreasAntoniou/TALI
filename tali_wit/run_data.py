import os
import pathlib
import time

import neptune
from rich import print
from rich.traceback import install
import tqdm
from tali_wit.data import dataclass_collate
from tali_wit.data_plus import CustomConcatDataset

from tali_wit.models import TALIModel
from tali_wit.utils import (
    create_hf_model_repo_and_download_maybe,
)
from tali_wit.ctools import get_max_supported_batch_size

os.environ[
    "HYDRA_FULL_ERROR"
] = "1"  # Makes sure that stack traces produced by hydra instantiation functions produce
# traceback errors related to the modules they built, rather than generic instantiate related errors that
# are generally useless for debugging

os.environ[
    "TORCH_DISTRIBUTED_DEBUG"
] = "DETAIL"  # extremely useful when debugging DDP setups

install()  # beautiful and clean tracebacks for debugging


from typing import List, Optional

import hydra
import torch
from hydra_zen import instantiate
from omegaconf import OmegaConf
from torch.utils.data import Dataset, Subset

from tali_wit.boilerplate import Learner
from tali_wit.callbacks import Callback
from tali_wit.config import BaseConfig, collect_config_store
from tali_wit.evaluators import ClassificationEvaluator
from tali_wit.trainers import ClassificationTrainer
from tali_wit.utils import get_logger, pretty_config, set_seed

config_store = collect_config_store()

logger = get_logger(name=__name__)


def instantiate_callbacks(callback_dict: dict) -> List[Callback]:
    callbacks = []
    for cb_conf in callback_dict.values():
        callbacks.append(instantiate(cb_conf))

    return callbacks


@hydra.main(config_path=None, config_name="config", version_base=None)
def run(cfg: BaseConfig) -> None:
    print(pretty_config(cfg, resolve=True))

    set_seed(seed=cfg.seed)

    train_datasets = []

    for dataset_name, (batch_size, dataset) in cfg.dataset.items():
        logger.info(f"Setting up {dataset_name} train dataset")

        train_dataset: Dataset = instantiate(
            dataset,
            set_name="train",
            infinite_sampling=True,
            num_samples_per_episode=batch_size,
        )

        train_datasets.append(train_dataset)

    train_dataset = CustomConcatDataset(train_datasets)

    train_dataloader = instantiate(
        cfg.dataloader,
        dataset=train_dataset,
        batch_size=1,
        shuffle=True,
        collate_fn=dataclass_collate,
    )

    with tqdm.tqdm(total=len(train_dataloader)) as pbar:
        for batch in train_dataloader:
            time.sleep(1)
            pbar.update(1)


if __name__ == "__main__":
    run()
