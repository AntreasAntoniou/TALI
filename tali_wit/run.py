import os

import neptune
from rich import print
from rich.traceback import install

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

# os.environ["TOKENIZERS_PARALLELISM"] = "false"

install()  # beautiful and clean tracebacks for debugging


from typing import List, Optional

import hydra
import torch
from hydra_zen import instantiate
from omegaconf import OmegaConf
from torch.utils.data import Dataset

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
    ckpt_path, repo_url = create_hf_model_repo_and_download_maybe(cfg)

    if ckpt_path is not None:
        logger.info(
            f"ckpt_path: {ckpt_path}, exists: {ckpt_path.exists()}, resume: {cfg.resume}, not resume: {not cfg.resume}"
        )
    else:
        logger.info(
            f"ckpt_path: {ckpt_path}, resume: {cfg.resume}, not resume: {not cfg.resume}"
        )

    logger.info(f"Using checkpoint: {ckpt_path}")

    config_dict = OmegaConf.to_container(cfg, resolve=True)
    experiment_tracker = neptune.init_run(
        source_files=["tali_wit/*.py", "kubernetes/*.py"]
    )
    experiment_tracker["config"] = config_dict
    experiment_tracker["notes"] = repo_url

    print(pretty_config(cfg, resolve=True))

    set_seed(seed=cfg.seed)

    model: TALIModel = instantiate(cfg.model)

    train_dataloaders = []
    val_dataloaders = []
    test_dataloaders = []

    for dataset_name, (batch_size, dataset) in cfg.dataset.items():
        logger.info(f"Setting up {dataset_name} train dataset")
        train_dataset: Dataset = instantiate(
            dataset, set_name="train", infinite_sampling=True
        )
        logger.info(f"Setting up {dataset_name} train dataloader")
        train_dataloader = instantiate(
            cfg.dataloader,
            dataset=train_dataset,
            batch_size=1,
            shuffle=True,
        )
        dummy_batch = next(iter(train_dataloader))
        logger.info(f"Finding max batch size for {dataset_name} train dataloader")
        optimal_batch_size = get_max_supported_batch_size(
            model=model, batch=dummy_batch
        )

        train_dataloader = instantiate(
            cfg.dataloader,
            dataset=train_dataset,
            batch_size=optimal_batch_size,
            shuffle=True,
        )

        train_dataloaders.append(train_dataloader)

    for dataset_name, (batch_size, dataset) in cfg.dataset.items():
        logger.info(f"Setting up {dataset_name} val dataset")
        val_dataset: Dataset = instantiate(dataset, set_name="val")
        val_dataloader = instantiate(
            cfg.dataloader,
            dataset=val_dataset,
            batch_size=1,
            shuffle=False,
        )

        dummy_batch = next(iter(val_dataloader))

        optimal_batch_size = get_max_supported_batch_size(
            model=model, batch=dummy_batch
        )

        val_dataloader = instantiate(
            cfg.dataloader,
            dataset=val_dataset,
            batch_size=optimal_batch_size,
            shuffle=False,
        )

        val_dataloaders.append(val_dataloader)

    for dataset_name, (batch_size, dataset) in cfg.dataset.items():
        logger.info(f"Setting up {dataset_name} test dataset")
        test_dataset: Dataset = instantiate(dataset, set_name="test")
        test_dataloader = instantiate(
            cfg.dataloader,
            dataset=test_dataset,
            batch_size=1,
            shuffle=False,
        )

        dummy_batch = next(iter(test_dataloader))

        optimal_batch_size = get_max_supported_batch_size(
            model=model, batch=dummy_batch
        )

        test_dataloader = instantiate(
            cfg.dataloader,
            dataset=test_dataset,
            batch_size=optimal_batch_size,
            shuffle=False,
        )

        test_dataloaders.append(test_dataloader)

    experiment_tracker["num_parameters"] = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )

    optimizer: torch.optim.Optimizer = instantiate(
        cfg.optimizer, params=model.parameters(), _partial_=False
    )

    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = instantiate(
        cfg.scheduler,
        optimizer=optimizer,
        t_initial=cfg.learner.train_iters,
        _partial_=False,
    )

    learner: Learner = instantiate(
        cfg.learner,
        model=model,
        trainers=[
            ClassificationTrainer(
                optimizer=optimizer,
                scheduler=scheduler,
                experiment_tracker=experiment_tracker,
            )
        ],
        evaluators=[ClassificationEvaluator(experiment_tracker=experiment_tracker)],
        train_dataloaders=train_dataloaders,
        val_dataloaders=val_dataloaders,
        callbacks=instantiate_callbacks(cfg.callbacks),
        resume=ckpt_path,
        experiment_tracker=experiment_tracker,
    )

    if cfg.train:
        learner.train()

    if cfg.test:
        learner.test(test_dataloaders=test_dataloaders)


if __name__ == "__main__":
    run()
