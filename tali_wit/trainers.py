from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, Tuple

import torch
import torch.nn.functional as F
import wandb
from accelerate import Accelerator
from hydra_zen import instantiate
from torch.utils.data import DataLoader

from tali_wit.callbacks import Interval
from tali_wit.models import extract_all_possible_pairs

from .decorators import collect_metrics
from .utils import get_logger

logger = get_logger(__name__)


def get_dict_shapes(x):
    if not isinstance(x, dict):
        return get_dict_shapes(x.__dict__)
    return {
        key: value.shape if isinstance(value, torch.Tensor) else len(value)
        for key, value in x.items()
    }


class Trainer(object):
    def __init__(self):
        pass


@dataclass
class TrainerOutput:
    opt_loss: torch.Tensor
    step_idx: int
    metrics: Dict[str, Any]
    phase_name: str


class ClassificationTrainer(Trainer):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler = None,
        scheduler_interval: str = Interval.STEP,
        experiment_tracker: wandb.wandb_sdk.wandb_run.Run = None,
    ):
        super().__init__()

        self.optimizer = optimizer
        self.scheduler = scheduler
        self.experiment_tracker = experiment_tracker
        self.epoch_metrics = {}

        if self.scheduler is not None:
            assert scheduler_interval in {"step", "epoch"}
            self.scheduler_interval = scheduler_interval

    def get_optimizer(self):
        return self.optimizer

    @collect_metrics
    def training_step(
        self,
        model,
        batch,
        batch_idx,
        step_idx,
        epoch_idx,
        accelerator: Accelerator,
    ) -> TrainerOutput:
        model.train()

        overall_loss = []
        overall_accuracy = []
        overall_accuracy_top_5 = []
        overall_output_dict = {}
        for (
            modality_a,
            sub_modality_a,
            modality_b,
            sub_modality_b,
        ) in extract_all_possible_pairs(batch):
            sample = {
                modality_a: {sub_modality_a: batch[modality_a][sub_modality_a]},
                modality_b: {sub_modality_b: batch[modality_b][sub_modality_b]},
            }
            self.optimizer.zero_grad()
            output_dict = model.forward(sample, return_loss=True)
            loss = torch.mean(
                torch.stack(
                    [
                        value
                        for key, value in output_dict.items()
                        if "_loss" in key
                    ]
                )
            )
            accuracy = torch.mean(
                torch.stack(
                    [
                        value.cpu()
                        for key, value in output_dict.items()
                        if "_accuracy" in key
                    ]
                )
            )
            accuracy_top_5 = torch.mean(
                torch.stack(
                    [
                        value.cpu()
                        for key, value in output_dict.items()
                        if "_accuracy_top_5" in key
                    ]
                )
            )
            accelerator.backward(loss)
            overall_output_dict |= output_dict
            overall_loss.append(loss)
            overall_accuracy.append(accuracy)
            overall_accuracy_top_5.append(accuracy_top_5)

        metrics = {
            "accuracy": torch.mean(overall_accuracy),
            "accuracy_top_5": torch.mean(overall_accuracy_top_5),
            "loss": torch.mean(overall_loss),
        }
        metrics |= overall_output_dict

        for key, value in metrics.items():
            self.epoch_metrics.setdefault(key, []).append(value.detach().cpu())

        metrics["lr"] = self.optimizer.param_groups[0]["lr"]

        return TrainerOutput(
            phase_name="training",
            opt_loss=torch.mean(overall_loss),
            step_idx=step_idx,
            metrics=metrics,
        )

    @collect_metrics
    def start_training(
        self,
        epoch_idx: int,
        step_idx: int,
        train_dataloader: DataLoader = None,
    ):
        self.epoch_metrics = {}
        return TrainerOutput(
            opt_loss=None, step_idx=step_idx, metrics={}, phase_name="training"
        )

    @collect_metrics
    def end_training(
        self,
        epoch_idx: int,
        step_idx: int,
        train_dataloader: DataLoader = None,
    ):
        epoch_metrics = {}
        for key, value in self.epoch_metrics.items():
            epoch_metrics[f"{key}-epoch-mean"] = torch.stack(value).mean()
            epoch_metrics[f"{key}-epoch-std"] = torch.stack(value).std()

        return TrainerOutput(
            opt_loss=None,
            step_idx=step_idx,
            metrics=epoch_metrics,
            phase_name="training",
        )
