from ast import Dict
from dataclasses import dataclass
from typing import Any, Iterator, List, Tuple

import torch
import torch.nn.functional as F
import wandb
from accelerate import Accelerator
from attr import field
from hydra_zen import instantiate
from torch.utils.data import DataLoader

from tali_wit.models import extract_all_possible_pairs

from .decorators import collect_metrics
from .utils import get_logger

logger = get_logger(__name__)


def get_dict_shapes(x):
    return (
        {
            key: value.shape if isinstance(value, torch.Tensor) else len(value)
            for key, value in x.items()
        }
        if isinstance(x, dict)
        else get_dict_shapes(x.__dict__)
    )


class Evaluator(object):
    def __init__(self):
        pass


@dataclass
class EvaluatorOutput:
    step_idx: int
    metrics: Dict
    phase_name: str


class ClassificationEvaluator(Evaluator):
    def __init__(
        self, experiment_tracker: wandb.wandb_sdk.wandb_run.Run = None
    ):
        super().__init__()
        self.epoch_metrics = {}
        self.experiment_tracker = experiment_tracker

    def validation_step(
        self,
        model,
        batch,
        batch_idx,
        step_idx,
        epoch_idx,
        accelerator: Accelerator,
    ):
        model.eval()
        with torch.no_grad():
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
                    modality_a: {
                        sub_modality_a: batch[modality_a][sub_modality_a]
                    },
                    modality_b: {
                        sub_modality_b: batch[modality_b][sub_modality_b]
                    },
                }
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
                keys = list(output_dict.keys())
                for key in keys:
                    if "_loss" not in key and "_accuracy" not in key:
                        del output_dict[key]
                overall_output_dict |= output_dict
                overall_loss.append(loss)
                overall_accuracy.append(accuracy)
                overall_accuracy_top_5.append(accuracy_top_5)

            metrics = {
                "accuracy": torch.mean(torch.stack(overall_accuracy)),
                "accuracy_top_5": torch.mean(
                    torch.stack(overall_accuracy_top_5)
                ),
                "loss": torch.mean(torch.stack(overall_loss)),
            }
            metrics |= overall_output_dict

            for key, value in metrics.items():
                self.epoch_metrics.setdefault(key, []).append(value)

        return EvaluatorOutput(
            step_idx=step_idx,
            phase_name="validation",
            metrics=metrics,
        )

    def test_step(
        self,
        model,
        batch,
        batch_idx,
        step_idx,
        epoch_idx,
        accelerator: Accelerator,
    ):
        model.eval()
        with torch.no_grad():
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
                    modality_a: {
                        sub_modality_a: batch[modality_a][sub_modality_a]
                    },
                    modality_b: {
                        sub_modality_b: batch[modality_b][sub_modality_b]
                    },
                }
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
                keys = list(output_dict.keys())
                for key in keys:
                    if "_loss" not in key and "_accuracy" not in key:
                        del output_dict[key]
                overall_output_dict |= output_dict
                overall_loss.append(loss)
                overall_accuracy.append(accuracy)
                overall_accuracy_top_5.append(accuracy_top_5)

            metrics = {
                "accuracy": torch.mean(torch.stack(overall_accuracy)),
                "accuracy_top_5": torch.mean(
                    torch.stack(overall_accuracy_top_5)
                ),
                "loss": torch.mean(torch.stack(overall_loss)),
            }
            metrics |= overall_output_dict

            for key, value in metrics.items():
                self.epoch_metrics.setdefault(key, []).append(value)

        return EvaluatorOutput(
            step_idx=step_idx,
            phase_name="test",
            metrics=metrics,
        )

    @collect_metrics
    def start_validation(
        self,
        epoch_idx: int,
        step_idx: int,
        val_dataloaders: List[DataLoader] = None,
    ):
        self.epoch_metrics = {}
        return EvaluatorOutput(
            step_idx=step_idx,
            phase_name="validation",
            metrics=self.epoch_metrics,
        )

    @collect_metrics
    def start_testing(
        self,
        epoch_idx: int,
        step_idx: int,
        test_dataloaders: List[DataLoader] = None,
    ):
        self.epoch_metrics = {}
        return EvaluatorOutput(
            step_idx=step_idx, phase_name="testing", metrics=self.epoch_metrics
        )

    @collect_metrics
    def end_validation(
        self,
        epoch_idx: int,
        step_idx: int,
        val_dataloaders: List[DataLoader] = None,
    ):
        epoch_metrics = {}
        for key, value in self.epoch_metrics.items():
            epoch_metrics[f"{key}-epoch-mean"] = torch.stack(value).mean()
            epoch_metrics[f"{key}-epoch-std"] = torch.stack(value).std()

        return EvaluatorOutput(
            step_idx=step_idx, phase_name="validation", metrics=epoch_metrics
        )

    @collect_metrics
    def end_testing(
        self,
        epoch_idx: int,
        step_idx: int,
        test_dataloaders: List[DataLoader] = None,
    ):
        epoch_metrics = {}
        for key, value in self.epoch_metrics.items():
            epoch_metrics[f"{key}-epoch-mean"] = torch.stack(value).mean()
            epoch_metrics[f"{key}-epoch-std"] = torch.stack(value).std()

        return EvaluatorOutput(
            step_idx=step_idx, phase_name="testing", metrics=epoch_metrics
        )
