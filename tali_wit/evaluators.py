from ast import Dict
from dataclasses import dataclass
from typing import Any, List

import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader
from tali_wit.data_plus import *

from tali_wit.models import extract_all_possible_pairs
from tali_wit.trainers import StepOutput

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
    global_step: int
    metrics: Dict
    phase_name: str
    experiment_tracker: Any = None


class ClassificationEvaluator(Evaluator):
    def __init__(self, experiment_tracker: Any):
        super().__init__()
        self.state_dict = {}
        self.experiment_tracker = experiment_tracker

    def step(self, model, batch, global_step, accelerator: Accelerator):
        try:
            output_dict = model.forward(batch, return_loss=True)
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

            return StepOutput(
                output_dict=output_dict,
                loss=loss,
                accuracy=accuracy,
                accuracy_top_5=accuracy_top_5,
            )
        except Exception as e:
            logger.warning(f"Exception: {e}")
            # some_key = list(batch.keys())
            # some_sub_key = list(batch[some_key[0]].keys())
            # some_value = batch[some_key[0]][some_sub_key[0]]
            # if some_value.shape[0] > 1:
            #     smaller_batch_dict = {}

            #     for key, value in batch.items():
            #         smaller_batch_dict[key] = {}
            #         logger.info(f"{key}")
            #         for sub_key, sub_value in value.items():
            #             logger.info(
            #                 f"sub_key, sub_value.shape: {sub_key}, {sub_value.shape}"
            #             )
            #             smaller_batch_dict[key][sub_key] = sub_value[1:]
            #             logger.info(
            #                 f"smaller_batch_dict[key][sub_key].shape: {smaller_batch_dict[key][sub_key].shape}"
            #             )

            #     return self.step(model, smaller_batch_dict, global_step, accelerator)
            # else:
            return None

    @torch.inference_mode()
    def validation_step(
        self,
        model,
        batch,
        global_step,
        accelerator: Accelerator,
    ):
        model.eval()
        batch = generate_hierarchical_data_dict(batch)

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

                step_output: StepOutput = self.step(
                    model=model,
                    batch=sample,
                    global_step=global_step,
                    accelerator=accelerator,
                )

                keys = list(step_output.output_dict.keys())
                for key in keys:
                    if "_loss" not in key and "_accuracy" not in key:
                        del step_output.output_dict[key]

                if step_output is not None:
                    overall_output_dict |= step_output.output_dict
                    overall_loss.append(step_output.loss)
                    overall_accuracy.append(step_output.accuracy)
                    overall_accuracy_top_5.append(step_output.accuracy_top_5)

            if len(overall_loss) > 0:
                metrics = {
                    "accuracy": torch.mean(torch.stack(overall_accuracy)),
                    "accuracy_top_5": torch.mean(
                        torch.stack(overall_accuracy_top_5)
                    ),
                    "loss": torch.mean(torch.stack(overall_loss)),
                }
                metrics |= overall_output_dict
            else:
                metrics = {}

            for key, value in metrics.items():
                self.state_dict.setdefault(key, []).append(value)

        return EvaluatorOutput(
            global_step=global_step,
            phase_name="validation",
            metrics=metrics,
            experiment_tracker=self.experiment_tracker,
        )

    @torch.inference_mode()
    def test_step(
        self,
        model,
        batch,
        global_step,
        accelerator: Accelerator,
    ):
        model.eval()
        batch = generate_hierarchical_data_dict(batch)

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
                step_output: StepOutput = self.step(
                    model=model,
                    batch=sample,
                    global_step=global_step,
                    accelerator=accelerator,
                )

                keys = list(step_output.output_dict.keys())
                for key in keys:
                    if "_loss" not in key and "_accuracy" not in key:
                        del step_output.output_dict[key]

                if step_output is not None:
                    overall_output_dict |= step_output.output_dict
                    overall_loss.append(step_output.loss)
                    overall_accuracy.append(step_output.accuracy)
                    overall_accuracy_top_5.append(step_output.accuracy_top_5)

            if len(overall_loss) > 0:
                metrics = {
                    "accuracy": torch.mean(torch.stack(overall_accuracy)),
                    "accuracy_top_5": torch.mean(
                        torch.stack(overall_accuracy_top_5)
                    ),
                    "loss": torch.mean(torch.stack(overall_loss)),
                }
                metrics |= overall_output_dict
            else:
                metrics = {}

            for key, value in metrics.items():
                self.state_dict.setdefault(key, []).append(value)

        return EvaluatorOutput(
            global_step=global_step,
            phase_name="test",
            metrics=metrics,
            experiment_tracker=self.experiment_tracker,
        )

    @collect_metrics
    def start_validation(
        self,
        global_step: int,
    ):
        self.state_dict = {}
        return EvaluatorOutput(
            global_step=global_step,
            phase_name="validation",
            metrics=self.state_dict,
            experiment_tracker=self.experiment_tracker,
        )

    @collect_metrics
    def start_testing(
        self,
        global_step: int,
    ):
        self.state_dict = {}
        return EvaluatorOutput(
            global_step=global_step,
            phase_name="testing",
            metrics=self.state_dict,
            experiment_tracker=self.experiment_tracker,
        )

    @collect_metrics
    def end_validation(
        self,
        global_step: int,
    ):
        epoch_metrics = {}
        for key, value in self.state_dict.items():
            epoch_metrics[f"{key}-epoch-mean"] = torch.stack(value).mean()
            epoch_metrics[f"{key}-epoch-std"] = torch.stack(value).std()

        return EvaluatorOutput(
            global_step=global_step,
            phase_name="validation",
            metrics=epoch_metrics,
            experiment_tracker=self.experiment_tracker,
        )

    @collect_metrics
    def end_testing(
        self,
        global_step: int,
    ):
        epoch_metrics = {}
        for key, value in self.state_dict.items():
            epoch_metrics[f"{key}-epoch-mean"] = torch.stack(value).mean()
            epoch_metrics[f"{key}-epoch-std"] = torch.stack(value).std()

        return EvaluatorOutput(
            global_step=global_step,
            phase_name="testing",
            metrics=epoch_metrics,
            experiment_tracker=self.experiment_tracker,
        )
