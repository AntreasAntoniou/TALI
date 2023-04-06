from dataclasses import dataclass
import time
from typing import Any, Dict

import torch
from accelerate import Accelerator

from tali_wit.callbacks import Interval
from tali_wit.data_plus import *
from tali_wit.models import extract_all_possible_pairs

from .decorators import collect_metrics
from .utils import get_logger

logger = get_logger(__name__)


def get_dict_shapes(x):
    print(x)
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
    global_step: int
    metrics: Dict[str, Any]
    phase_name: str
    experiment_tracker: Any = None


@dataclass
class StepOutput:
    output_dict: Dict
    loss: torch.Tensor
    accuracy: torch.Tensor
    accuracy_top_5: torch.Tensor


class ClassificationTrainer(Trainer):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler = None,
        scheduler_interval: str = Interval.STEP,
        experiment_tracker: Any = None,
    ):
        super().__init__()

        self.optimizer = optimizer
        self.scheduler = scheduler
        self.experiment_tracker = experiment_tracker
        self.state_dict = {}

        if self.scheduler is not None:
            assert scheduler_interval in {"step", "epoch"}
            self.scheduler_interval = scheduler_interval

    def get_optimizer(self):
        return self.optimizer

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

            accelerator.backward(loss)

            return StepOutput(
                output_dict=output_dict,
                loss=loss,
                accuracy=accuracy,
                accuracy_top_5=accuracy_top_5,
            )
        except Exception as e:
            logger.warning(e)
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

    @collect_metrics
    def training_step(
        self,
        model,
        batch,
        global_step,
        accelerator: Accelerator,
    ) -> TrainerOutput:
        model.train()

        overall_loss = []
        overall_accuracy = []
        overall_accuracy_top_5 = []
        overall_output_dict = {}
        generate_hierarchical_batch_start_time = time.time()
        batch = generate_hierarchical_data_dict(batch)
        generate_hierarchical_batch_end_time = time.time()
        logger.debug(
            f"generate_hierarchical_batch_time: {generate_hierarchical_batch_end_time - generate_hierarchical_batch_start_time}"
        )

        possible_pairs_start_time = time.time()
        possible_pairs = extract_all_possible_pairs(batch)
        possible_pairs_end_time = time.time()
        logger.debug(
            f"possible_pairs_time: {possible_pairs_end_time - possible_pairs_start_time}"
        )
        self.optimizer.zero_grad()

        for (
            modality_a,
            sub_modality_a,
            modality_b,
            sub_modality_b,
        ) in possible_pairs:
            sample = {
                modality_a: {sub_modality_a: batch[modality_a][sub_modality_a]},
                modality_b: {sub_modality_b: batch[modality_b][sub_modality_b]},
            }

            step_output: StepOutput = self.step(
                model=model,
                batch=sample,
                global_step=global_step,
                accelerator=accelerator,
            )
            if step_output is not None:
                overall_output_dict |= step_output.output_dict
                overall_loss.append(step_output.loss)
                overall_accuracy.append(step_output.accuracy)
                overall_accuracy_top_5.append(step_output.accuracy_top_5)

        self.optimizer.step()

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

        metrics["lr"] = self.optimizer.param_groups[0]["lr"]

        return TrainerOutput(
            phase_name="training",
            opt_loss=torch.mean(torch.stack(overall_loss))
            if len(overall_loss) > 0
            else None,
            global_step=global_step,
            metrics=metrics,
            experiment_tracker=self.experiment_tracker,
        )

    @collect_metrics
    def start_training(
        self,
        global_step: int,
    ):
        self.state_dict = {}
        return TrainerOutput(
            opt_loss=None,
            global_step=global_step,
            metrics={},
            phase_name="training",
            experiment_tracker=self.experiment_tracker,
        )

    @collect_metrics
    def end_training(
        self,
        global_step: int,
    ):
        epoch_metrics = {}
        for key, value in self.state_dict.items():
            epoch_metrics[f"{key}-epoch-mean"] = torch.stack(value).mean()
            epoch_metrics[f"{key}-epoch-std"] = torch.stack(value).std()

        return TrainerOutput(
            opt_loss=None,
            global_step=global_step,
            metrics=epoch_metrics,
            phase_name="training",
            experiment_tracker=self.experiment_tracker,
        )
