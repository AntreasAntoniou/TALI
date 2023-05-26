import time
from dataclasses import dataclass
from typing import Any, Dict

import torch
from rich import print
from rich import traceback
from accelerate import Accelerator

from tali.callbacks import Interval
from tali.data.data_plus import generate_hierarchical_data_dict
from tali.models import extract_all_possible_pairs

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


def collect_grad_stats(model):
    grad_stats = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            abs_mean = torch.mean(torch.abs(param.grad.detach().cpu())).item()
            std = torch.std(param.grad.detach().cpu()).item()
            grad_stats[name] = {"abs_mean": abs_mean, "std": std}
    return grad_stats


class ClassificationTrainer(Trainer):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler = None,
        scheduler_interval: str = Interval.STEP,
        experiment_tracker: Any = None,
        gradient_clipping: float = None,
    ):
        """
        Initializes the ClassificationTrainer class.

        Args:
            optimizer: The optimizer to be used during training.
            scheduler: The learning rate scheduler, defaults to None.
            scheduler_interval: Interval to update the learning rate, either "step" or "epoch".
            experiment_tracker: A tracker to log experimental details, defaults to None.
        """
        super().__init__()

        self.optimizer = optimizer
        self.scheduler = scheduler
        self.experiment_tracker = experiment_tracker
        self.gradient_clipping = gradient_clipping
        self.state_dict = {}

        if self.scheduler is not None:
            assert scheduler_interval in {"step", "epoch"}
            self.scheduler_interval = scheduler_interval

    @staticmethod
    def _calculate_metric_mean(
        output_dict: Dict[str, torch.Tensor], metric: str
    ) -> torch.Tensor:
        """
        Helper function to calculate the mean of a specific metric from the output dictionary.

        Args:
            output_dict: A dictionary containing output metrics.
            metric: The specific metric to calculate the mean for.

        Returns:
            The mean value of the specific metric.
        """
        return torch.mean(
            torch.stack(
                [
                    value.cpu()
                    for key, value in output_dict.items()
                    if metric in key
                ]
            )
        )

    def step(self, model, batch, global_step, accelerator: Accelerator):
        """
        Performs a single step in the training process.

        Args:
            model: The model to be trained.
            batch: The batch of data for this step.
            global_step: The current global step in the training process.
            accelerator: The PyTorch accelerator.

        Returns:
            A StepOutput object containing the output_dict, loss, accuracy, and accuracy_top_5.
        """
        try:
            output_dict = model.forward(batch, return_loss=True)
            loss = self._calculate_metric_mean(output_dict, "_loss")
            accuracy = self._calculate_metric_mean(output_dict, "_accuracy")
            accuracy_top_5 = self._calculate_metric_mean(
                output_dict, "_accuracy_top_5"
            )

            accelerator.backward(loss)

            if (
                accelerator.sync_gradients
                and self.gradient_clipping is not None
            ):
                accelerator.clip_grad_norm_(
                    model.parameters(), max_norm=self.gradient_clipping
                )

            return StepOutput(
                output_dict=output_dict,
                loss=loss,
                accuracy=accuracy,
                accuracy_top_5=accuracy_top_5,
            )
        except Exception as e:
            caught_exception = traceback.Traceback(e)
            logger.warning(f"Error in step: {e} {caught_exception}")
            return None

    @staticmethod
    def _generate_sample_from_batch(
        batch: Dict,
        modality_a: str,
        sub_modality_a: str,
        modality_b: str,
        sub_modality_b: str,
    ) -> Dict:
        """
        Helper function to generate a sample from a batch.

        Args:
            batch: The batch of data.
            modality_a: The first modality in the pair.
            sub_modality_a: The sub-modality related to the first modality.
            modality_b: The second modality in the pair.
            sub_modality_b: The sub-modality related to the second modality.

        Returns:
            A sample generated from the batch.
        """
        return {
            modality_a: {sub_modality_a: batch[modality_a][sub_modality_a]},
            modality_b: {sub_modality_b: batch[modality_b][sub_modality_b]},
        }

    @collect_metrics
    def training_step(
        self,
        model,
        batch,
        global_step,
        accelerator: Accelerator,
    ) -> TrainerOutput:
        """
        Performs a training step.

        Args:
            model: The model to be trained.
            batch: The batch of data for this step.
            global_step: The current global step in the training process.
            accelerator: The PyTorch accelerator.

        Returns:
            A TrainerOutput object containing the metrics and other details of the training step.
        """
        model.train()
        with accelerator.accumulate(model):
            overall_loss = []
            overall_accuracy = []
            overall_accuracy_top_5 = []
            overall_output_dict = {}
            batch = generate_hierarchical_data_dict(batch)

            possible_pairs = extract_all_possible_pairs(batch)
            self.optimizer.zero_grad()

            for (
                modality_a,
                sub_modality_a,
                modality_b,
                sub_modality_b,
            ) in possible_pairs:
                sample = self._generate_sample_from_batch(
                    batch,
                    modality_a,
                    sub_modality_a,
                    modality_b,
                    sub_modality_b,
                )

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
            grad_dict = None
            if global_step % 100 == 0:
                grad_dict = collect_grad_stats(model)

            self.optimizer.step()

        metrics = {}
        if overall_loss:
            metrics = {
                "accuracy": torch.mean(torch.stack(overall_accuracy)),
                "accuracy_top_5": torch.mean(
                    torch.stack(overall_accuracy_top_5)
                ),
                "loss": torch.mean(torch.stack(overall_loss)),
                **overall_output_dict,
            }

        for key, value in metrics.items():
            self.state_dict.setdefault(key, []).append(value)

        metrics["lr"] = self.optimizer.param_groups[0]["lr"]
        if grad_dict is not None:
            metrics.update(grad_dict)

        return TrainerOutput(
            phase_name="training",
            opt_loss=torch.mean(torch.stack(overall_loss))
            if overall_loss
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
