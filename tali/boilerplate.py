import itertools
import pathlib
import shutil
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, List, Union

import torch
import torch.nn as nn
from accelerate import Accelerator
from neptune import Run
from rich import print
from torch.utils.data import DataLoader
from tqdm import tqdm

from tali.callbacks import Callback, CallbackHandler, Interval
from tali.decorators import configurable
from tali.evaluators import Evaluator
from tali.trainers import Trainer
from tali.utils import get_logger

logger = get_logger(__name__)


def compare_models(model1, model2, optimizer1, optimizer2):
    """Compare parameters and optimizer states between two models."""

    # Compare model parameters
    for name1, param1 in model1.items():
        param2 = model2[name1]
        if not torch.equal(param1, param2):
            print(f"Parameter {name1} is not identical between models.")
            print("Model1 parameter:", param1)
            print("Model2 parameter:", param2)

    # Compare optimizer states
    for i, (opt_state1, opt_state2) in enumerate(
        zip(
            optimizer1["state"].values(),
            optimizer2["state"].values(),
        )
    ):
        for k1, k2 in zip(opt_state1.keys(), opt_state2.keys()):
            if not torch.equal(opt_state1[k1], opt_state2[k2]):
                print(
                    f"Optimizer state {k1} for parameter {i} is not identical between optimizers."
                )
                print("Optimizer1 state:", opt_state1[k1])
                print("Optimizer2 state:", opt_state2[k2])


def copy_optimizer_with_state(optimizer, model):
    """Copy optimizer with state."""
    # Get the type of the original optimizer
    OptimType = type(optimizer)

    # Get the arguments of the original optimizer
    args = optimizer.defaults

    # Create a new optimizer of the same type
    optimizer_copy = OptimType(model.parameters(), **args)

    # Copy the state of the original optimizer
    optimizer_copy.load_state_dict(optimizer.state_dict())
    return optimizer_copy


@configurable
class Learner(nn.Module):
    def __init__(
        self,
        accelerator: Accelerator,
        experiment_name: str,
        experiment_dir: Union[str, Path],
        model: torch.nn.Module,
        resume: Union[bool, str] = False,
        evaluate_every_n_steps: int = None,
        checkpoint_every_n_steps: int = None,
        checkpoint_after_validation: bool = False,
        train_iters: int = None,
        train_dataloader: DataLoader = None,
        limit_train_iters: int = None,
        val_dataloader: Union[List[DataLoader], DataLoader] = None,
        limit_val_iters: int = None,
        test_dataloader: Union[List[DataLoader], DataLoader] = None,
        trainer: Union[List[Trainer], Trainer] = None,
        evaluator: Union[List[Evaluator], Evaluator] = None,
        callbacks: Union[List[Callback], Callback] = None,
        print_model_parameters: bool = False,
        hf_cache_dir: str = None,
        hf_repo_path: str = None,
        experiment_tracker: Run = None,
        dummy_batch_mode: bool = False,
    ):
        super().__init__()
        self.accelerator = accelerator
        self.experiment_name = experiment_name
        self.experiment_dir = (
            experiment_dir if isinstance(experiment_dir, Path) else Path(experiment_dir)
        )
        self.hf_cache_dir = hf_cache_dir
        self.hf_repo_path = hf_repo_path
        self.background_threads = []
        self.checkpoints_dir = Path(self.experiment_dir / "checkpoints")
        self.neptune_run = experiment_tracker

        if not self.experiment_dir.exists():
            self.experiment_dir.mkdir(parents=True)

        if not self.checkpoints_dir.exists():
            self.checkpoints_dir.mkdir(parents=True)
        self.model = model
        self.evaluate_every_n_steps = evaluate_every_n_steps
        self.checkpoint_every_n_steps = checkpoint_every_n_steps
        self.checkpoint_after_validation = checkpoint_after_validation
        self.step_idx = 0
        self.epoch_idx = 0
        self.global_step = 0
        self.limit_train_iters = limit_train_iters
        self.limit_val_iters = limit_val_iters
        self.dummy_batch_mode = dummy_batch_mode

        self.train_iters = train_iters

        self.train_dataloader = train_dataloader

        self.val_dataloader = val_dataloader

        self.test_dataloader = test_dataloader

        for name, params in self.model.named_parameters():
            logger.info(f"{name}, {params.shape}")

        self.callbacks = [callbacks] if isinstance(callbacks, Callback) else callbacks

        if self.callbacks is None:
            self.callbacks = []

        self.callback_handler = CallbackHandler(self.callbacks)

        self.callback_handler.on_init_start(
            experiment=self,
            model=self.model,
            train_dataloader=self.train_dataloader,
            val_dataloader=self.val_dataloader,
            test_dataloader=self.test_dataloader,
        )

        self.resume = resume

        self.eval_mode = (
            Interval.STEP if self.train_iters is not None else Interval.EPOCH
        )

        if self.evaluate_every_n_steps is None:
            self.evaluate_every_n_steps = 99999999999

        self.trainer = trainer
        self.evaluator = evaluator

        self.callback_handler.on_init_end(
            experiment=self,
            model=self.model,
            train_dataloader=self.train_dataloader,
            val_dataloader=self.val_dataloader,
            test_dataloader=self.test_dataloader,
        )

        if isinstance(resume, str):
            checkpoint_path = Path(resume)
            if not checkpoint_path.exists():
                raise ValueError(
                    f"Checkpoint path {checkpoint_path} does not exist, please check your resume= argument"
                )
            self.load_checkpoint(checkpoint_path=checkpoint_path)

        elif isinstance(resume, Path):
            self.load_checkpoint(checkpoint_path=resume)

        if print_model_parameters:
            for key, value in self.named_parameters():
                logger.info(
                    f"Parameter {key} -> {value.shape} requires grad {value.requires_grad}"
                )

    def run(self):
        self.train()

    def forward(self, x):
        return self.model(x)

    def __repr__(self):
        attributes = "\n".join(
            [f"{key}={value}" for key, value in self.__dict__.items()]
        )
        return f"{self.__class__.__name__}\n {attributes}"

    def __str__(self):
        return self.__repr__()

    def training_step(self, model, batch):
        self.callback_handler.on_batch_start(model, batch)
        self.callback_handler.on_training_step_start(model, batch)
        output_list = []

        cur_output_dict = self.trainer.training_step(
            model=model,
            batch=batch,
            global_step=self.global_step,
            accelerator=self.accelerator,
        )
        output_list.append(cur_output_dict)

        self.callback_handler.on_batch_end(model, batch)
        self.callback_handler.on_training_step_end(model, batch)
        self.global_step += 1
        return output_list

    def validation_step(self, model, batch):
        self.callback_handler.on_batch_start(model, batch)
        self.callback_handler.on_validation_step_start(model, batch)

        self.evaluator.validation_step(
            model=model,
            batch=batch,
            global_step=self.global_step,
            accelerator=self.accelerator,
        )

        self.callback_handler.on_batch_end(model, batch)
        self.callback_handler.on_validation_step_end(model, batch)

    def testing_step(self, model, batch):
        self.callback_handler.on_batch_start(model, batch)
        self.callback_handler.on_testing_step_start(model, batch)

        self.evaluator.testing_step(
            model=model,
            batch=batch,
            global_step=self.global_step,
            accelerator=self.accelerator,
        )

        self.callback_handler.on_batch_end(model, batch)
        self.callback_handler.on_testing_step_end(model, batch)

    def start_training(self):
        self.callback_handler.on_train_start(
            experiment=self,
            model=self.model,
        )

        self.trainer.start_training(
            global_step=self.global_step,
        )

        logger.info("Starting training 🏋🏽")

    def end_training(self):
        self.callback_handler.on_train_end(
            experiment=self,
            model=self.model,
        )

        self.trainer.end_training(global_step=self.global_step)

        for background_thread in self.background_threads:
            background_thread.join()

        logger.info("Training finished 🎉")

    def check_manage_background_threads(self):
        # iterate threads to find up to where they are done, and start the next one
        for thread in self.background_threads:
            if not thread.done:
                if not thread.is_alive():
                    print(f"Starting thread {thread}")
                    thread.start()
                    break
            else:
                self.background_threads.remove(thread)
                print(f"Removing thread {thread} since it is done")

    def start_validation(self):
        self.callback_handler.on_validation_start(experiment=self, model=self.model)

        self.evaluator.start_validation(
            global_step=self.global_step,
        )

        logger.info("Starting validation 🧪")

    def end_validation(self):
        self.callback_handler.on_validation_end(experiment=self, model=self.model)

        self.evaluator.end_validation(
            global_step=self.global_step,
        )
        logger.info(f"{self.checkpoint_after_validation}")
        if self.checkpoint_after_validation:
            logger.info("Saving checkpoint after validation")
            self.save_checkpoint(checkpoint_name=f"ckpt_{self.global_step}")

        logger.info("Validation finished 🎉")

    def start_testing(self):
        self.callback_handler.on_testing_start(experiment=self, model=self.model)

        self.evaluator.start_testing(
            epoch_idx=self.epoch_idx,
            step_idx=self.global_step,
            global_step=self.global_step,
        )
        logger.info("Starting testing 🧪")

    def end_testing(self):
        self.callback_handler.on_testing_end(
            experiment=self,
            model=self.model,
        )

        self.evaluator.end_testing(
            global_step=self.global_step,
        )

        logger.info("Testing finished 🎉")

    def train(self, train_dataloader: DataLoader = None):
        if train_dataloader is not None:
            train_dataloader = self.accelerator.prepare(train_dataloader)
            self.train_dataloader = train_dataloader

        if self.dummy_batch_mode:
            self._dummy_training_loop(train_dataloader=train_dataloader)
        else:
            self._training_loop(train_dataloader=train_dataloader)

    def validate(
        self, val_dataloader: List[DataLoader] = None, model: nn.Module = None
    ):
        if val_dataloader is not None:
            self.val_dataloader = self.accelerator.prepare(val_dataloader)
            model = self.accelerator.prepare(model)
        self._validation_loop(val_dataloader=self.val_dataloader, model=model)

    def test(self, test_dataloader: List[DataLoader] = None):
        if test_dataloader is not None:
            self.test_dataloader = self.accelerator.prepare(test_dataloader)
            model = self.accelerator.prepare(self.model)
        self._testing_loop(
            test_dataloader=test_dataloader,
            model=model,
        )

    def _validation_loop(
        self, val_dataloader: List[DataLoader] = None, model: nn.Module = None
    ):
        if val_dataloader is None:
            val_dataloader = self.val_dataloader

        if model is None:
            model = self.model

        if val_dataloader is not None:
            self.start_validation()

            with tqdm(total=len(val_dataloader)) as pbar_dataloader:
                for batch_idx, batch in enumerate(val_dataloader):
                    if self.limit_val_iters is not None:
                        if batch_idx >= self.limit_val_iters:
                            break
                        if batch is not None:
                            self.validation_step(
                                model=self.model,
                                batch=batch,
                            )
                    pbar_dataloader.update(1)

            self.end_validation()

    def _testing_loop(
        self,
        test_dataloader: List[DataLoader] = None,
        model: nn.Module = None,
    ):
        if test_dataloader is None:
            test_dataloader = self.test_dataloader

        if model is None:
            model = self.model

        if test_dataloader is not None:
            self.start_testing()

            with tqdm(total=len(test_dataloader)) as pbar_dataloader:
                for batch_idx, batch in enumerate(test_dataloader):
                    if batch is not None:
                        self.testing_step(
                            model=self.model,
                            batch=batch,
                        )
                    pbar_dataloader.update(1)

            self.end_testing()

    def _training_loop(self, train_dataloader: DataLoader = None):
        if train_dataloader is None:
            train_dataloader = self.train_dataloader

        if train_dataloader is not None:
            self.start_training()

            if self.train_iters is None:
                self.train_iters = len(train_dataloader)

            with tqdm(initial=self.step_idx, total=self.train_iters) as pbar_steps:
                while self.step_idx < self.train_iters:
                    if self.limit_train_iters is not None:
                        if self.step_idx >= self.limit_train_iters:
                            return self.end_training()

                    for batch_idx, batch in enumerate(train_dataloader):
                        if batch is not None:
                            self.training_step(
                                model=self.model,
                                batch=batch,
                            )

                        if self.step_idx % self.evaluate_every_n_steps == 0:
                            self._validation_loop()
                            self.check_manage_background_threads()

                        if (
                            self.checkpoint_every_n_steps is not None
                            and self.step_idx % self.checkpoint_every_n_steps == 0
                            and self.step_idx > 0
                        ):
                            self.save_checkpoint(
                                checkpoint_name=f"ckpt_{self.global_step}"
                            )

                        if self.step_idx >= self.train_iters:
                            return self.end_training()

                        self.step_idx += 1
                        pbar_steps.update(1)

            return self.end_training()

    def save_checkpoint(
        self,
        checkpoint_name: str,
    ):
        ckpt_save_path = self.checkpoints_dir / checkpoint_name

        if not ckpt_save_path.exists():
            ckpt_save_path.mkdir(parents=True)

        experiment_hyperparameters = dict(
            step_idx=self.step_idx,
            epoch_idx=self.epoch_idx,
            global_step=self.global_step,
            state_dict={
                "train": self.trainer.state_dict,
                "eval": self.evaluator.state_dict,
            },
            neptune_id=self.neptune_run._id if self.neptune_run else None,
        )
        torch.save(
            obj=experiment_hyperparameters,
            f=ckpt_save_path / "trainer_state.pt",
        )
        save_location = self.accelerator.save_state(ckpt_save_path)

        # print(
        #     f"save_location: {save_location}, ckpt_save_path: {ckpt_save_path}"
        # )

        # save_state_snapshot = {
        #     "model": deepcopy(self.model.state_dict()),
        #     "optimizer": deepcopy(self.trainer.optimizer.state_dict()),
        # }

        # self.model = self.accelerator.prepare(self.dummy_model)
        # self.trainer.optimizer = self.accelerator.prepare(
        #     self.trainer.dummy_optimizer
        # )

        # self.accelerator.load_state(ckpt_save_path)

        # load_state_snapshot = {
        #     "model": self.model.state_dict(),
        #     "optimizer": self.trainer.optimizer.state_dict(),
        # }

        # compare_models(
        #     model1=save_state_snapshot["model"],
        #     model2=load_state_snapshot["model"],
        #     optimizer1=save_state_snapshot["optimizer"],
        #     optimizer2=load_state_snapshot["optimizer"],
        # )

        logger.info(f"Saved checkpoint to {save_location}")
        self.callback_handler.on_save_checkpoint(
            model=self.model,
            optimizers=self.trainer.optimizer,
            experiment=self,
            checkpoint_path=ckpt_save_path,
        )

        return ckpt_save_path

    def load_checkpoint(
        self,
        checkpoint_path: Union[str, Path],
    ):
        checkpoint_path = (
            checkpoint_path
            if isinstance(checkpoint_path, Path)
            else Path(checkpoint_path)
        )

        if not (pathlib.Path(checkpoint_path) / "trainer_state.pt").exists():
            return
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        trainer_state = torch.load(pathlib.Path(checkpoint_path) / "trainer_state.pt")
        self.step_idx = trainer_state["step_idx"]
        self.epoch_idx = trainer_state["epoch_idx"]
        self.global_step = trainer_state["global_step"]
        state_dict = trainer_state["state_dict"]

        setattr(
            self.trainer,
            "state_dict",
            state_dict["train"],
        )

        setattr(
            self.evaluator,
            "state_dict",
            state_dict["eval"],
        )

        # if (checkpoint_path / "optimizer.bin").exists():
        #     (checkpoint_path / "optimizer.bin").unlink()

        self.accelerator.load_state(checkpoint_path)

        # self.trainer.optimizer = self.accelerator.prepare(self.trainer.optimizer)

        self.callback_handler.on_load_checkpoint(
            model=self.model,
            optimizers=self.trainer.optimizer,
            experiment=self,
            checkpoint_path=checkpoint_path,
        )
