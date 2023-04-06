import itertools
import pathlib
from pathlib import Path
import time
from typing import Any, List, Union
from neptune import Run

import torch
import torch.nn as nn
from accelerate import Accelerator, DistributedDataParallelKwargs
from torch.utils.data import DataLoader
from tqdm import tqdm

from tali_wit.callbacks import Callback, CallbackHandler, Interval
from tali_wit.decorators import configurable
from tali_wit.evaluators import ClassificationEvaluator, Evaluator
from tali_wit.trainers import (
    ClassificationTrainer,
    Trainer,
)
from tali_wit.utils import get_logger

logger = get_logger(__name__)

# silence logger for accelerate
accelerate_logger = get_logger("accelerate", logging_level="ERROR")


@configurable
class Learner(nn.Module):
    def __init__(
        self,
        experiment_name: str,
        experiment_dir: Union[str, Path],
        model: torch.nn.Module,
        resume: Union[bool, str] = False,
        evaluate_every_n_steps: int = None,
        checkpoint_every_n_steps: int = None,
        checkpoint_after_validation: bool = False,
        train_iters: int = None,
        train_dataloaders: DataLoader = None,
        limit_train_iters: int = None,
        val_dataloaders: Union[List[DataLoader], DataLoader] = None,
        limit_val_iters: int = None,
        test_dataloaders: Union[List[DataLoader], DataLoader] = None,
        trainers: Union[List[Trainer], Trainer] = None,
        evaluators: Union[List[Evaluator], Evaluator] = None,
        callbacks: Union[List[Callback], Callback] = None,
        print_model_parameters: bool = False,
        hf_cache_dir: str = None,
        hf_repo_path: str = None,
        experiment_tracker: Run = None,
        dummy_batch_mode: bool = False,
    ):
        super().__init__()
        self.experiment_name = experiment_name
        self.experiment_dir = (
            experiment_dir
            if isinstance(experiment_dir, Path)
            else Path(experiment_dir)
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
        self.checkpoint_every_n_steps = checkpoint_every_n_steps or 99999999999
        self.checkpoint_after_validation = checkpoint_after_validation
        self.step_idx = 0
        self.epoch_idx = 0
        self.global_step = 0
        self.limit_train_iters = limit_train_iters
        self.limit_val_iters = limit_val_iters
        self.dummy_batch_mode = dummy_batch_mode

        self.train_iters = train_iters

        self.train_dataloaders = train_dataloaders

        self.val_dataloaders = (
            [val_dataloaders]
            if isinstance(val_dataloaders, DataLoader)
            else val_dataloaders
        )

        self.test_dataloaders = (
            [test_dataloaders]
            if isinstance(test_dataloaders, DataLoader)
            else test_dataloaders
        )

        for name, params in self.model.named_parameters():
            logger.info(f"{name}, {params.shape}")

        self.callbacks = (
            [callbacks] if isinstance(callbacks, Callback) else callbacks
        )

        if self.callbacks is None:
            self.callbacks = []

        self.callback_handler = CallbackHandler(self.callbacks)

        self.callback_handler.on_init_start(
            experiment=self,
            model=self.model,
            train_dataloaders=self.train_dataloaders,
            val_dataloaders=self.val_dataloaders,
            test_dataloaders=self.test_dataloaders,
        )

        self.resume = resume

        self.eval_mode = (
            Interval.STEP if self.train_iters is not None else Interval.EPOCH
        )

        if self.evaluate_every_n_steps is None:
            self.evaluate_every_n_steps = 99999999999

        self.trainers = (
            [trainers] if isinstance(trainers, Trainer) else trainers
        )
        self.evaluators = (
            [evaluators] if isinstance(evaluators, Evaluator) else evaluators
        )

        self.callback_handler.on_init_end(
            experiment=self,
            model=self.model,
            train_dataloaders=self.train_dataloaders,
            val_dataloaders=self.val_dataloaders,
            test_dataloaders=self.test_dataloaders,
        )

        # use if you want to debug unused parameter errors in DDP
        self.accelerator = Accelerator(
            kwargs_handlers=[
                DistributedDataParallelKwargs(find_unused_parameters=True)
            ]
        )

        self.model = self.accelerator.prepare(self.model)

        for trainer in self.trainers:
            trainer.optimizer = self.accelerator.prepare(
                trainer.get_optimizer()
            )
            if trainer.scheduler is not None:
                trainer.scheduler = self.accelerator.prepare(trainer.scheduler)

        if self.train_dataloaders is not None:
            for i in range(len(self.train_dataloaders)):
                self.train_dataloaders[i] = self.accelerator.prepare(
                    self.train_dataloaders[i]
                )

        if self.val_dataloaders is not None:
            for i in range(len(self.val_dataloaders)):
                self.val_dataloaders[i] = self.accelerator.prepare(
                    self.val_dataloaders[i]
                )

        if self.test_dataloaders is not None:
            for i in range(len(self.test_dataloaders)):
                self.test_dataloaders[i] = self.accelerator.prepare(
                    self.test_dataloaders[i]
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

        for trainer in self.trainers:
            cur_output_dict = trainer.training_step(
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

        for evaluator in self.evaluators:
            evaluator.validation_step(
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

        for evaluator in self.evaluators:
            evaluator.testing_step(
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

        for trainer in self.trainers:
            trainer.start_training(
                global_step=self.global_step,
            )

        logger.info("Starting training 🏋🏽")

    def end_training(self):
        self.callback_handler.on_train_end(
            experiment=self,
            model=self.model,
        )

        for trainer in self.trainers:
            trainer.end_training(global_step=self.global_step)

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
        self.callback_handler.on_validation_start(
            experiment=self, model=self.model
        )

        for evaluator in self.evaluators:
            evaluator.start_validation(
                global_step=self.global_step,
            )

        logger.info("Starting validation 🧪")

    def end_validation(self):
        self.callback_handler.on_validation_end(
            experiment=self, model=self.model
        )

        for evaluator in self.evaluators:
            evaluator.end_validation(
                global_step=self.global_step,
            )
        logger.info(f"{self.checkpoint_after_validation}")
        if self.checkpoint_after_validation:
            logger.info("Saving checkpoint after validation")
            self.save_checkpoint(checkpoint_name=f"ckpt_{self.global_step}")

        logger.info("Validation finished 🎉")

    def start_testing(self):
        self.callback_handler.on_testing_start(
            experiment=self, model=self.model
        )

        for evaluator in self.evaluators:
            evaluator.start_testing(
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

        for evaluator in self.evaluators:
            evaluator.end_testing(
                global_step=self.global_step,
            )

        logger.info("Testing finished 🎉")

    def train(self, train_dataloaders: DataLoader = None):
        if train_dataloaders is not None:
            train_dataloaders = self.accelerator.prepare(train_dataloaders)
            self.train_dataloaders = train_dataloaders

        if self.dummy_batch_mode:
            self._dummy_training_loop(train_dataloaders=train_dataloaders)
        else:
            self._training_loop(train_dataloaders=train_dataloaders)

    def validate(
        self, val_dataloaders: List[DataLoader] = None, model: nn.Module = None
    ):
        if val_dataloaders is not None:
            self.val_dataloaders = []
            for val_dataloader in val_dataloaders:
                val_dataloader = self.accelerator.prepare(val_dataloader)
                self.val_dataloaders.append(val_dataloader)
            model = self.accelerator.prepare(model)
        self._validation_loop(val_dataloaders=val_dataloaders, model=model)

    def test(self, test_dataloaders: List[DataLoader] = None):
        if test_dataloaders is not None:
            self.test_dataloaders = []
            for test_dataloader in test_dataloaders:
                test_dataloader = self.accelerator.prepare(test_dataloader)
                self.test_dataloaders.append(test_dataloader)
            model = self.accelerator.prepare(model)
        self._testing_loop(
            test_dataloaders=test_dataloaders,
            model=model,
        )

    def _validation_loop(
        self, val_dataloaders: List[DataLoader] = None, model: nn.Module = None
    ):
        if val_dataloaders is None:
            val_dataloaders = self.val_dataloaders

        if model is None:
            model = self.model

        if val_dataloaders is not None:
            self.start_validation()

            with tqdm(
                total=max([len(d) for d in val_dataloaders])
            ) as pbar_dataloaders:
                for batch_idx, batch in enumerate(
                    itertools.zip_longest(*val_dataloaders)
                ):
                    if self.limit_val_iters is not None:
                        if batch_idx >= self.limit_val_iters:
                            break
                    for multi_modal_batch in batch:
                        if multi_modal_batch is not None:
                            self.validation_step(
                                model=self.model,
                                batch=multi_modal_batch,
                            )
                    pbar_dataloaders.update(1)

            self.end_validation()

    def _testing_loop(
        self,
        test_dataloaders: List[DataLoader] = None,
        model: nn.Module = None,
    ):
        if test_dataloaders is None:
            test_dataloaders = self.test_dataloaders

        if model is None:
            model = self.model

        if test_dataloader is not None:
            self.start_testing()

            with tqdm(
                total=max([len(d) for d in test_dataloaders])
            ) as pbar_dataloaders:
                for batch_idx, batch in enumerate(
                    itertools.zip_longest(*test_dataloaders)
                ):
                    for multi_modal_batch in batch:
                        if multi_modal_batch is not None:
                            self.testing_step(
                                model=self.model,
                                batch=multi_modal_batch,
                            )
                    pbar_dataloaders.update(1)
                    pbar_dataloaders.update(1)

            self.end_testing()

    def _dummy_training_loop(self, train_dataloaders: DataLoader = None):
        if train_dataloaders is None:
            train_dataloaders = self.train_dataloaders

        if train_dataloaders is not None:
            self.start_training(train_dataloaders=train_dataloaders)
            dummy_batch = next(iter(itertools.zip_longest(*train_dataloaders)))
            if self.train_iters is None:
                self.train_iters = len(train_dataloaders)

            with tqdm(
                initial=self.step_idx, total=self.train_iters
            ) as pbar_steps:
                while self.step_idx < self.train_iters:
                    if self.limit_train_iters is not None:
                        if self.step_idx >= self.limit_train_iters:
                            return self.end_training()

                    for i in range(self.train_iters):
                        loading_start_time = time.time()
                        for multi_modal_batch in dummy_batch:
                            if multi_modal_batch is not None:
                                loading_end_time = time.time()
                                loading_time_in_seconds = (
                                    loading_end_time - loading_start_time
                                )
                                logger.info(
                                    f"Loading time: {loading_time_in_seconds} seconds"
                                )
                                step_time_start = time.time()
                                self.training_step(
                                    model=self.model,
                                    batch=multi_modal_batch,
                                )
                                step_time_end = time.time()
                                logger.info(
                                    f"step time: {step_time_end - step_time_start} seconds"
                                )
                            loading_start_time = time.time()

                        if self.step_idx % self.evaluate_every_n_steps == 0:
                            self._validation_loop()
                            self.check_manage_background_threads()

                        if (
                            self.step_idx % self.checkpoint_every_n_steps == 0
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

    def _training_loop(self, train_dataloaders: DataLoader = None):
        if train_dataloaders is None:
            train_dataloaders = self.train_dataloaders

        if train_dataloaders is not None:
            self.start_training()

            if self.train_iters is None:
                self.train_iters = len(train_dataloaders)

            with tqdm(
                initial=self.step_idx, total=self.train_iters
            ) as pbar_steps:
                while self.step_idx < self.train_iters:
                    if self.limit_train_iters is not None:
                        if self.step_idx >= self.limit_train_iters:
                            return self.end_training()

                    for batch_idx, batch in enumerate(
                        itertools.zip_longest(*train_dataloaders)
                    ):
                        for multi_modal_batch in batch:
                            if multi_modal_batch is not None:
                                self.training_step(
                                    model=self.model,
                                    batch=multi_modal_batch,
                                )

                        if self.step_idx % self.evaluate_every_n_steps == 0:
                            self._validation_loop()
                            self.check_manage_background_threads()

                        if (
                            self.step_idx % self.checkpoint_every_n_steps == 0
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
                "train": [trainer.state_dict for trainer in self.trainers],
                "eval": [evaluator.state_dict for evaluator in self.evaluators],
            },
            neptune_id=self.neptune_run._id if self.neptune_run else None,
        )
        torch.save(
            obj=experiment_hyperparameters,
            f=ckpt_save_path / "trainer_state.pt",
        )
        self.accelerator.save_state(ckpt_save_path)
        logger.info(f"Saved checkpoint to {ckpt_save_path}")
        self.callback_handler.on_save_checkpoint(
            model=self.model,
            optimizers=[trainer.optimizer for trainer in self.trainers],
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
        trainer_state = torch.load(
            pathlib.Path(checkpoint_path) / "trainer_state.pt"
        )
        self.step_idx = trainer_state["step_idx"]
        self.epoch_idx = trainer_state["epoch_idx"]
        self.global_step = trainer_state["global_step"]
        state_dict = trainer_state["state_dict"]

        for trainer in self.trainers:
            setattr(
                trainer,
                "state_dict",
                state_dict["train"][self.trainers.index(trainer)],
            )

        for evaluator in self.evaluators:
            setattr(
                evaluator,
                "state_dict",
                state_dict["eval"][self.evaluators.index(evaluator)],
            )

        self.accelerator.load_state(checkpoint_path)

        self.callback_handler.on_load_checkpoint(
            model=self.model,
            optimizers=[trainer.get_optimizer() for trainer in self.trainers],
            experiment=self,
            checkpoint_path=checkpoint_path,
        )


if __name__ == "__main__":
    # a minimal example of how to use the Learner class
    import torch
    from datasets import load_dataset
    from rich import print
    from torch.nn import CrossEntropyLoss
    from torch.optim import Adam
    from torch.utils.data import DataLoader
    from torchvision.transforms import ColorJitter, Compose, Resize, ToTensor

    train_dataset = load_dataset("beans", split="train")
    val_dataset = load_dataset("beans", split="validation")
    test_dataset = load_dataset("beans", split="test")

    jitter = Compose(
        [
            Resize(size=(96, 96)),
            ColorJitter(brightness=0.5, hue=0.5),
            ToTensor(),
        ]
    )

    def transforms(examples):
        examples["pixel_values"] = [
            jitter(image.convert("RGB")) for image in examples["image"]
        ]
        return examples

    train_dataset = train_dataset.with_transform(transforms)
    val_dataset = val_dataset.with_transform(transforms)
    test_dataset = test_dataset.with_transform(transforms)

    def collate_fn(examples):
        images = []
        labels = []
        for example in examples:
            images.append((example["pixel_values"]))
            labels.append(example["labels"])

        pixel_values = torch.stack(images)
        labels = torch.tensor(labels)
        return {"pixel_values": pixel_values, "labels": labels}

    train_dataloader = DataLoader(
        train_dataset,
        collate_fn=collate_fn,
        batch_size=256,
        shuffle=True,
        num_workers=4,
    )
    val_dataloader = DataLoader(
        val_dataset, collate_fn=collate_fn, batch_size=256, num_workers=4
    )
    test_dataloader = DataLoader(
        test_dataset, collate_fn=collate_fn, batch_size=256, num_workers=4
    )

    model = torch.hub.load(
        "pytorch/vision:v0.9.0", "resnet18", pretrained=False
    )
    model.fc = torch.nn.Linear(512, 4)

    optimizer = Adam(model.parameters(), lr=1e-3)

    criterion = CrossEntropyLoss()

    experiment = Learner(
        experiment_name="debug_checkpointing",
        experiment_dir="/exp/debug_checkpointing",
        model=model,
        train_dataloaders=[train_dataloader],
        val_dataloaders=[val_dataloader],
        test_dataloaders=[test_dataloader],
        trainers=[ClassificationTrainer(optimizer=optimizer)],
        evaluators=[ClassificationEvaluator()],
        evaluate_every_n_steps=5,
        checkpoint_every_n_steps=5,
        checkpoint_after_validation=True,
        train_iters=1000,
        resume=True,
    )

    experiment.run()
