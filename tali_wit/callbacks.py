import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .utils import get_logger

logger = get_logger(__name__)
hf_logger = get_logger("huggingface_hub", logging_level=logging.CRITICAL)


@dataclass
class Interval:
    EPOCH: str = "epoch"
    STEP: str = "step"


class Callback(object):
    def __init__(self) -> None:
        pass

    def on_init_start(
        self,
        experiment: Any,
        model: nn.Module,
        train_dataloaders: DataLoader = None,
        val_dataloaders: Union[List[DataLoader], DataLoader] = None,
        test_dataloaders: Union[List[DataLoader], DataLoader] = None,
    ) -> None:
        pass

    def on_init_end(
        self,
        experiment: Any,
        model: nn.Module,
        train_dataloaders: DataLoader = None,
        val_dataloaders: Union[List[DataLoader], DataLoader] = None,
        test_dataloaders: Union[List[DataLoader], DataLoader] = None,
    ) -> None:
        pass

    def on_epoch_start(
        self,
        experiment: Any,
        model: nn.Module,
        train_dataloaders: DataLoader = None,
        val_dataloaders: Union[List[DataLoader], DataLoader] = None,
        test_dataloaders: Union[List[DataLoader], DataLoader] = None,
    ) -> None:
        pass

    def on_epoch_end(
        self,
        experiment: Any,
        model: nn.Module,
        train_dataloaders: DataLoader = None,
        val_dataloaders: Union[List[DataLoader], DataLoader] = None,
        test_dataloaders: Union[List[DataLoader], DataLoader] = None,
    ) -> None:
        pass

    def on_batch_start(self, model: nn.Module, batch: Dict) -> None:
        pass

    def on_batch_end(self, model: nn.Module, batch: Dict) -> None:
        pass

    def on_training_step_start(self, model: nn.Module, batch: Dict) -> None:
        pass

    def on_training_step_end(self, model: nn.Module, batch: Dict) -> None:
        pass

    def on_validation_step_start(self, model: nn.Module, batch: Dict) -> None:
        pass

    def on_validation_step_end(self, model: nn.Module, batch: Dict) -> None:
        pass

    def on_testing_step_start(self, model: nn.Module, batch: Dict) -> None:
        pass

    def on_testing_step_end(self, model: nn.Module, batch: Dict) -> None:
        pass

    def on_train_start(
        self,
        experiment: Any,
        model: nn.Module,
    ) -> None:
        pass

    def on_train_end(
        self,
        experiment: Any,
        model: nn.Module,
    ) -> None:
        pass

    def on_validation_start(
        self,
        experiment: Any,
        model: nn.Module,
    ):
        pass

    def on_validation_end(
        self,
        experiment: Any,
        model: nn.Module,
    ) -> None:
        pass

    def on_testing_start(
        self,
        experiment: Any,
        model: nn.Module,
    ) -> None:
        pass

    def on_testing_end(
        self,
        experiment: Any,
        model: nn.Module,
    ) -> None:
        pass

    def on_save_checkpoint(
        self,
        model: nn.Module,
        optimizers: List[torch.optim.Optimizer],
        experiment: Any,
        checkpoint_path: Path,
    ) -> None:
        pass

    def on_load_checkpoint(
        self,
        model: nn.Module,
        optimizers: List[torch.optim.Optimizer],
        experiment: Any,
        checkpoint_path: Path,
    ) -> None:
        pass


class CallbackHandler(Callback):
    def __init__(self, callbacks: List[Callback]) -> None:
        super().__init__()
        self.callbacks = callbacks

    def on_init_start(
        self,
        experiment: Any,
        model: nn.Module,
        train_dataloaders: DataLoader = None,
        val_dataloaders: Union[List[DataLoader], DataLoader] = None,
        test_dataloaders: Union[List[DataLoader], DataLoader] = None,
    ) -> None:
        for callback in self.callbacks:
            callback.on_init_start(
                experiment,
                model,
                train_dataloaders,
                val_dataloaders,
                test_dataloaders,
            )

    def on_init_end(
        self,
        experiment: Any,
        model: nn.Module,
        train_dataloaders: DataLoader = None,
        val_dataloaders: Union[List[DataLoader], DataLoader] = None,
        test_dataloaders: Union[List[DataLoader], DataLoader] = None,
    ) -> None:
        for callback in self.callbacks:
            callback.on_init_end(
                experiment,
                model,
                train_dataloaders,
                val_dataloaders,
                test_dataloaders,
            )

    def on_epoch_start(
        self,
        experiment: Any,
        model: nn.Module,
        train_dataloader: DataLoader = None,
        val_dataloaders: Union[List[DataLoader], DataLoader] = None,
        test_dataloaders: Union[List[DataLoader], DataLoader] = None,
    ) -> None:
        for callback in self.callbacks:
            callback.on_epoch_start(
                experiment,
                model,
                train_dataloader,
                val_dataloaders,
                test_dataloaders,
            )

    def on_epoch_end(
        self,
        experiment: Any,
        model: nn.Module,
        train_dataloader: DataLoader = None,
        val_dataloaders: Union[List[DataLoader], DataLoader] = None,
        test_dataloaders: Union[List[DataLoader], DataLoader] = None,
    ) -> None:
        for callback in self.callbacks:
            callback.on_epoch_end(
                experiment,
                model,
                train_dataloader,
                val_dataloaders,
                test_dataloaders,
            )

    def on_batch_start(self, model: nn.Module, batch: Dict) -> None:
        for callback in self.callbacks:
            callback.on_batch_start(model, batch)

    def on_batch_end(self, model: nn.Module, batch: Dict) -> None:
        for callback in self.callbacks:
            callback.on_batch_end(model, batch)

    def on_training_step_start(self, model: nn.Module, batch: Dict) -> None:
        for callback in self.callbacks:
            callback.on_training_step_start(model, batch)

    def on_training_step_end(self, model: nn.Module, batch: Dict) -> None:
        for callback in self.callbacks:
            callback.on_training_step_end(model, batch)

    def on_validation_step_start(self, model: nn.Module, batch: Dict) -> None:
        for callback in self.callbacks:
            callback.on_validation_step_start(model, batch)

    def on_validation_step_end(self, model: nn.Module, batch: Dict) -> None:
        for callback in self.callbacks:
            callback.on_validation_step_end(model, batch)

    def on_testing_step_start(self, model: nn.Module, batch: Dict) -> None:
        for callback in self.callbacks:
            callback.on_testing_step_start(model, batch)

    def on_testing_step_end(self, model: nn.Module, batch: Dict) -> None:
        for callback in self.callbacks:
            callback.on_testing_step_end(model, batch)

    def on_train_start(
        self,
        experiment: Any,
        model: nn.Module,
    ) -> None:
        for callback in self.callbacks:
            callback.on_train_start(experiment, model)

    def on_train_end(
        self,
        experiment: Any,
        model: nn.Module,
    ) -> None:
        for callback in self.callbacks:
            callback.on_train_end(
                experiment,
                model,
            )

    def on_validation_start(
        self,
        experiment: Any,
        model: nn.Module,
    ):
        for callback in self.callbacks:
            callback.on_validation_start(experiment, model)

    def on_validation_end(
        self,
        experiment: Any,
        model: nn.Module,
    ) -> None:
        for callback in self.callbacks:
            callback.on_validation_end(experiment, model)

    def on_testing_start(
        self,
        experiment: Any,
        model: nn.Module,
    ) -> None:
        for callback in self.callbacks:
            callback.on_testing_start(experiment, model)

    def on_testing_end(
        self,
        experiment: Any,
        model: nn.Module,
    ) -> None:
        for callback in self.callbacks:
            callback.on_testing_end(experiment, model)

    def on_save_checkpoint(
        self,
        model: nn.Module,
        optimizers: List[torch.optim.Optimizer],
        experiment: Any,
        checkpoint_path: Path,
    ) -> None:
        for callback in self.callbacks:
            callback.on_save_checkpoint(model, optimizers, experiment, checkpoint_path)

    def on_load_checkpoint(
        self,
        model: nn.Module,
        optimizers: List[torch.optim.Optimizer],
        experiment: Any,
        checkpoint_path: Path,
    ) -> None:
        for callback in self.callbacks:
            callback.on_load_checkpoint(model, optimizers, experiment, checkpoint_path)


class UploadCheckpointToHuggingFaceBackground(threading.Thread):
    def __init__(self, repo_name: str, repo_owner: str, checkpoint_path: Path):
        from huggingface_hub import HfApi

        super().__init__()
        self.repo_name = repo_name
        self.repo_owner = repo_owner
        self.checkpoint_path = checkpoint_path
        self.hf_api = HfApi()
        self.done = False

    def run(self):
        try:
            self.hf_api.upload_folder(
                repo_id=f"{self.repo_owner}/{self.repo_name}",
                folder_path=self.checkpoint_path,
                path_in_repo=f"checkpoints/{self.checkpoint_path.name}",
            )

            self.done = True
        except Exception as e:
            logger.exception(e)
            self.done = True


class UploadCheckpointsToHuggingFace(Callback):
    def __init__(self, repo_name: str, repo_owner: str):
        from huggingface_hub import HfApi

        super().__init__()
        self.repo_name = repo_name
        self.repo_owner = repo_owner
        self.hf_api = HfApi()

    def on_save_checkpoint(
        self,
        model: nn.Module,
        optimizers: List[torch.optim.Optimizer],
        experiment: Any,
        checkpoint_path: Path,
    ) -> None:
        background_upload_thread = UploadCheckpointToHuggingFaceBackground(
            repo_name=self.repo_name,
            repo_owner=self.repo_owner,
            checkpoint_path=checkpoint_path,
        )
        experiment.background_threads.append(background_upload_thread)
