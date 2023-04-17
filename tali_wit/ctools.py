from typing import Any, Dict, Optional
import torch
import torch.nn as nn
import accelerate
from tali_wit.data import get_base_modality

from tali_wit.models import extract_all_possible_pairs
from tali_wit.utils import get_logger

logger = get_logger(__name__)


def generate_hierarchical_data_dict(
    data_dict: Dict[str, Any]
) -> Dict[str, Any]:
    modality_hierarchical_output_dict = {}
    for sub_modality_name in list(data_dict.keys()):
        modality_type = get_base_modality(sub_modality_name)
        if modality_type is None:
            if "other" not in modality_hierarchical_output_dict:
                modality_hierarchical_output_dict["other"] = {}
            modality_hierarchical_output_dict["other"][
                sub_modality_name
            ] = data_dict[sub_modality_name]
            continue

        if modality_type not in modality_hierarchical_output_dict:
            modality_hierarchical_output_dict[modality_type.value] = {}

        modality_hierarchical_output_dict[modality_type.value][
            sub_modality_name
        ] = data_dict[sub_modality_name]
    return modality_hierarchical_output_dict


def dummy_fprop_bprop(
    model: nn.Module,
    batch: Dict[str, Any],
    accelerator: accelerate.Accelerator,
    do_bprop: bool = False,
) -> None:
    """
    Performs a forward and optional backward pass using the given model and batch.

    Args:
        model (nn.Module): The model to use for the forward and backward pass.
        batch (Dict[str, Any]): The input batch.
        accelerator (accelerate.Accelerator): The accelerator for model and batch preparation.
        do_bprop (bool, optional): If True, perform a backward pass. Defaults to False.
    """
    # Prepare the model and batch with the accelerator 🏎️
    model = accelerator.prepare(model)
    batch = accelerator.prepare(batch)

    # Generate hierarchical data dictionary 🌲
    batch = generate_hierarchical_data_dict(batch)

    # Iterate through all possible pairs in the batch 🔀
    for (
        modality_a,
        sub_modality_a,
        modality_b,
        sub_modality_b,
    ) in extract_all_possible_pairs(batch):
        # Prepare the sample 📦
        sample = {
            modality_a: {sub_modality_a: batch[modality_a][sub_modality_a]},
            modality_b: {sub_modality_b: batch[modality_b][sub_modality_b]},
        }
        # Forward pass and compute loss 📈
        output_dict = model.forward(sample, return_loss=True)
        loss = torch.mean(
            torch.stack(
                [value for key, value in output_dict.items() if "_loss" in key]
            )
        )

        # Perform backward pass if needed 🔄
        if do_bprop:
            accelerator.backward(loss)


def set_batch_size(batch: Dict[str, Any], batch_size: int) -> Dict[str, Any]:
    """
    Sets the batch size for a given input batch.

    Args:
        batch (Dict[str, Any]): The input batch.
        batch_size (int): The desired batch size.

    Returns:
        Dict[str, Any]: The input batch with the specified batch size.
    """
    return {
        key: value[0]
        .unsqueeze(0)
        .repeat([batch_size, *[1] * (value.dim() - 1)])
        if isinstance(value, torch.Tensor)
        else [value[0]] * batch_size
        for key, value in batch.items()
    }


def get_max_supported_batch_size(
    model: nn.Module,
    batch: Dict[str, Any],
    accelerator: Optional[accelerate.Accelerator] = None,
    train_mode: bool = False,
) -> int:
    """
    Finds the maximum supported batch size for the given model and batch.

    Args:
        model (nn.Module): The model to test.
        batch (Dict[str, Any]): The input batch.
        accelerator (Optional[accelerate.Accelerator], optional): The accelerator for model and batch preparation. Defaults to None.
        train_mode (bool, optional): If True, use training mode. Defaults to False.

    Returns:
        int: The maximum supported batch size.
    """
    # Create accelerator if not provided 🚀
    if accelerator is None:
        accelerator = accelerate.Accelerator()

    # Set model mode based on train_mode flag 🚦
    if train_mode:
        model.train()
    else:
        model.eval()

    batch_size = 16
    crashed = False

    for key, value in batch.items():
        print(key, value.shape if isinstance(value, torch.Tensor) else value)

    # Iterate and test different batch sizes until crash 🛠️
    while not crashed:
        logger.debug(f"Trying batch size {batch_size}... 👾")
        try:
            if batch is not None:
                cur_batch = set_batch_size(batch, batch_size)

                # Perform dummy forward and backward passes with the current batch size 📊
                if not train_mode:
                    with torch.no_grad():
                        dummy_fprop_bprop(model, cur_batch, accelerator)
                else:
                    dummy_fprop_bprop(
                        model, cur_batch, accelerator, do_bprop=True
                    )
        except Exception as e:
            crashed = True
            batch_size = batch_size // 2
            logger.debug(f"Batch size {batch_size} crashed with error: {e} 🤯")
            logger.info(f"Max supported batch size: {batch_size} 🎉")
            return batch_size

        # Double the batch size for the next iteration 📈
        batch_size = batch_size * 2
