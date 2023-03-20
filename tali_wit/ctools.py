from typing import Any, Dict, Optional
import torch
import torch.nn as nn
import accelerate

from tali_wit.data_plus import generate_hierarchical_data_dict
from tali_wit.models import extract_all_possible_pairs
from tali_wit.utils import get_logger

logger = get_logger(__name__)


def dummy_fprop_bprop(
    model: nn.Module, batch: Dict[str, Any], accelerator: accelerate.Accelerator
):

    model = accelerator.prepare(model)
    batch = accelerator.prepare(batch)

    model.train()
    batch = generate_hierarchical_data_dict(batch)

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
        output_dict = model.forward(sample, return_loss=True)

        loss = torch.mean(
            torch.stack([value for key, value in output_dict.items() if "_loss" in key])
        )
        accelerator.backward(loss)


def set_batch_size(batch: Dict[str, Any], batch_size: int):

    return {
        key: value[0].unsqueeze(0).repeat([batch_size, *[1] * (value.dim() - 1)])
        if isinstance(value, torch.Tensor)
        else [value[0]] * batch_size
        for key, value in batch.items()
    }


def get_max_supported_batch_size(
    model: nn.Module,
    batch: Dict[str, Any],
    accelerator: Optional[accelerate.Accelerator] = None,
):

    if accelerator is None:
        accelerator = accelerate.Accelerator()

    batch_size = 2
    crashed = False
    while not crashed:
        logger.debug(f"Trying batch size {batch_size}... 👾")
        try:
            if batch is not None:
                cur_batch = set_batch_size(batch, batch_size)
                dummy_fprop_bprop(model, cur_batch, accelerator)
        except Exception as e:
            crashed = True
            batch_size = batch_size // 2
            logger.debug(f"Batch size {batch_size} crashed with error: {e} 🤯")
            logger.info(f"Max supported batch size: {batch_size} 🤯")
            return batch_size

        batch_size = batch_size * 2
