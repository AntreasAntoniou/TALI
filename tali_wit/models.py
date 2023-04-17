import math
import os
from dataclasses import dataclass
import time
import torch.nn.functional
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from rich import print
from torch.utils.data import DataLoader
from transformers import (
    CLIPModel,
    CLIPProcessor,
    WhisperModel,
    WhisperProcessor,
)
from transformers.models.clip.modeling_clip import contrastive_loss

from accelerate import Accelerator
from tali_wit.data import ModalityTypes, TALIDataset, dataclass_collate
from tali_wit.decorators import configurable
from tali_wit.utils import get_logger

logger = get_logger(__name__)


@dataclass
class ModelAndTransform:
    model: nn.Module
    transform: Any


@dataclass
class ModalityConfig:
    support: bool = False
    pretrained: bool = False


@dataclass
class MultiModalityConfig:
    image: ModalityConfig = ModalityConfig(support=True, pretrained=True)
    text: ModalityConfig = ModalityConfig(support=True, pretrained=True)
    audio: ModalityConfig = ModalityConfig(support=True, pretrained=True)
    video: ModalityConfig = ModalityConfig(support=True, pretrained=True)


def contrastive_accuracy(logits):
    targets = torch.arange(logits.shape[0]).to(logits.device)
    return (logits.argmax(dim=-1) == targets).float().mean()


def contrastive_accuracy_top_k(logits, k: int = 5):
    targets = torch.arange(logits.shape[0]).to(logits.device)
    accuracy = [
        any(logit.argsort(dim=-1, descending=True)[:k] == target)
        for logit, target in zip(logits, targets)
    ]
    return torch.mean(torch.tensor(accuracy).float())


def num_parameters(
    model, only_trainable: bool = False, exclude_embeddings: bool = False
) -> int:
    """
    Get number of (optionally, trainable or non-embeddings) parameters in the module.

    Args:
        only_trainable (`bool`, *optional*, defaults to `False`):
            Whether or not to return only the number of trainable parameters

        exclude_embeddings (`bool`, *optional*, defaults to `False`):
            Whether or not to return only the number of non-embeddings parameters

    Returns:
        `int`: The number of parameters.
    """

    if exclude_embeddings:
        embedding_param_names = [
            f"{name}.weight"
            for name, module_type in model.named_modules()
            if isinstance(module_type, nn.Embedding)
        ]
        non_embedding_parameters = [
            parameter
            for name, parameter in model.named_parameters()
            if name not in embedding_param_names
        ]
        return sum(
            p.numel()
            for p in non_embedding_parameters
            if p.requires_grad or not only_trainable
        )
    else:
        return sum(
            p.numel()
            for p in model.parameters()
            if p.requires_grad or not only_trainable
        )


class PositionalEncoding(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        max_len = x.shape[1]
        d_model = x.shape[2]
        position = torch.arange(max_len).unsqueeze(1).to(x.device)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        ).to(x.device)
        pe = torch.zeros(1, max_len, d_model).to(x.device)
        pe[0, :, 0::2] = torch.sin(position * div_term).to(x.device)
        pe[0, :, 1::2] = torch.cos(position * div_term).to(x.device)
        x = x + pe[: x.size(0)]
        return x


@configurable
class VideoTransformer(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float,
        num_layers: int,
        batch_first: bool = True,
        norm_first: bool = True,
        activation: nn.Module = nn.GELU,
        output_linear_layer_dim: int = 512,
    ):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.norm_first = norm_first
        self.activation = activation
        self.output_linear_layer_dim = output_linear_layer_dim

        self.pos_encoder = PositionalEncoding()
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=batch_first,
            norm_first=norm_first,
        )
        self.transformer = nn.TransformerEncoder(
            num_layers=num_layers,
            encoder_layer=transformer_layer,
            norm=nn.LayerNorm(d_model),
        )
        self.output_norm = nn.LayerNorm(d_model)

    def init_weights(self):
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pos_encoder(x)
        x = self.transformer(x)[:, -1, :]  # take the last frame
        x = self.output_norm(x)
        return x


def get_device():
    return torch.cuda.current_device() if torch.cuda.is_available() else "cpu"


def get_similarities(
    modality_a_name: str,
    modality_b_name: str,
    tensor_modality_a: torch.Tensor,
    tensor_modality_b: torch.Tensor,
    logit_scale: torch.Tensor,
    return_loss: bool = False,
) -> torch.Tensor:
    """
    Args:
        tensor_modality_a: Tensor, shape [batch_size, seq_len, embedding_dim]
        tensor_modality_b: Tensor, shape [batch_size, seq_len, embedding_dim]
    """

    tensor_modality_a = tensor_modality_a.unsqueeze(0)
    tensor_modality_b = tensor_modality_b.unsqueeze(0)
    similarities = {
        f"{modality_a_name}_to_{modality_b_name}_similarities": torch.einsum(
            "ijk,ilk->ijl", tensor_modality_a, tensor_modality_b
        )[0]
        * logit_scale
    }

    similarities[
        f"{modality_b_name}_to_{modality_a_name}_similarities"
    ] = similarities[f"{modality_a_name}_to_{modality_b_name}_similarities"].T

    if return_loss:
        contrastive_losses_dict = {
            f"{key.replace('_similarities', '_loss')}": contrastive_loss(value)
            for key, value in similarities.items()
        }

        contrastive_accuracy_dict = {
            f"{key.replace('_similarities', '_accuracy')}": contrastive_accuracy(
                value
            )
            for key, value in similarities.items()
        }

        contrastive_accuracy_top_5_dict = {
            f"{key.replace('_similarities', '_accuracy_top_5')}": contrastive_accuracy_top_k(
                value, k=5
            )
            for key, value in similarities.items()
        }

        return (
            similarities
            | contrastive_losses_dict
            | contrastive_accuracy_dict
            | contrastive_accuracy_top_5_dict
        )

    return similarities


def reinit(input_module: nn.Module):
    for name, module in input_module.named_modules():
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, torch.nn.Embedding):
            torch.nn.init.normal_(module.weight, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, torch.nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, torch.nn.Conv2d):
            torch.nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, torch.nn.Conv1d):
            torch.nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, torch.nn.ConvTranspose1d):
            torch.nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)


@configurable
class TALIModel(nn.Module):
    def __init__(
        self,
        image_text_model_name: str = "openai/clip-vit-large-patch14",
        audio_model_name: str = "openai/whisper-small",
        multi_modality_config: MultiModalityConfig = MultiModalityConfig(),
    ):
        super().__init__()

        self.image_text_model_name = image_text_model_name
        self.audio_model_name = audio_model_name
        self.multi_modality_config = multi_modality_config

        self.build_model()
        self.build_logit_scales()

    def build_model(self):
        self.model = nn.ModuleDict()

        self.clip_model = CLIPModel.from_pretrained(self.image_text_model_name)
        logger.info(
            f"Attention here: {(not self.multi_modality_config.image.pretrained and self.multi_modality_config.image.support)} "
            f"and {(not self.multi_modality_config.text.pretrained and self.multi_modality_config.text.support)} "
            f"specifically {self.multi_modality_config.image.pretrained} and {self.multi_modality_config.text.pretrained} "
            f"and {self.multi_modality_config.image.support} and {self.multi_modality_config.text.support}"
        )
        if (
            not self.multi_modality_config.image.pretrained
            and self.multi_modality_config.image.support
        ) or (
            not self.multi_modality_config.text.pretrained
            and self.multi_modality_config.text.support
        ):
            logger.info("Reinitializing the image and text models")
            # for name, module in self.clip_model.named_modules():
            reinit(self.clip_model)

        self.linear_projection_dim = self.clip_model.projection_dim

        self.model["image"] = self.clip_model.vision_model
        self.image_linear_layer = self.clip_model.visual_projection

        self.model["text"] = self.clip_model.text_model
        self.text_linear_layer = self.clip_model.text_projection

        self.model["audio"] = WhisperModel.from_pretrained(
            self.audio_model_name
        )
        self.audio_output_shape = self.model["audio"].config.d_model
        self.model["audio"] = WhisperModel.from_pretrained(
            self.audio_model_name
        ).encoder

        self.audio_linear_layer = nn.Linear(
            in_features=self.audio_output_shape,
            out_features=self.linear_projection_dim,
            bias=False,
        )

        self.model["video"] = VideoTransformer(
            d_model=self.model["image"].config.projection_dim,
            nhead=8,
            dim_feedforward=self.model["image"].config.projection_dim * 4,
            dropout=0.0,
            num_layers=8,
            batch_first=True,
            norm_first=False,
            activation=F.gelu,
            output_linear_layer_dim=self.model["image"].config.projection_dim,
        )

        self.video_linear_layer = nn.Linear(
            self.model["video"].d_model, self.linear_projection_dim, bias=False
        )

        self.logit_init_value = float(
            self.clip_model.config.logit_scale_init_value
        )

        if (
            not self.multi_modality_config.image.support
            and not self.multi_modality_config.video.support
        ):
            delattr(self.model, "image")
            delattr(self, "image_linear_layer")
        if not self.multi_modality_config.text.support:
            delattr(self.model, "text")
            delattr(self, "text_linear_layer")
        if not self.multi_modality_config.audio.support:
            delattr(self.model, "audio")
            delattr(self, "audio_linear_layer")
        if not self.multi_modality_config.video.support:
            delattr(self.model, "video")
            delattr(self, "video_linear_layer")

        # if (
        #     not self.multi_modality_config.image.pretrained
        #     and self.multi_modality_config.image.support
        # ):
        #     self.model["image"].init_weights()
        # if (
        #     not self.multi_modality_config.text.pretrained
        #     and self.multi_modality_config.text.support
        # ):
        #     self.model["text"].init_weights()
        if (
            not self.multi_modality_config.audio.pretrained
            and self.multi_modality_config.audio.support
        ):
            reinit(self.model["audio"])
            logger.info("Reinitializing the audio model")
        if (
            not self.multi_modality_config.video.pretrained
            and self.multi_modality_config.video.support
        ):
            reinit(self.model["video"])
            logger.info("Reinitializing the video model")

    def build_logit_scales(self):
        self.logit_scales = nn.ParameterDict()

        self.logit_scales["image_to_text"] = nn.Parameter(
            torch.ones(1, requires_grad=True) * self.clip_model.logit_scale
        )
        self.logit_scales["text_to_image"] = nn.Parameter(
            torch.ones(1, requires_grad=True) * self.clip_model.logit_scale
        )

        for modality_a in self.model.keys():
            for modality_b in self.model.keys():
                if modality_a != modality_b:
                    name = f"{modality_a}_to_{modality_b}"
                    if name not in self.logit_scales.keys():
                        self.logit_scales[name] = nn.Parameter(
                            torch.ones(1, requires_grad=True)
                            * self.logit_init_value
                        )

    def print_model_summary(self):
        # Print model layer summary
        logger.debug(self)
        # Print total number of parameters
        logger.debug(f"Total number of parameters: {num_parameters(self)}")

    def forward(
        self,
        x: Dict[str, torch.Tensor],
        return_features_only: bool = False,
        return_loss: bool = True,
        restrict_source_modality: Optional[List[str]] = None,
    ) -> torch.Tensor:
        output_dict = {modality: {} for modality in self.model.keys()}
        for modality in self.model.keys():
            if modality in x:
                if restrict_source_modality is not None:
                    if modality not in restrict_source_modality:
                        continue
                for sub_modality in x[modality].keys():
                    output_dict[modality][sub_modality] = getattr(
                        self, f"forward_{modality}"
                    )(x[modality][sub_modality])

        if return_features_only:
            return output_dict

        similarity_dict = {}
        processed_pairs = set()
        for modality_a_name, modality_a in output_dict.items():
            for modality_b_name, modality_b in output_dict.items():
                if modality_a_name == modality_b_name:
                    continue
                for sub_modality_a_name, sub_modality_a in modality_a.items():
                    for (
                        sub_modality_b_name,
                        sub_modality_b,
                    ) in modality_b.items():
                        pair_name = f"{modality_a_name}_to_{modality_b_name}"
                        reverse_pair_name = (
                            f"{modality_b_name}_to_{modality_a_name}"
                        )
                        if (
                            pair_name in processed_pairs
                            or reverse_pair_name in processed_pairs
                        ):
                            continue
                        similarity_dict |= get_similarities(
                            sub_modality_a_name,
                            sub_modality_b_name,
                            sub_modality_a["projection_output"],
                            sub_modality_b["projection_output"],
                            logit_scale=self.logit_scales[
                                f"{modality_a_name}_to_{modality_b_name}"
                            ],
                            return_loss=return_loss,
                        )
                        processed_pairs.add(pair_name)

        return output_dict | similarity_dict

    def forward_image(self, x: torch.Tensor) -> torch.Tensor:
        cast_to_device_start_time = time.time()
        if len(x.shape) == 5:
            x = x.squeeze(1)
        x = x.to(self.image_linear_layer.weight.device)
        cast_to_device_end_time = time.time()
        logger.debug(
            f"Cast Image to device time: {cast_to_device_end_time - cast_to_device_start_time}"
        )

        fprop_actual_start_time = time.time()
        features = self.model["image"](pixel_values=x).pooler_output
        projection_output = self.image_linear_layer(features)
        fprop_actual_end_time = time.time()
        logger.debug(
            f"Fprop image actual time: {fprop_actual_end_time - fprop_actual_start_time}"
        )

        return {"features": features, "projection_output": projection_output}

    def forward_text(self, x: torch.Tensor) -> torch.Tensor:
        if len(x.shape) == 3:
            x = x.squeeze(1)
        cast_to_device_start_time = time.time()
        x = x.to(self.text_linear_layer.weight.device)
        cast_to_device_end_time = time.time()
        logger.debug(
            f"Cast Text to device time: {cast_to_device_end_time - cast_to_device_start_time}"
        )

        fprop_actual_start_time = time.time()
        features = self.model["text"](x).pooler_output
        projection_output = self.text_linear_layer(features)
        fprop_actual_end_time = time.time()
        logger.debug(
            f"Fprop text actual time: {fprop_actual_end_time - fprop_actual_start_time}"
        )
        return {"features": features, "projection_output": projection_output}

    def forward_audio(self, x: torch.Tensor) -> torch.Tensor:
        if len(x.shape) == 4:
            x = x.squeeze(1)
        cast_to_device_start_time = time.time()
        x = x.to(self.audio_linear_layer.weight.device)
        cast_to_device_end_time = time.time()
        logger.debug(
            f"cast audio to device time: {cast_to_device_end_time - cast_to_device_start_time}"
        )

        fprop_actual_start_time = time.time()
        features = self.model["audio"](x).last_hidden_state[:, -1, :]
        projection_output = self.audio_linear_layer(features)
        fprop_actual_end_time = time.time()
        logger.debug(
            f"Fprop audio actual time: {fprop_actual_end_time - fprop_actual_start_time}"
        )

        return {"features": features, "projection_output": projection_output}

    def forward_video(self, x: torch.Tensor) -> torch.Tensor:
        logger.debug("Video ---------------------------------")
        input_shape = (
            x.shape
        )  # (batch_size, num_frames, channels, height, width)

        out = x.to(self.video_linear_layer.weight.device)
        out = self.forward_image(out.view(-1, *out.shape[-3:]))[
            "projection_output"
        ]
        out = out.view(input_shape[0], input_shape[1], -1)
        features = self.model["video"](out)
        projection_output = self.video_linear_layer(features)
        return {"features": features, "projection_output": projection_output}


def extract_all_possible_pairs(batch_dict):
    from itertools import combinations

    modality_dict = {}
    for key, value in batch_dict.items():
        if isinstance(value, dict) and key != "other":
            modality_dict[key] = list(value.keys())

    pairs_keys = combinations(list(modality_dict.keys()), 2)

    # get all possible pairs of two lists
    pairs = []
    for key1, key2 in pairs_keys:
        for sub_key1, sub_key2 in zip(modality_dict[key1], modality_dict[key2]):
            pairs.append((key1, sub_key1, key2, sub_key2))

    return pairs


if __name__ == "__main__":
    # test all possible options

    import tqdm
    from rich import print
    from rich.traceback import install

    install()
    os.environ["HYDRA_FULL_ERROR"] = "1"

    # dummy_input = torch.randn((10000, 10000))
    # print(contrastive_accuracy(dummy_input))
    # print(contrastive_accuracy_top_k(dummy_input, k=5))

    # For 24GB GPU at 16-bit
    # 128 batch size for wit_image to text
    # 18 batch size for wit_image to youtube_audio
    # 24 batch size for wit_image to youtube_video
    # 20 batch size for text to youtube_audio
    # 16 batch size for text to youtube_video
    # 12 batch size for audio to youtube_video
    # set up a config that allows to select modality pairs + batch size and then build unique dataloaders for each pair
    # add accuracy and top-k accuracy metrics
