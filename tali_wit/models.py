import math
import os
from collections import defaultdict
from dataclasses import dataclass
import torch.nn.functional
from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from rich import print
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import ToTensor
from transformers import (
    CLIPModel,
    CLIPProcessor,
    WhisperModel,
    WhisperProcessor,
    WhisperFeatureExtractor,
)
from transformers.models.clip.modeling_clip import CLIPOutput, contrastive_loss

from tali_wit.data import ModalityTypes, TALIDataset, dataclass_collate
from tali_wit.decorators import configurable
from tali_wit.utils import get_logger

logger = get_logger(__name__, set_default_rich_handler=True)


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
    ):
        super().__init__()
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

    def init_weights(self):
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pos_encoder(x)
        x = self.transformer(x)
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
    similarities = {
        f"{modality_a_name}_{modality_b_name}_similarities": torch.einsum(
            "ijk,ilk->ijl", tensor_modality_a, tensor_modality_b
        ).sum(2)
        * logit_scale,
        f"{modality_b_name}_{modality_a_name}_similarities": torch.einsum(
            "ijk,ilk->ijl", tensor_modality_b, tensor_modality_a
        ).sum(2)
        * logit_scale,
    }
    if return_loss:
        contrastive_losses_dict = {
            f"{key.replace('_similarities', 'loss')}": contrastive_loss(value)
            for key, value in similarities.items()
        }
        return similarities | contrastive_losses_dict

    return similarities


@configurable
class TALIModel(nn.Module):
    def __init__(
        self,
        image_text_model_name: str = "openai/clip-vit-large-patch14",
        audio_model_name: str = "openai/whisper-small",
        multi_modality_config: MultiModalityConfig = MultiModalityConfig(),
        num_video_frames: int = 8,
        num_audio_frames: int = 8,
        audio_sampling_rate: int = 16000,
    ):
        super().__init__()

        self.image_text_model_name = image_text_model_name
        self.audio_model_name = audio_model_name
        self.multi_modality_config = multi_modality_config
        self.num_video_frames = num_video_frames
        self.num_audio_frames = num_audio_frames
        self.audio_sampling_rate = audio_sampling_rate

        self.build_model()
        self.build_transforms()

    def build_model(self):
        self.model = nn.ModuleDict()

        self.clip_model = CLIPModel.from_pretrained(self.image_text_model_name)

        self.model["image"] = self.clip_model.vision_model

        self.model["text"] = self.clip_model.text_model

        self.model["audio"] = WhisperModel.from_pretrained(
            self.audio_model_name
        ).encoder

        self.model["video"] = VideoTransformer(
            d_model=self.model["image"].config.hidden_size,
            nhead=8,
            dim_feedforward=self.model["image"].config.hidden_size * 4,
            dropout=0.0,
            num_layers=8,
            batch_first=True,
            norm_first=False,
            activation=F.gelu,
        )

        if (
            not self.multi_modality_config.image.support
            and not self.multi_modality_config.video.support
        ):
            delattr(self.model, "image")
        if not self.multi_modality_config.text.support:
            delattr(self.model, "text")
        if not self.multi_modality_config.audio.support:
            delattr(self.model, "audio")
        if not self.multi_modality_config.video.support:
            delattr(self.model, "video")

        if not self.multi_modality_config.image.pretrained:
            self.model["image"].init_weights()
        if not self.multi_modality_config.text.pretrained:
            self.model["text"].init_weights()
        if not self.multi_modality_config.audio.pretrained:
            self.model["audio"].init_weights()
        if not self.multi_modality_config.video.pretrained:
            self.model["video"].init_weights()

    def build_transforms(self):
        self.image_text_processor = CLIPProcessor.from_pretrained(
            self.image_text_model_name
        )
        self.audio_processor = WhisperProcessor.from_pretrained(
            self.audio_model_name
        )

        self.transforms = {
            "image": lambda x: self.image_text_processor(
                images=x, return_tensors="pt"
            ).pixel_values,
            "text": lambda x: self.image_text_processor(
                text=x, return_tensors="pt", padding=True, truncation=True
            ).input_ids,
            "audio": lambda x: torch.cat(
                [
                    self.audio_processor(
                        item,
                        sampling_rate=self.audio_sampling_rate,
                        return_tensors="pt",
                    ).input_features
                    for item in x.unbind(0)
                ]
            ),
        }

    def print_model_summary(self):
        # Print model layer summary
        logger.info(self)
        # Print total number of parameters
        logger.info(f"Total number of parameters: {num_parameters(self)}")

    def forward(
        self,
        x: Dict[str, torch.Tensor],
        return_features_only: bool = False,
        return_loss: bool = True,
    ) -> torch.Tensor:
        output_dict = defaultdict(dict)
        for modality in self.model.keys():
            if modality in x.keys():
                for sub_modality in x[modality].keys():
                    output_dict[modality][sub_modality] = getattr(
                        self, f"forward_{modality}"
                    )(x[modality][sub_modality])

        if return_features_only:
            return output_dict

        similarity_dict = {}
        for modality_a_name, modality_a in output_dict.items():
            for modality_b_name, modality_b in output_dict.items():
                if modality_a_name == modality_b_name:
                    continue
                similarity_dict.update(
                    get_similarities(
                        modality_a_name,
                        modality_b_name,
                        modality_a,
                        modality_b,
                        return_loss=return_loss,
                    )
                )

        return output_dict | similarity_dict

    def forward_image(self, x: torch.Tensor) -> torch.Tensor:
        input_shape = x.shape
        if "image" in self.transforms:
            x = self.transforms["image"](x.unbind(0))

        return self.model["image"](pixel_values=x.to(get_device()))

    def forward_text(self, x: torch.Tensor) -> torch.Tensor:
        if "text" in self.transforms:
            x = self.transforms["text"](x)
        return self.model["text"](x.to(get_device()))

    def forward_audio(self, x: torch.Tensor) -> torch.Tensor:
        if "audio" in self.transforms:
            x = self.transforms["audio"](x)
        x = x.to(get_device())
        return self.model["audio"](x)

    def forward_video(self, x: torch.Tensor) -> torch.Tensor:
        input_shape = (
            x.shape
        )  # (batch_size, num_frames, channels, height, width)

        if "video" in self.transforms:
            x = self.transforms["video"](x)
        out = x.to(get_device())
        out = self.forward_image(out.view(-1, *out.shape[-3:])).pooler_output
        out = out.view(input_shape[0], input_shape[1], -1).to(get_device())
        out = self.model["video"](out)
        return out


if __name__ == "__main__":
    # test all possible options

    import tqdm
    from rich import print
    from rich.traceback import install

    install()
    os.environ["HYDRA_FULL_ERROR"] = "1"

    dataset = TALIDataset(
        set_name="train",
        root_filepath="/data/datasets/tali-wit-2-1-buckets/",
        modality_list=[
            ModalityTypes.wit_image.value,
            ModalityTypes.wit_caption.value,
            ModalityTypes.wit_title.value,
            ModalityTypes.wit_main_body.value,
            ModalityTypes.youtube_video.value,
            ModalityTypes.youtube_subtitles.value,
            ModalityTypes.youtube_audio.value,
            ModalityTypes.youtube_description.value,
        ],
        language_id="en",
        rng_seed=42,
        top_k_tali=10,
        image_size=224,
        transforms=None,
        num_video_frames=5,
        num_audio_frames=1 * 16000,
        clip_duration_in_seconds=1.5,
    )

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=2,
        num_workers=2,
        shuffle=True,
        pin_memory=True,
        collate_fn=dataclass_collate,
    )

    dummy_batch = next(iter(dataloader))

    # build a TALI model with all modalities available
    model = TALIModel(
        image_text_model_name="openai/clip-vit-base-patch16",
        audio_model_name="openai/whisper-small",
        multi_modality_config=MultiModalityConfig(
            image=ModalityConfig(support=True, pretrained=True),
            text=ModalityConfig(support=True, pretrained=True),
            audio=ModalityConfig(support=True, pretrained=True),
            video=ModalityConfig(support=True, pretrained=False),
        ),
        num_video_frames=8,
        num_audio_frames=8,
        audio_sampling_rate=16000,
    )

    model = model.to(get_device())
    output_dict = model.forward(dummy_batch, return_loss=True)

    # # test no image
    # model = TALIModel(
    #     model_name="openai/clip-vit-large-patch14",
    #     pretrained=False,
    #     modality_config=ModalityConfig(
    #         image=False, text=True, audio=True, video=True
    #     ),
    # )
    # model.print_model_summary()

    # # test no text
    # model = TALIModel(
    #     model_name="openai/clip-vit-large-patch14",
    #     pretrained=False,
    #     modality_config=ModalityConfig(
    #         image=True, text=False, audio=True, video=True
    #     ),
    # )
    # model.print_model_summary()

    # # test no audio
    # model = TALIModel(
    #     model_name="openai/clip-vit-large-patch14",
    #     pretrained=False,
    #     modality_config=ModalityConfig(
    #         image=True, text=True, audio=False, video=True
    #     ),
    # )
    # model.print_model_summary()

    # # test no video
    # model = TALIModel(
    #     model_name="openai/clip-vit-large-patch14",
    #     pretrained=False,
    #     modality_config=ModalityConfig(
    #         image=True, text=True, audio=True, video=False
    #     ),
    # )
    # model.print_model_summary()

    # # test no image, no text
    # model = TALIModel(
    #     model_name="openai/clip-vit-large-patch14",
    #     pretrained=False,
    #     modality_config=ModalityConfig(
    #         image=False, text=False, audio=True, video=True
    #     ),
    # )
    # model.print_model_summary()

    # # test no image, no audio
    # model = TALIModel(
    #     model_name="openai/clip-vit-large-patch14",
    #     pretrained=False,
    #     modality_config=ModalityConfig(
    #         image=False, text=True, audio=False, video=True
    #     ),
    # )
    # model.print_model_summary()

    # # test no image, no video
    # model = TALIModel(
    #     model_name="openai/clip-vit-large-patch14",
    #     pretrained=False,
    #     modality_config=ModalityConfig(
    #         image=False, text=True, audio=True, video=False
    #     ),
    # )

    # # test pretrained, all modalities

    # model = TALIModel(
    #     model_name="openai/clip-vit-large-patch14",
    #     pretrained=True,
    #     modality_config=ModalityConfig(
    #         image=True, text=True, audio=True, video=True
    #     ),
    # )
    # model.print_model_summary()
