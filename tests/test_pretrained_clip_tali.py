import pytest
import torch
from transformers import CLIPModel, CLIPProcessor

from tali.models import ModalityConfig, MultiModalityConfig, TALIModel

image_text_model_name: str = "openai/clip-vit-base-patch16"

clip_model = CLIPModel.from_pretrained(image_text_model_name)
clip_preprocessor = CLIPProcessor.from_pretrained(image_text_model_name)
tali_model = TALIModel(image_text_model_name=image_text_model_name)

config = MultiModalityConfig(
    image=ModalityConfig(support=True, pretrained=False),
    text=ModalityConfig(support=True, pretrained=False),
)

tali_scratch_model = TALIModel(
    image_text_model_name, multi_modality_config=config
)


def test_compare_equal():
    image = torch.rand((5, 3, 224, 224))
    text = [
        "a photo of a cat",
        "a photo of a dog",
        "a photo of a human",
        "a photo of a car",
        "a photo of a plane",
    ]

    clip_image_input = clip_preprocessor(images=image, return_tensors="pt")
    clip_text_input = clip_preprocessor(
        text=text, return_tensors="pt", padding=True, truncation=True
    )

    clip_image_features = clip_model.get_image_features(**clip_image_input)
    clip_text_features = clip_model.get_text_features(**clip_text_input)

    tali_image_features = tali_model.forward_image(
        clip_image_input.pixel_values
    )["projection_output"]
    tali_text_features = tali_model.forward_text(clip_text_input.input_ids)[
        "projection_output"
    ]

    diff_image = torch.abs(clip_image_features - tali_image_features)
    diff_text = torch.abs(clip_text_features - tali_text_features)

    print(diff_image.sum(), diff_text.sum())

    assert diff_image.sum() == 0.0
    assert diff_text.sum() == 0.0


def test_compare_not_equal():
    image = torch.rand((5, 3, 224, 224))
    text = [
        "a photo of a cat",
        "a photo of a dog",
        "a photo of a human",
        "a photo of a car",
        "a photo of a plane",
    ]

    clip_image_input = clip_preprocessor(images=image, return_tensors="pt")
    clip_text_input = clip_preprocessor(
        text=text, return_tensors="pt", padding=True, truncation=True
    )

    clip_image_features = clip_model.get_image_features(**clip_image_input)
    clip_text_features = clip_model.get_text_features(**clip_text_input)

    tali_image_features = tali_scratch_model.forward_image(
        clip_image_input.pixel_values
    )["projection_output"]
    tali_text_features = tali_scratch_model.forward_text(
        clip_text_input.input_ids
    )["projection_output"]

    diff_image = torch.abs(clip_image_features - tali_image_features)
    diff_text = torch.abs(clip_text_features - tali_text_features)

    print(diff_image.sum(), diff_text.sum())

    assert diff_image.sum() > 0.0
    assert diff_text.sum() > 0.0
