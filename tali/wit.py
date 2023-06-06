import os
import pathlib
import time
from typing import Any, Callable, Dict, Optional
import PIL

import datasets
import numpy as np
import torch
import tqdm
from torch.utils.data import DataLoader, Dataset, Subset
from transformers import CLIPProcessor, WhisperProcessor
import multiprocessing as mp
from tali.data.data import (
    ModalityTypes,
    dataclass_collate,
    default_image_transforms,
)
from tali.data.data_plus import get_next_on_error, get_submodality_name
from tali.decorators import configurable
from tali.utils import get_logger, load_json, save_json

logger = get_logger(__name__)


@configurable
class WITBase(Dataset):
    def __init__(
        self,
        wit_dataset_dir: str,
        tali_dataset_dir: str,
        image_size: int,
        set_name: str,
        num_samples_per_episode: int,
        deterministic_sampling: bool = False,
        total_num_samples: Optional[int] = None,
        priority_caption_language: Optional[str] = None,
        dummy_batch_mode: bool = False,
        image_text_model_name: str = "openai/clip-vit-base-patch32",
        audio_model_name: str = "openai/whisper",
    ):
        super().__init__()
        self.wit_dataset_dir = wit_dataset_dir
        self.image_size = image_size
        self.wit_transform = WITBaseTransform(
            image_size=image_size,
            priority_caption_language=priority_caption_language,
            deterministic_sampling=deterministic_sampling,
        )
        self.dataset = datasets.load_dataset(
            "wikimedia/wit_base",
            split="train",
            cache_dir=os.environ["HF_CACHE_DIR"],
            num_proc=mp.cpu_count(),
        )
        self.num_samples_per_episode = num_samples_per_episode
        self.indices_filepath = (
            pathlib.Path(wit_dataset_dir) / "wit_indices.json"
        )

        if not self.indices_filepath.exists():
            tali_val_dataset = datasets.load_dataset(
                path="Antreas/TALI",
                split="val",
                keep_in_memory=False,
                cache_dir=os.environ["HF_CACHE_DIR"],
            )
            tali_val_indices = [
                sample["wit_idx"] for sample in tali_val_dataset
            ]

            tali_test_dataset = datasets.load_dataset(
                path="Antreas/TALI",
                split="test",
                keep_in_memory=False,
                cache_dir=os.environ["HF_CACHE_DIR"],
            )
            tali_test_indices = [
                sample["wit_idx"] for sample in tali_test_dataset
            ]

            train_wit_indices = []
            with tqdm.tqdm(total=len(self.dataset)) as pbar:
                for i in range(len(self.dataset)):
                    if (
                        i not in tali_val_indices
                        and i not in tali_test_indices
                    ):
                        train_wit_indices.append(i)
                    pbar.update(1)

            self.indices = {
                "train": train_wit_indices,
                "val": tali_val_indices,
                "test": tali_test_indices,
            }
            save_json(
                filepath=os.path.join(
                    self.wit_dataset_dir, "wit_indices.json"
                ),
                dict_to_store=self.indices,
            )
        else:
            self.indices = load_json(self.indices_filepath)

        self.dataset = Subset(self.dataset, self.indices[set_name])
        self.dummy_batch_mode = dummy_batch_mode
        self.dummy_batch = None

        self.num_samples = (
            total_num_samples
            if total_num_samples is not None
            else len(self.dataset)
        )
        self.image_text_model_name = image_text_model_name
        self.audio_model_name = audio_model_name
        self.dataset_size = len(self.dataset)
        self.transforms = self.build_transforms()

    def build_transforms(self) -> Dict[str, Callable]:
        """
        Build a dictionary of transformation functions for each modality of data (image, text, audio, video)
        using model-specific preprocessing steps.

        Returns:
            A dictionary of transformations with keys as modality names and values as transformation functions.
        """
        # Get processors from the models
        self.image_text_processor = CLIPProcessor.from_pretrained(
            self.image_text_model_name
        )
        self.audio_processor = WhisperProcessor.from_pretrained(
            self.audio_model_name
        )

        def image_transforms(x):
            if isinstance(x, PIL.Image.Image):
                temp_x = np.array(x)
                if temp_x.max() > 255:
                    temp_x = temp_x / 65535.0
                    temp_x = (temp_x * 255).astype(np.uint8)
                    x = PIL.Image.fromarray(temp_x)

            return self.image_text_processor(
                images=x, return_tensors="pt"
            ).pixel_values.squeeze()

        def text_transforms(x):
            return self.image_text_processor(
                text=x, return_tensors="pt", padding=True, truncation=True
            ).input_ids.squeeze(0)

        def audio_transforms(x):
            return torch.cat(
                [
                    self.audio_processor(
                        item.view(-1),
                        sampling_rate=16000,
                        return_tensors="pt",
                    ).input_features
                    for item in x.unbind(0)
                ]
            )

        def video_transforms(x):
            return torch.stack(
                [image_transforms(image) for image in x],
                dim=0,
            )

        # Return a dictionary of transformations
        return {
            "image": image_transforms,
            "text": text_transforms,
            "audio": audio_transforms,
            "video": video_transforms,
        }

    def __getitem__(self, idx):
        episode_dict = {}

        for i in range(idx, idx + self.num_samples_per_episode):
            sample = self.get_sample(idx=i)
            for key, value in sample.items():
                # 📝 If text value is too short, pad with pad_token_id
                if (
                    "text" in key
                    and len(value) < 77
                    and isinstance(value, torch.Tensor)
                ):
                    value = torch.cat(
                        [
                            value,
                            self.image_text_processor.tokenizer.pad_token_id
                            * torch.ones(77 - len(value)).long(),
                        ]
                    )
                # 🔑 If key is new, initialize it in the dictionary
                if key not in episode_dict:
                    episode_dict[key] = [value]
                else:
                    episode_dict[key].append(value)

        # 🔄 Convert value lists into tensors if possible
        for key, value in episode_dict.items():
            episode_dict[key] = (
                torch.stack(value, dim=0).squeeze()
                if isinstance(value[0], torch.Tensor)
                else value.squeeze()
                if isinstance(value, torch.Tensor)
                else value
            )

        return episode_dict

    @get_next_on_error
    def get_sample(self, idx):
        if idx >= self.dataset_size:
            idx = idx % self.dataset_size

        sample = self.dataset[idx] | {"wit_idx": idx}
        sample = self.wit_transform(sample)

        for key, value in sample.items():
            for transform_key, transform_value in self.transforms.items():
                if transform_key in key:
                    sample[key] = transform_value(value)
                    break

        return sample

    def __len__(self):
        return self.num_samples


class WITBaseTransform:
    def __init__(
        self,
        image_size,
        deterministic_sampling: bool = False,
        priority_caption_language: Optional[str] = None,
    ):
        self.image_size = image_size
        self.deterministic_sampling = deterministic_sampling
        self.priority_caption_language = priority_caption_language

    def __call__(self, input_dict: Dict[str, Any]) -> Any:
        input_dict = {
            "image": input_dict["image"],
            "image_url": input_dict["image_url"],
            "wit_idx": input_dict["wit_idx"],
            "wit_features": input_dict["wit_features"].copy(),
            "language": input_dict["wit_features"]["language"].copy(),
        }

        if self.deterministic_sampling:
            rng = np.random.RandomState(input_dict["wit_idx"])
        else:
            seconds_rng = int(time.time()) % 1000000
            rng = np.random.RandomState(input_dict["wit_idx"] + seconds_rng)

        output_dict = {}
        wit_sample = input_dict["wit_features"]
        output_dict["wit_idx"] = input_dict["wit_idx"]

        output_dict[
            get_submodality_name(ModalityTypes.wit_image.value)
        ] = input_dict["image"]

        if self.priority_caption_language is None:
            choose_language = rng.choice(wit_sample["language"])
        elif self.priority_caption_language in wit_sample["language"]:
            choose_language = self.priority_caption_language
        else:
            choose_language = rng.choice(wit_sample["language"])
        choose_language = rng.choice(wit_sample["language"])
        language_idx = wit_sample["language"].index(choose_language)
        wit_text = [
            f"<{key}> <{choose_language}>"
            + wit_sample[key][language_idx]
            + f"</{choose_language}> </{key}>"
            for key in [
                "caption_alt_text_description",
                "caption_reference_description",
                "caption_title_and_reference_description",
                "context_page_description",
                "context_section_description",
                "hierarchical_section_title",
                "page_title",
                "section_title",
            ]
            if wit_sample[key][language_idx] is not None
        ]
        output_dict[
            get_submodality_name(ModalityTypes.wit_caption.value)
        ] = rng.choice(wit_text)

        return output_dict
