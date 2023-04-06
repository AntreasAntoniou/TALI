import os
import pathlib
import time
from typing import Any, Dict, Optional
import datasets

import numpy as np
from tali_wit.data_plus import get_next_on_error, get_submodality_name
import torch
import tqdm
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import Dataset, Subset, DataLoader
import datasets
from tali_wit.data import (
    dataclass_collate,
    default_image_transforms,
    ModalityTypes,
)

from tali_wit.decorators import configurable
from tali_wit.utils import get_logger, load_json, save_json

from transformers import (
    CLIPProcessor,
    WhisperProcessor,
)

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
        infinite_sampling: bool = False,
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
            cache_dir=wit_dataset_dir,
        )
        self.num_samples_per_episode = num_samples_per_episode
        self.indices_filepath = (
            pathlib.Path(wit_dataset_dir) / "wit_indices.json"
        )

        if not self.indices_filepath.exists():
            tali_val_dataset = datasets.load_from_disk(
                pathlib.Path(tali_dataset_dir) / "val-set"
            )
            tali_val_indices = [
                sample["wit_idx"] for sample in tali_val_dataset
            ]

            tali_test_dataset = datasets.load_from_disk(
                pathlib.Path(tali_dataset_dir) / "test-set"
            )
            tali_test_indices = [
                sample["wit_idx"] for sample in tali_test_dataset
            ]

            train_wit_indices = []
            with tqdm.tqdm(total=len(self.dataset)) as pbar:
                for i in range(len(self.dataset)):
                    if i not in tali_val_indices and i not in tali_test_indices:
                        train_wit_indices.append(i)
                    pbar.update(1)

            self.indices = {
                "train": train_wit_indices,
                "val": tali_val_indices,
                "test": tali_test_indices,
            }
            save_json(
                filepath=os.path.join(self.wit_dataset_dir, "wit_indices.json"),
                dict_to_store=self.indices,
            )
        else:
            self.indices = load_json(self.indices_filepath)

        self.dataset = Subset(self.dataset, self.indices[set_name])
        self.dummy_batch_mode = dummy_batch_mode
        self.dummy_batch = None

        self.infinite_sampling = infinite_sampling
        self.num_samples = 10**8 if infinite_sampling else len(self.dataset)
        self.image_text_model_name = image_text_model_name
        self.audio_model_name = audio_model_name
        self.transforms = self.build_transforms()

    def build_transforms(self):
        self.image_text_processor = CLIPProcessor.from_pretrained(
            self.image_text_model_name
        )
        self.audio_processor = WhisperProcessor.from_pretrained(
            self.audio_model_name
        )

        return {
            "image": lambda x: self.image_text_processor(
                images=x, return_tensors="pt"
            ).pixel_values.squeeze(1),
            "text": lambda x: self.image_text_processor(
                text=x, return_tensors="pt", padding=True, truncation=True
            ).input_ids.squeeze(0),
            "audio": lambda x: torch.cat(
                [
                    self.audio_processor(
                        item,
                        sampling_rate=16000,
                        return_tensors="pt",
                    ).input_features
                    for item in x.unbind(0)
                ]
            ),
            "video": lambda x: torch.stack(
                [
                    self.image_text_processor(
                        images=image, return_tensors="pt"
                    ).pixel_values
                    for image in x
                ],
                dim=0,
            ),
        }

    def __getitem__(self, idx):
        episode_dict = {}

        if self.dummy_batch_mode and self.dummy_batch is not None:
            return self.dummy_batch

        for i in range(idx, idx + self.num_samples_per_episode):
            sample = self.get_sample(idx=i)
            for key, value in sample.items():
                if (
                    "text" in key
                    and len(value) < 77
                    and isinstance(value, torch.Tensor)
                ):
                    value = torch.cat(
                        [value, 49407 * torch.ones(77 - len(value)).long()]
                    )
                if key not in episode_dict:
                    episode_dict[key] = [value]
                else:
                    episode_dict[key].append(value)

        for key, value in episode_dict.items():
            episode_dict[key] = (
                torch.stack(value, dim=0)
                if isinstance(value[0], torch.Tensor)
                else value
            )

        if self.dummy_batch_mode:
            if self.dummy_batch is None:
                self.dummy_batch = sample
            return self.dummy_batch

        return episode_dict

    @get_next_on_error
    def get_sample(self, idx):
        if self.infinite_sampling:
            idx = idx % len(self.dataset)

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
        self.image_transform = default_image_transforms(self.image_size)
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
        ] = self.image_transform(input_dict["image"])

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


if __name__ == "__main__":
    import tqdm
    from rich.traceback import install
    import cProfile
    import pstats
    import datasets

    install()
    os.environ["HYDRA_FULL_ERROR"] = "1"

    def sample():
        dataset = WITBase(
            cache_dir="/data/wit_cache",
            tali_dataset_dir="/data/",
            image_size=224,
            deterministic_sampling=True,
            infinite_sampling=False,  # True,
            priority_caption_language="en",
            set_name="train",
            dummy_batch_mode=False,
            image_text_model_name="openai/clip-vit-base-patch16",
            audio_model_name="openai/whisper-base",
        )
        dataloader = DataLoader(
            dataset,
            batch_size=32,
            num_workers=12,
            shuffle=True,
            collate_fn=dataclass_collate,
        )
        num_samples = 100
        with tqdm.tqdm(total=len(dataloader)) as pbar:
            for i, example in enumerate(dataloader):
                pbar.set_description(f"Processing {i}th example")
                pbar.update(1)

    pr = cProfile.Profile()
    pr.runcall(sample)

    ps = pstats.Stats(pr).sort_stats("tottime")
    ps.print_stats()
# write a transform for the wit dataset, and, add an option for a youtube image sampling process
