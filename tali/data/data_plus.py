import functools
import logging
import os
import pathlib
import random
import time
from dataclasses import dataclass
from math import floor
from typing import Any, Callable, Dict, List, Optional, Union

import datasets
import numpy as np
import PIL
import torch
import torchaudio.transforms as T
import tqdm
from pytorchvideo.transforms import ApplyTransformToKey, ShortSideScale
from rich import print
from torch.utils.data import Dataset
from torchvision.transforms import Compose
from torchvision.transforms._transforms_video import CenterCropVideo
from transformers import CLIPProcessor, WhisperProcessor

from tali.data.data import (
    AnyModalSample,
    ModalityTypes,
    dataclass_collate,
    get_base_modality,
    select_subtitles_between_timestamps,
)
from tali.decorators import configurable
from tali.frame_extractor import (
    FrameSelectionMethod,
    duration_in_seconds_from_path,
    extract_frames_pyav,
)
from tali.utils import get_logger, load_json, set_seed

logger = get_logger(__name__)
pytorchvideo_logger = get_logger("pytorchvideo", logging_level=logging.NOTSET)


def get_video_clip(video, starting_second, ending_second):
    """Extracts a clip from a video given the start and end time in seconds.

    Args:
        video: A video file.
        starting_second (int): The start time of the clip in seconds.
        ending_second (int): The end time of the clip in seconds.

    Returns:
        A video clip from the starting to the ending second.
    """
    return video.get_clip(start_sec=starting_second, end_sec=ending_second)


def get_video_tensors(video_frames, image_size):
    """Converts video frames into tensor format and applies transforms.

    Args:
        video_frames: Frames extracted from a video.
        image_size (int): The size for each video frame.

    Returns:
        Transformed video frames in tensor format.
    """
    video_frames = video_frames.permute(3, 0, 1, 2).to(torch.float32)
    video_transform = Compose(
        [
            ShortSideScale(size=image_size),
            CenterCropVideo(crop_size=(image_size, image_size)),
        ]
    )
    output_dict = ApplyTransformToKey("video", video_transform)(
        {"video": video_frames}
    )
    return output_dict["video"].permute(1, 0, 2, 3) / 255.0


def convert_to_pil(image):
    image = image.numpy().transpose(1, 2, 0)
    image = (image * 255).astype(np.uint8)
    image = PIL.Image.fromarray(image)
    return image


def videoclip_to_video_audio_tensors(
    video_path: pathlib.Path,
    rng: Optional[np.random.Generator] = None,
    return_video: bool = True,
    return_audio: bool = False,
    return_image: bool = False,
    image_size: int = 224,
    starting_second: int = 0,
    ending_second: Optional[int] = None,
    num_audio_frames: int = 1 * 16000,
    num_video_frames: int = 10,
):
    """Extracts frames from a video clip and transforms them into tensors.

    Args:
        video_path (pathlib.Path): The path to the video file.
        rng (np.random.Generator): A random number generator.
        return_video (bool): Whether to return video frames.
        return_audio (bool): Whether to return audio frames.
        return_image (bool): Whether to return image frames.
        image_size (int): The size of each image frame.
        starting_second (int): The starting time of the clip in seconds.
        ending_second (Optional[int]): The ending time of the clip in seconds.
        num_audio_frames (int): The number of audio frames to extract.
        num_video_frames (int): The number of video frames to extract.

    Returns:
        A dictionary containing video frames, image frames, and/or audio frames
        in tensor format.
    """
    output = {}
    video = image = audio = None

    if rng is None:
        rng = np.random.default_rng()

    if return_video:
        video = extract_frames_pyav(
            video_path=video_path,
            starting_second=starting_second,
            ending_second=ending_second,
            num_frames=num_video_frames + (1 if return_image else 0),
            rng=rng,
            modality="video",
            frame_selection_method=FrameSelectionMethod.RANDOM,
        )

        video = get_video_tensors(video, image_size)

        if return_image:
            image = video[0]
            video = video[1:]

        if video.shape[0] < num_video_frames:
            video = torch.cat(
                [
                    video,
                    torch.zeros(
                        num_video_frames - video.shape[0],
                        video.shape[1],
                        video.shape[2],
                        video.shape[3],
                    ),
                ],
                dim=0,
            )
        output["video"] = [convert_to_pil(frame) for frame in video]

    if return_image:
        if image is None:
            image = extract_frames_pyav(
                video_path=video_path,
                starting_second=starting_second,
                ending_second=ending_second,
                num_frames=1,
                rng=rng,
                modality="video",
                frame_selection_method=FrameSelectionMethod.RANDOM,
                single_image_frame=True,
            )
            image = get_video_tensors(image, image_size)[0]

        output["image"] = convert_to_pil(image)

    if return_audio:
        audio = extract_frames_pyav(
            video_path=video_path,
            starting_second=starting_second,
            ending_second=ending_second,
            num_frames=44100 * num_audio_frames / 16000,
            rng=rng,
            modality="audio",
            frame_selection_method=FrameSelectionMethod.SEQUENTIAL,
        )[:, 0]

        audio = extract_audio(num_audio_frames, audio)
        output["audio"] = audio

    return output


def extract_audio(num_audio_frames, audio_frames):
    audio_duration_target = float(num_audio_frames) / 16000.0
    source_sample_rate = 44100
    target_sample_rate = 16000
    audio_frames = audio_frames[: int(floor(44100 * audio_duration_target))]
    resampler = T.Resample(
        source_sample_rate, target_sample_rate, dtype=audio_frames.dtype
    )
    audio = resampler(audio_frames)
    # audio_shape = audio.shape

    if audio.shape[0] < num_audio_frames:
        audio = torch.cat(
            [
                audio,
                torch.zeros(
                    num_audio_frames - audio.shape[0],
                ),
            ],
            dim=0,
        )
    return audio


def get_submodality_name(item: AnyModalSample):
    return str(item.sub_modality).replace("SubModalityTypes.", "")


@dataclass
class TALIBaseTransformConfig:
    root_filepath: Union[str, pathlib.Path]
    modality_list: List
    rng_seed: int = 42
    top_k_tali: int = 10
    image_size: int = 224
    num_video_frames: int = 30
    num_audio_frames: int = 44100
    clip_duration_in_seconds: float = 3
    deterministic_sampling: bool = False
    priority_caption_language: Optional[str] = "en"

    @staticmethod
    def get_submodality_name(item: AnyModalSample):
        return str(item.sub_modality).replace("SubModalityTypes.", "")


class TALIBaseTransform:
    def __init__(self, config: TALIBaseTransformConfig):
        self.config = config
        self.modality_list = [
            TALIBaseTransformConfig.get_submodality_name(item)
            for item in self.config.modality_list
        ]
        self._video_transform = self._build_video_transform()

    def _build_video_transform(self):
        def transform(x, start, end, rng):
            path = x.replace("/data/", self.config.root_filepath)
            return_video = (
                TALIBaseTransformConfig.get_submodality_name(
                    ModalityTypes.youtube_video.value
                )
                in self.modality_list
            )
            return_audio = (
                TALIBaseTransformConfig.get_submodality_name(
                    ModalityTypes.youtube_audio.value
                )
                in self.modality_list
            )
            return_image = (
                TALIBaseTransformConfig.get_submodality_name(
                    ModalityTypes.youtube_image.value
                )
                in self.modality_list
            )

            return videoclip_to_video_audio_tensors(
                video_path=path,
                image_size=self.config.image_size,
                starting_second=start,
                ending_second=end,
                return_video=return_video,
                return_audio=return_audio,
                return_image=return_image,
                num_audio_frames=self.config.num_audio_frames,
                num_video_frames=self.config.num_video_frames,
                rng=rng,
            )

        return transform

    def __call__(self, input_dict: Dict[str, Any]) -> Dict[str, Any]:
        output_dict = []
        for i in range(len(input_dict[list(input_dict.keys())[0]])):
            cur_dict = self.apply_transforms(
                {k: v[i] for k, v in input_dict.items()}
            )
            output_dict.append(cur_dict)

        output_dict = {
            k: [v[k] for v in output_dict] for k in output_dict[0].keys()
        }

        return output_dict

    def apply_transforms(self, input_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Wrapper function for the transform function.
        This function is used to make the transform function configurable.
        """
        rng = self._get_rng(input_dict)
        try:
            output_dict = self._initialize_output_dict(input_dict)
            if (
                self._is_in_modality_list(ModalityTypes.wit_caption)
                or self._is_in_modality_list(ModalityTypes.wit_title)
                or self._is_in_modality_list(ModalityTypes.wit_main_body)
            ):
                output_dict[
                    TALIBaseTransformConfig.get_submodality_name(
                        ModalityTypes.wit_caption.value
                    )
                ] = self._generate_wit_text(rng, input_dict)

            if (
                self._is_in_modality_list(ModalityTypes.youtube_video)
                or self._is_in_modality_list(ModalityTypes.youtube_audio)
                or self._is_in_modality_list(ModalityTypes.youtube_image)
            ):
                (
                    youtube_media_data,
                    clip_starting_second,
                    clip_ending_second,
                ) = self._process_youtube_video(rng, input_dict)
                output_dict.update(youtube_media_data)

            if self._is_in_modality_list(ModalityTypes.youtube_description):
                output_dict[
                    TALIBaseTransformConfig.get_submodality_name(
                        ModalityTypes.youtube_description.value
                    )
                ] = self._get_youtube_description(input_dict)

            if self._is_in_modality_list(ModalityTypes.youtube_title):
                output_dict[
                    TALIBaseTransformConfig.get_submodality_name(
                        ModalityTypes.youtube_title.value
                    )
                ] = self._get_youtube_title(input_dict)

            if self._is_in_modality_list(ModalityTypes.youtube_subtitles):
                output_dict[
                    TALIBaseTransformConfig.get_submodality_name(
                        ModalityTypes.youtube_subtitles.value
                    )
                ] = self._get_youtube_subtitles(
                    input_dict,
                    youtube_media_data,
                    clip_starting_second,
                    clip_ending_second,
                )
        except Exception as e:
            logger.exception(e)
            return {}

        return output_dict

    def _get_rng(self, input_dict):
        if self.config.deterministic_sampling:
            return np.random.RandomState(input_dict["wit_idx"])
        else:
            seconds_rng = int(time.time()) % 1000000
            return np.random.RandomState(input_dict["wit_idx"] + seconds_rng)

    def _initialize_output_dict(self, input_dict):
        output_dict = {
            "wit_idx": input_dict["wit_idx"],
        }

        if self._is_in_modality_list(ModalityTypes.wit_image):
            output_dict[
                TALIBaseTransformConfig.get_submodality_name(
                    ModalityTypes.wit_image.value
                )
            ] = input_dict["image"]

        return output_dict

    def _is_in_modality_list(self, modality_type):
        return (
            TALIBaseTransformConfig.get_submodality_name(modality_type.value)
            in self.modality_list
        )

    def _generate_wit_text(self, rng, input_dict):
        wit_sample = input_dict["wit_features"]
        if self.config.priority_caption_language is None:
            choose_language = rng.choice(wit_sample["language"])
        elif self.config.priority_caption_language in wit_sample["language"]:
            choose_language = self.config.priority_caption_language
        else:
            choose_language = rng.choice(wit_sample["language"])
        language_idx = wit_sample["language"].index(choose_language)
        wit_text = [
            f"<{key}> <{choose_language}> "
            + wit_sample[key][language_idx]
            + f" </{choose_language}> </{key}>"
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
        return rng.choice(wit_text)

    def _process_youtube_video(self, rng, input_dict):
        output_dict = {}
        choose_video = rng.choice(
            input_dict["youtube_content_video"][: self.config.top_k_tali]
        )
        video_id = choose_video.split("/")[-2]
        video_starting_second = float(
            choose_video.split("/")[-1].split("_")[1].replace(".mp4", "")
        )
        choose_video = choose_video.replace(
            "/data/",
            self.config.root_filepath,
        )
        duration = duration_in_seconds_from_path(
            choose_video, modality="video"
        )
        total_time_in_seconds = int(floor(duration))
        max_starting_second = (
            total_time_in_seconds - self.config.clip_duration_in_seconds
        )
        if max_starting_second <= 0:
            clip_starting_second = 0
            clip_ending_second = total_time_in_seconds
        else:
            clip_starting_second = rng.randint(0, max_starting_second)
            clip_ending_second = (
                clip_starting_second + self.config.clip_duration_in_seconds
            )
        output_dict["youtube_video_id"] = video_id
        if (
            self._is_in_modality_list(ModalityTypes.youtube_video)
            or self._is_in_modality_list(ModalityTypes.youtube_audio)
            or self._is_in_modality_list(ModalityTypes.youtube_image)
        ):
            youtube_media_data = self._video_transform(
                x=choose_video,
                start=clip_starting_second,
                end=clip_ending_second,
                rng=rng,
            )
            if "video" in youtube_media_data:
                output_dict[
                    TALIBaseTransformConfig.get_submodality_name(
                        ModalityTypes.youtube_video.value
                    )
                ] = youtube_media_data["video"]
            if "audio" in youtube_media_data:
                output_dict[
                    TALIBaseTransformConfig.get_submodality_name(
                        ModalityTypes.youtube_audio.value
                    )
                ] = youtube_media_data["audio"]
            if "image" in youtube_media_data:
                output_dict[
                    TALIBaseTransformConfig.get_submodality_name(
                        ModalityTypes.youtube_image.value
                    )
                ] = youtube_media_data["image"]
        return output_dict, clip_starting_second, clip_ending_second

    def _get_youtube_description(self, input_dict):
        return (
            "<ydesc> " + input_dict["youtube_description_text"] + " </ydesc>"
        )

    def _get_youtube_title(self, input_dict):
        return "<ytitle> " + input_dict["youtube_title_text"] + " </ytitle>"

    def _get_youtube_subtitles(
        self,
        input_dict,
        youtube_media_data,
        clip_starting_second,
        clip_ending_second,
    ):
        return (
            "<ysub> "
            + select_subtitles_between_timestamps(
                subtitle_dict=load_json(
                    input_dict["youtube_subtitle_text"].replace(
                        "/data/",
                        self.config.root_filepath,
                    )
                ),
                starting_timestamp=clip_starting_second,
                ending_timestamp=clip_starting_second + clip_ending_second,
            )
            + " </ysub>"
        )


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


def get_next_on_error(func: Callable[..., Any]) -> Callable[..., Any]:
    """A decorator that catches exceptions in the wrapped function.

    If an exception occurs, it re-runs the function with the next index in sequence.

    Args:
        func: The function to decorate.

    Returns:
        A function with the same signature as `func`, but that catches exceptions and re-runs.
    """

    @functools.wraps(func)
    def wrapper_collect_metrics(*args, **kwargs) -> Any:
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.exception(
                f"Error occurred at idx {kwargs['idx']} {e}, getting the next item instead."
            )
            random_number = random.randint(0, 10**8)
            kwargs["idx"] = kwargs["idx"] + random_number
            return func(*args, **kwargs)

    return wrapper_collect_metrics


@configurable
class TALIBase(Dataset):
    """Represents the TALI dataset.

    This dataset is configurable and allows for custom transformations to be applied.
    """

    def __init__(
        self,
        set_name: str,
        tali_dataset_dir: str,
        modality_list: List,
        num_samples_per_episode: int,
        rng_seed: int = 42,
        top_k_tali: int = 10,
        image_size: int = 224,
        num_video_frames: int = 5,
        num_audio_frames: int = 16000,
        clip_duration_in_seconds: float = 3.0,
        deterministic_sampling: bool = True,
        dummy_batch_mode: bool = False,
        image_text_model_name: str = "openai/clip-vit-base-patch16",
        audio_model_name: str = "openai/whisper-base",
        use_model_preprocessing: bool = True,
        total_num_samples: Optional[int] = None,
        cache_generated_samples_in_memory: bool = False,
        cache_num_samples: int = 10,
    ) -> None:
        """Initializes the TALIBase dataset.

        Args:
            set_name: The set name in the TALI dataset directory.
            tali_dataset_dir: Directory where the TALI dataset is located.
            modality_list: List of modalities to consider in the dataset.
            num_samples_per_episode: Number of samples per episode.
            rng_seed: Seed for the random number generator.
            top_k_tali: The top K TALI instances to consider.
            image_size: Size of the images.
            num_video_frames: Number of video frames.
            num_audio_frames: Number of audio frames.
            clip_duration_in_seconds: Duration of the video clips in seconds.
            deterministic_sampling: If True, deterministic sampling is used.
            dummy_batch_mode: If True, the same batch is returned for every call to get_sample.
            image_text_model_name: The name of the image-text model.
            audio_model_name: The name of the audio model.
            use_model_preprocessing: If True, use model preprocessing.
            total_num_samples: Total number of samples. If None, equals the length of the dataset.
            cache_generated_samples_in_memory: If True, cache generated samples in memory.
            cache_num_samples: Number of samples to cache.
        """
        super().__init__()
        # Prepare transformations
        transform = TALIBaseTransform(
            config=TALIBaseTransformConfig(
                root_filepath=f"{tali_dataset_dir}/",
                modality_list=modality_list,
                top_k_tali=top_k_tali,
                rng_seed=rng_seed,
                image_size=image_size,
                num_video_frames=num_video_frames,
                num_audio_frames=num_audio_frames,
                clip_duration_in_seconds=clip_duration_in_seconds,
                deterministic_sampling=deterministic_sampling,
            )
        )

        # Load dataset
        print(
            f"path: Antreas/TALI, split: {set_name}, cache_dir: {os.environ['HF_CACHE_DIR']}"
        )
        self.dataset = datasets.load_dataset(
            path="Antreas/TALI",
            split=set_name,
            keep_in_memory=False,
            cache_dir=os.environ["HF_CACHE_DIR"],
        )

        # Apply transformations
        self.dataset = self.dataset.with_transform(transform)

        # Initialize variables
        self.num_dataset_samples = len(self.dataset)
        self.dummy_batch_mode = dummy_batch_mode
        self.cache_generated_samples_in_memory = (
            cache_generated_samples_in_memory
        )
        self.cache_num_samples = cache_num_samples

        # Initialize in-memory cache
        if self.cache_generated_samples_in_memory:
            self.mem_cache = []

        self.dummy_batch = None
        self.num_samples_per_episode = num_samples_per_episode
        self.num_samples = (
            total_num_samples
            if total_num_samples is not None
            else len(self.dataset)
        )
        self.image_text_model_name = image_text_model_name
        self.audio_model_name = audio_model_name
        self.use_model_preprocessing = use_model_preprocessing
        if self.use_model_preprocessing:
            self.transforms = self.build_model_transforms()
        else:
            self.transforms = self.build_basic_transforms()

    def build_model_transforms(self) -> Dict[str, Callable]:
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
            ).pixel_values.squeeze(1)

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

    def build_basic_transforms(self) -> Dict[str, Callable]:
        """
        Build a dictionary of basic transformation functions for each modality of data (image, text, audio, video).

        These transformations do not use any model-specific preprocessing steps.

        Returns:
            A dictionary of transformations with keys as modality names and values as transformation functions.
        """
        # Return a dictionary of basic transformations
        return {
            "image": lambda x: (x * 255).to(
                torch.uint8
            ),  # Scale and convert to uint8
            "text": lambda x: x,  # No transformation on text
            "audio": lambda x: x.view(-1),  # Flatten audio tensor
            "video": lambda x: [
                (item * 255).to(torch.uint8) for item in x
            ],  # Scale and convert each frame to uint8
        }

    def __getitem__(
        self, idx: int
    ) -> Dict[str, Union[torch.Tensor, List[Union[str, torch.Tensor]]]]:
        """
        Retrieve the samples from the dataset at the specified index.

        The method first retrieves all samples from the index to the (index + number of samples per episode).
        It then transforms the collected samples into a dictionary with the sample's key as the dictionary key and
        a list of the corresponding values as the dictionary value. The list of values is converted into a tensor if possible.

        Args:
            idx: The index of the first sample to retrieve.

        Returns:
            A dictionary of the episode's data. The dictionary's keys are the data keys, and the values are tensors or
            lists of the corresponding data items.

        """
        episode_dict = {}  # 📔 store all samples in an episode

        # 🔄 Iterate over samples in the episode
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
                torch.stack(value, dim=0)
                if isinstance(value[0], torch.Tensor)
                else value
            )

        # 💾 If caching is enabled, add the episode to the cache and ensure cache size
        if self.cache_generated_samples_in_memory:
            self.mem_cache.append(episode_dict)
            if len(self.mem_cache) > self.cache_num_samples:
                self.mem_cache.pop(0)

        return episode_dict

    @get_next_on_error
    def get_sample(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Retrieve the sample at the specified index, performing necessary transformations based on the modality of the data.

        This method retrieves the sample at the specified index. If the dummy_batch_mode is active and the dummy_batch has
        been initialized, it will return the dummy_batch instead of a real sample. The method then applies the corresponding
        transformations to each value in the sample depending on its modality.

        Args:
            idx: The index of the sample to retrieve.

        Returns:
            A dictionary with the transformed sample data. The dictionary's keys are the data keys, and the values are the
            corresponding transformed data items.

        Raises:
            If an exception occurs, this method will catch it, log the error, and try to retrieve a different sample.
        """

        # 💡 If dummy batch mode is active and dummy batch is available, return the dummy batch
        if self.dummy_batch_mode and self.dummy_batch is not None:
            return self.dummy_batch

        # 🔄 If requested number of samples exceeds the dataset size, wrap around to the start of the dataset
        if self.num_samples > self.num_dataset_samples:
            idx = idx % self.num_dataset_samples

        # 📖 Get the sample from the dataset
        sample = self.dataset[idx]

        # 🔄 Iterate over the sample's items
        for key, value in sample.items():
            # 🔄 Check each possible transform key
            for transform_key, transform_value in self.transforms.items():
                # 🔄 If a transform applies to this key, apply it
                if transform_key in key and key != "youtube_video_id":
                    # 🎵 If audio data, add an extra dimension
                    if "audio" in key:
                        value = value.unsqueeze(0)

                    # 💫 Apply the transformation
                    sample[key] = transform_value(value)
                    # 🧮 If the result is a list, stack it into a tensor
                    sample[key] = (
                        torch.stack(sample[key], dim=0).squeeze()
                        if isinstance(sample[key], list)
                        else sample[key].squeeze()
                    )

                    break

        # 💡 If dummy batch mode is active and dummy batch is not yet available, store this sample as the dummy batch
        if self.dummy_batch_mode:
            if self.dummy_batch is None:
                self.dummy_batch = sample
            return self.dummy_batch

        return sample

    def __len__(self) -> int:
        """
        Return the total number of samples that the dataset contains.

        This method allows the use of Python's built-in `len()` function on instances of this class. It returns the total
        number of samples, which is either the total number of samples in the dataset, or a predetermined value.

        Returns:
            The total number of samples in the dataset.
        """

        # 📚 Return the number of samples
        return self.num_samples


class CustomConcatDataset(Dataset):
    def __init__(self, datasets: List[Dataset]):
        """
        🎯 Initialize a dataset that concatenates multiple datasets.

        Parameters:
        - datasets (List[Dataset]): A list of datasets to be concatenated.

        """
        self.datasets = datasets

    def __len__(self) -> int:
        """
        📚 Return the total number of samples that the dataset contains.

        This method allows the use of Python's built-in `len()` function on instances of this class. It sums up the length
        of all the datasets passed during instantiation.

        Returns:
            The total number of samples in all the datasets.
        """
        return sum(len(d) for d in self.datasets)

    def __getitem__(self, idx: int) -> Any:
        """
        🧪 Get a single data point from the dataset.

        Parameters:
        - idx (int): The index of the item to fetch.

        Returns:
            The data point from the respective dataset.

        """
        # Dividing the idx by the total number of datasets gives us which dataset to use
        dataset_idx = idx % len(self.datasets)
        # The remainder gives us which item in the selected dataset to use
        item_idx = idx // len(self.datasets)

        return self.datasets[dataset_idx][item_idx]


if __name__ == "__main__":
    import cProfile
    import pstats

    import tqdm
    from rich import print
    from rich.traceback import install
    from torch.utils.data import DataLoader

    install()
    os.environ["HYDRA_FULL_ERROR"] = "1"
    set_seed(42)

    def sample():
        # transform = TALIBaseTransform(
        #     config=TALIBaseTransformConfig(
        #         root_filepath="/data/datasets/tali-wit-2-1-buckets/",
        #         modality_list=[
        #             ModalityTypes.wit_image.value,
        #             ModalityTypes.wit_caption.value,
        #             ModalityTypes.wit_title.value,
        #             ModalityTypes.wit_main_body.value,
        #             ModalityTypes.youtube_image.value,
        #             ModalityTypes.youtube_video.value,
        #             ModalityTypes.youtube_subtitles.value,
        #             ModalityTypes.youtube_audio.value,
        #             ModalityTypes.youtube_description.value,
        #         ],
        #         rng_seed=42,
        #         top_k_tali=10,
        #         image_size=224,
        #         num_video_frames=5,
        #         num_audio_frames=16000,
        #         clip_duration_in_seconds=3.0,
        #         deterministic_sampling=True,
        #     )
        # )
        # dataset = datasets.load_from_disk(
        #     "/home/evolvingfungus/forge/workspaces/tali-2-2/train-set"
        # )
        # dataset = dataset.with_transform(transform)
        dataset = TALIBase(
            set_name="train",
            tali_dataset_dir="/data_fast/tali-v-3-4/",
            modality_list=[
                ModalityTypes.wit_image.value,
                ModalityTypes.wit_caption.value,
                ModalityTypes.wit_title.value,
                ModalityTypes.wit_main_body.value,
                ModalityTypes.youtube_title.value,
                ModalityTypes.youtube_image.value,
                ModalityTypes.youtube_video.value,
                ModalityTypes.youtube_subtitles.value,
                ModalityTypes.youtube_audio.value,
                ModalityTypes.youtube_description.value,
            ],
            rng_seed=42,
            top_k_tali=10,
            image_size=224,
            num_video_frames=10,
            num_audio_frames=16000 * 2,
            clip_duration_in_seconds=3,
            deterministic_sampling=False,
            dummy_batch_mode=False,
            image_text_model_name="openai/clip-vit-base-patch16",
            audio_model_name="openai/whisper-base",
            use_model_preprocessing=False,
            num_samples_per_episode=16,
            total_num_samples=1000000,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=1,
            num_workers=16,
            shuffle=False,
            collate_fn=dataclass_collate,
            persistent_workers=True,
            prefetch_factor=2,
            pin_memory=False,
        )
        num_samples = 10000

        with tqdm.tqdm(total=len(dataloader)) as pbar:
            for i, example in enumerate(dataloader):
                # example = generate_hierarchical_data_dict(example)
                # if i == 0:
                #     start_time = time.time()
                time.sleep(1)
                # print(example["youtube_title_text"])
                print(list(example.keys()))

                pbar.set_description(f"Processing {i}th example")
                pbar.update(1)
        # end_time = time.time()
        # print(f"Processed {num_samples} samples in {end_time - start_time} seconds")

    sample()
    # pr = cProfile.Profile()
    # pr.runcall(sample)

    # ps = pstats.Stats(pr).sort_stats("tottime")
    # ps.print_stats(10)
# write a transform for the wit dataset, and, add an option for a youtube image sampling process
