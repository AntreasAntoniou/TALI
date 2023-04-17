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
import torch
import tqdm
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    ShortSideScale,
)
from rich import print
from torch.utils.data import Dataset
from torchvision.transforms import Compose
from torchvision.transforms._transforms_video import CenterCropVideo
import torch
import torchaudio.transforms as T
import datasets
from tali_wit.data import (
    AnyModalSample,
    dataclass_collate,
    default_image_transforms,
    ModalityTypes,
    get_base_modality,
    select_subtitles_between_timestamps,
)

from tali_wit.decorators import configurable
from tali_wit.frame_extractor import (
    FrameSelectionMethod,
    duration_in_seconds_from_path,
    extract_frames_pyav,
    extract_frames_torchvision,
)
from tali_wit.utils import get_logger, load_json, set_seed, timeout
from transformers import (
    CLIPProcessor,
    WhisperProcessor,
)

logger = get_logger(__name__)
pytorchvideo_logger = get_logger("pytorchvideo", logging_level=logging.NOTSET)


def get_video_clip(video, starting_second, ending_second):
    return video.get_clip(start_sec=starting_second, end_sec=ending_second)


def get_video_tensors(video_frames, image_size):
    video_frames = video_frames.permute(3, 0, 1, 2).to(torch.float32)
    video_transform = Compose(
        [
            ShortSideScale(size=image_size),
            CenterCropVideo(crop_size=(image_size, image_size)),
        ]
    )
    output_dict = ApplyTransformToKey("video", video_transform)({"video": video_frames})
    return output_dict["video"].permute(1, 0, 2, 3) / 255.0


def videoclip_to_video_audio_tensors(
    video_path: pathlib.Path,
    rng: np.random.Generator,
    return_video: bool = True,
    return_audio: bool = False,
    return_image: bool = False,
    image_size: int = 224,
    starting_second: int = 0,
    ending_second: Optional[int] = None,
    num_audio_frames: int = 1 * 16000,
    num_video_frames: int = 10,
):
    output = {}
    video = None
    image = None
    audio = None

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
        output["video"] = video

    if return_image:
        if image is None:
            # logger.info("Extracting image from video")
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
            # logger.info("Extracted image from video")
            image = get_video_tensors(image, image_size)[0]
            # logger.info("Got image from video")

        output["image"] = image

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
    # print("HELLLOO")
    # for key, value in output.items():
    #     print(key, value.shape)

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


@dataclass
class TALIBaseTransformConfig:
    root_filepath: Union[str, pathlib.Path]
    modality_list: List[AnyModalSample]
    rng_seed: int = 42
    top_k_tali: int = 10
    image_size: int = 224
    num_video_frames: int = 30
    num_audio_frames: int = 44100
    clip_duration_in_seconds: float = 3
    deterministic_sampling: bool = False
    priority_caption_language: Optional[str] = "en"


def get_submodality_name(item: AnyModalSample):
    return str(item.sub_modality).replace("SubModalityTypes.", "")


class TALIBaseTransform:
    def __init__(self, config: TALIBaseTransformConfig):
        self.config = config
        self.modality_list = [
            get_submodality_name(item) for item in self.config.modality_list
        ]
        self.image_transform = default_image_transforms(self.config.image_size)
        self.video_transform = (
            lambda x, start, end, rng: videoclip_to_video_audio_tensors(
                video_path=x.replace(
                    "/data/",
                    self.config.root_filepath,
                ),
                image_size=self.config.image_size,
                starting_second=start,
                ending_second=end,
                return_video=get_submodality_name(ModalityTypes.youtube_video.value)
                in self.modality_list,
                return_audio=get_submodality_name(ModalityTypes.youtube_audio.value)
                in self.modality_list,
                return_image=get_submodality_name(ModalityTypes.youtube_image.value)
                in self.modality_list,
                num_audio_frames=self.config.num_audio_frames,
                num_video_frames=self.config.num_video_frames,
                rng=rng,
            )
        )

        self.select_subtitles_between_timestamps = select_subtitles_between_timestamps

    def __call__(self, input_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Transforms the input dictionary.

        Args:
            input_dict (Dict[str, Any]): The input dictionary.
            dict_keys([
                'image', 'image_url', 'item_idx',
                'wit_features', 'wit_idx', 'youtube_content_video',
                'youtube_subtitle_text', 'youtube_title_text',
                'youtube_description_text'])

        Returns:
            Dict[str, Any]: The transformed dictionary.
        """
        output_dict = []
        for i in range(len(input_dict[list(input_dict.keys())[0]])):
            cur_dict = self.apply_transforms({k: v[i] for k, v in input_dict.items()})
            output_dict.append(cur_dict)

        output_dict = {k: [v[k] for v in output_dict] for k in output_dict[0].keys()}

        return output_dict

    def apply_transforms(self, input_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Wrapper function for the transform function.

        This function is used to make the transform function configurable.

        Args:
            input_dict (Dict[str, Any]): The input dictionary.
            dict_keys([
                'image', 'image_url', 'item_idx',
                'wit_features', 'wit_idx', 'youtube_content_video',
                'youtube_subtitle_text', 'youtube_title_text',
                'youtube_description_text'])

        Returns:
            Dict[str, Any]: The transformed dictionary.
        """

        if self.config.deterministic_sampling:
            rng = np.random.RandomState(input_dict["wit_idx"])
        else:
            seconds_rng = int(time.time()) % 1000000
            rng = np.random.RandomState(input_dict["wit_idx"] + seconds_rng)

        try:
            output_dict = {}

            wit_sample = input_dict["wit_features"]
            output_dict["wit_idx"] = input_dict["wit_idx"]

            if (
                get_submodality_name(ModalityTypes.wit_image.value)
                in self.modality_list
            ):
                output_dict[
                    get_submodality_name(ModalityTypes.wit_image.value)
                ] = self.image_transform(input_dict["image"])

            if (
                get_submodality_name(ModalityTypes.wit_caption.value)
                in self.modality_list
                or get_submodality_name(ModalityTypes.wit_title.value)
                in self.modality_list
                or get_submodality_name(ModalityTypes.wit_main_body.value)
                in self.modality_list
            ):
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
                output_dict[
                    get_submodality_name(ModalityTypes.wit_caption.value)
                ] = rng.choice(wit_text)

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
            duration = duration_in_seconds_from_path(choose_video, modality="video")
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
                get_submodality_name(ModalityTypes.youtube_video.value)
                in self.modality_list
                or get_submodality_name(ModalityTypes.youtube_audio.value)
                in self.modality_list
                or get_submodality_name(ModalityTypes.youtube_image.value)
                in self.modality_list
            ):
                youtube_media_data = self.video_transform(
                    x=choose_video,
                    start=clip_starting_second,
                    end=clip_ending_second,
                    rng=rng,
                )

                if "video" in youtube_media_data:
                    output_dict[
                        get_submodality_name(ModalityTypes.youtube_video.value)
                    ] = youtube_media_data["video"]

                if "audio" in youtube_media_data:
                    output_dict[
                        get_submodality_name(ModalityTypes.youtube_audio.value)
                    ] = youtube_media_data["audio"]

                if "image" in youtube_media_data:
                    output_dict[
                        get_submodality_name(ModalityTypes.youtube_image.value)
                    ] = youtube_media_data["image"]

            if (
                get_submodality_name(ModalityTypes.youtube_description.value)
                in self.modality_list
            ):
                output_dict[
                    get_submodality_name(ModalityTypes.youtube_description.value)
                ] = (
                    f"<ydesc> " + input_dict["youtube_description_text"] + f" </ydesc>"
                )

            if (
                get_submodality_name(ModalityTypes.youtube_title.value)
                in self.modality_list
            ):
                output_dict[get_submodality_name(ModalityTypes.youtube_title.value)] = (
                    f"<ytitle> " + input_dict["youtube_title_text"] + f" </ytitle>"
                )

            if (
                get_submodality_name(ModalityTypes.youtube_subtitles.value)
                in self.modality_list
            ):
                output_dict[
                    get_submodality_name(ModalityTypes.youtube_subtitles.value)
                ] = (
                    "<ysub> "
                    + select_subtitles_between_timestamps(
                        subtitle_dict=load_json(
                            input_dict["youtube_subtitle_text"].replace(
                                "/data/",
                                self.config.root_filepath,
                            )
                        ),
                        starting_timestamp=video_starting_second + clip_starting_second,
                        ending_timestamp=video_starting_second
                        + clip_starting_second
                        + clip_ending_second,
                    )
                    + " </ysub>"
                )
        except Exception as e:
            logger.exception(e)
            return {}

        # for key, value in list(output_dict.items()):
        #     if not isinstance(value, list):
        #         output_dict[key] = [value]

        return output_dict

    def __repr__(self):
        return f"{self.__class__.__name__}({self.config})"


def generate_hierarchical_data_dict(data_dict: Dict[str, Any]) -> Dict[str, Any]:
    modality_hierarchical_output_dict = {}
    for sub_modality_name in list(data_dict.keys()):
        modality_type = get_base_modality(sub_modality_name)
        if modality_type is None:
            if "other" not in modality_hierarchical_output_dict:
                modality_hierarchical_output_dict["other"] = {}
            modality_hierarchical_output_dict["other"][sub_modality_name] = data_dict[
                sub_modality_name
            ]
            continue

        if modality_type not in modality_hierarchical_output_dict:
            modality_hierarchical_output_dict[modality_type.value] = {}

        modality_hierarchical_output_dict[modality_type.value][
            sub_modality_name
        ] = data_dict[sub_modality_name]
    return modality_hierarchical_output_dict


class TALIBaseDemoTransform:
    def __init__(self, config: TALIBaseTransformConfig):
        self.config = config
        self.modality_list = [
            get_submodality_name(item) for item in self.config.modality_list
        ]
        self.image_transform = default_image_transforms(self.config.image_size)
        self.video_transform = (
            lambda x, start, end, rng: videoclip_to_video_audio_tensors(
                video_path=x.replace(
                    "/data/",
                    self.config.root_filepath.as_posix()
                    if isinstance(self.config.root_filepath, pathlib.Path)
                    else self.config.root_filepath,
                ),
                image_size=self.config.image_size,
                starting_second=start,
                ending_second=end,
                return_video=get_submodality_name(ModalityTypes.youtube_video.value)
                in self.modality_list,
                return_audio=get_submodality_name(ModalityTypes.youtube_audio.value)
                in self.modality_list,
                return_image=get_submodality_name(ModalityTypes.youtube_image.value)
                in self.modality_list,
                num_audio_frames=self.config.num_audio_frames,
                num_video_frames=self.config.num_video_frames,
                rng=rng,
            )
        )

        self.select_subtitles_between_timestamps = select_subtitles_between_timestamps

    def __call__(self, input_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Wrapper function for the transform function.

        This function is used to make the transform function configurable.

        Args:
            input_dict (Dict[str, Any]): The input dictionary.
            dict_keys([
                'image', 'image_url', 'item_idx',
                'wit_features', 'wit_idx', 'youtube_content_video',
                'youtube_subtitle_text', 'youtube_title_text',
                'youtube_description_text'])

        Returns:
            Dict[str, Any]: The transformed dictionary.
        """
        if self.config.deterministic_sampling:
            rng = np.random.RandomState(input_dict["wit_idx"])
        else:
            # Get the number of seconds since the start of computer time
            seconds_rng = int(time.time()) % 1000000
            rng = np.random.RandomState(input_dict["wit_idx"] + seconds_rng)

        try:
            output_dict = {}
            for key in list(input_dict.keys()):
                input_dict[key] = input_dict[key][0]

            wit_sample = input_dict["wit_features"]
            output_dict["wit_idx"] = [input_dict["wit_idx"]]
            output_dict["captions"] = {}
            if (
                get_submodality_name(ModalityTypes.wit_image.value)
                in self.modality_list
            ):
                output_dict[
                    get_submodality_name(ModalityTypes.wit_image.value)
                ] = self.image_transform(input_dict["image"])

            if (
                get_submodality_name(ModalityTypes.wit_caption.value)
                in self.modality_list
                or get_submodality_name(ModalityTypes.wit_title.value)
                in self.modality_list
                or get_submodality_name(ModalityTypes.wit_main_body.value)
                in self.modality_list
            ):
                for language in wit_sample["language"]:
                    language_idx = wit_sample["language"].index(language)
                    wit_text = {
                        key: f"" + wit_sample[key][language_idx] + f""
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
                    }
                    output_dict["captions"][language] = wit_text

            choose_video = rng.choice(
                input_dict["youtube_content_video"][: self.config.top_k_tali]
            )
            video_id = choose_video.split("/")[-2]
            video_starting_second = float(
                choose_video.split("/")[-1].split("_")[1].replace(".mp4", "")
            )
            clip_starting_second = rng.randint(
                0, 30 - self.config.clip_duration_in_seconds
            )
            clip_ending_second = (
                clip_starting_second + self.config.clip_duration_in_seconds
            )
            output_dict["youtube_video_id"] = video_id

            if (
                get_submodality_name(ModalityTypes.youtube_video.value)
                in self.modality_list
                or get_submodality_name(ModalityTypes.youtube_audio.value)
                in self.modality_list
            ):
                youtube_media_data = self.video_transform(
                    x=choose_video,
                    start=clip_starting_second,
                    end=clip_ending_second,
                    rng=rng,
                )

                if "video" in youtube_media_data:
                    output_dict[
                        get_submodality_name(ModalityTypes.youtube_video.value)
                    ] = youtube_media_data["video"]

                if "audio" in youtube_media_data:
                    output_dict[
                        get_submodality_name(ModalityTypes.youtube_audio.value)
                    ] = youtube_media_data["audio"]

                if "image" in youtube_media_data:
                    output_dict[
                        get_submodality_name(ModalityTypes.youtube_image.value)
                    ] = youtube_media_data["image"]

            if (
                get_submodality_name(ModalityTypes.youtube_description.value)
                in self.modality_list
            ):
                output_dict[
                    get_submodality_name(ModalityTypes.youtube_description.value)
                ] = ("<ydesc> " + input_dict["youtube_description_text"] + " </ydesc>")

            if (
                get_submodality_name(ModalityTypes.youtube_subtitles.value)
                in self.modality_list
            ):
                output_dict[
                    get_submodality_name(ModalityTypes.youtube_description.value)
                ] = (
                    "<ysub> "
                    + select_subtitles_between_timestamps(
                        subtitle_dict=load_json(
                            input_dict["youtube_subtitle_text"].replace(
                                "/data/",
                                self.config.root_filepath.as_posix()
                                if isinstance(self.config.root_filepath, pathlib.Path)
                                else self.config.root_filepath,
                            )
                        ),
                        starting_timestamp=video_starting_second + clip_starting_second,
                        ending_timestamp=video_starting_second
                        + clip_starting_second
                        + clip_ending_second,
                    )
                    + " </ysub>"
                )
        except Exception as e:
            logger.exception(e)
            return {}

        for key, value in list(output_dict.items()):
            if not isinstance(value, list):
                output_dict[key] = [value]

        return output_dict

    def __repr__(self):
        return f"{self.__class__.__name__}({self.config})"


@configurable
def get_dataset(
    set_name: str,
    tali_root_filepath,
    hf_tali_root_filepath,
    modality_list,
    rng_seed=42,
    top_k_tali=10,
    image_size=224,
    num_video_frames=5,
    num_audio_frames=16000,
    clip_duration_in_seconds=3.0,
    deterministic_sampling=True,
    dummy_batch_mode: bool = False,
):
    transform = TALIBaseTransform(
        config=TALIBaseTransformConfig(
            root_filepath=tali_root_filepath,
            modality_list=modality_list,
            top_k_tali=top_k_tali,
            rng_seed=rng_seed,
            image_size=image_size,
            num_video_frames=num_video_frames,
            num_audio_frames=num_audio_frames,
            clip_duration_in_seconds=clip_duration_in_seconds,
            deterministic_sampling=deterministic_sampling,
            dummy_batch_mode=dummy_batch_mode,
        )
    )
    dataset = datasets.load_from_disk(
        pathlib.Path(hf_tali_root_filepath) / f"{set_name}-set"
    )
    dataset = dataset.with_transform(transform)
    return dataset


def get_next_on_error(func: Callable) -> Callable:
    @functools.wraps(func)
    def wrapper_collect_metrics(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.exception(
                f"Error occurred at idx {args} {e}, getting the next item instead."
            )
            current_time = time.time()
            random.seed(current_time)
            random_number = random.randint(0, 10**8)
            args = list(args)
            kwargs["idx"] = kwargs["idx"] + random_number
            args = tuple(args)
            return func(*args, **kwargs)

    return wrapper_collect_metrics


@configurable
class TALIBase(Dataset):
    def __init__(
        self,
        set_name: str,
        tali_dataset_dir,
        modality_list,
        num_samples_per_episode: int,
        rng_seed=42,
        top_k_tali=10,
        image_size=224,
        num_video_frames=5,
        num_audio_frames=16000,
        clip_duration_in_seconds=3.0,
        deterministic_sampling=True,
        dummy_batch_mode: bool = False,
        image_text_model_name: str = "openai/clip-vit-base-patch16",
        audio_model_name: str = "openai/whisper-base",
        use_model_preprocessing: bool = True,
        total_num_samples: Optional[int] = None,
        cache_generated_samples_in_memory: bool = False,
        cache_num_samples: int = 10,
    ):
        super().__init__()
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
        self.dataset = datasets.load_from_disk(
            pathlib.Path(tali_dataset_dir) / f"{set_name}-set",
            keep_in_memory=True,
        )
        self.dataset = self.dataset.with_transform(transform)
        self.num_dataset_samples = len(self.dataset)
        self.dummy_batch_mode = dummy_batch_mode
        self.cache_generated_samples_in_memory = cache_generated_samples_in_memory
        self.cache_num_samples = cache_num_samples

        if self.cache_generated_samples_in_memory:
            self.mem_cache = []

        self.dummy_batch = None
        self.num_samples_per_episode = num_samples_per_episode
        self.num_samples = (
            total_num_samples if total_num_samples is not None else len(self.dataset)
        )
        self.image_text_model_name = image_text_model_name
        self.audio_model_name = audio_model_name
        self.use_model_preprocessing = use_model_preprocessing
        if self.use_model_preprocessing:
            self.transforms = self.build_model_transforms()
        else:
            self.transforms = self.build_basic_transforms()

    def build_model_transforms(self):
        self.image_text_processor = CLIPProcessor.from_pretrained(
            self.image_text_model_name
        )
        self.audio_processor = WhisperProcessor.from_pretrained(self.audio_model_name)

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
                        item.view(-1),
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

    def build_basic_transforms(self):
        return {
            "image": lambda x: (x * 255).to(torch.uint8),
            "text": lambda x: x,
            "audio": lambda x: x.view(-1),
            "video": lambda x: [(item * 255).to(torch.uint8) for item in x],
        }

    def __getitem__(self, idx):
        episode_dict = {}

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

        if self.cache_generated_samples_in_memory:
            self.mem_cache.append(episode_dict)
            if len(self.mem_cache) > self.cache_num_samples:
                self.mem_cache.pop(0)

        return episode_dict

    @get_next_on_error
    def get_sample(self, idx):
        if self.dummy_batch_mode and self.dummy_batch is not None:
            return self.dummy_batch

        if self.num_samples > self.num_dataset_samples:
            idx = idx % self.num_dataset_samples

        sample = self.dataset[idx]

        for key, value in sample.items():
            for transform_key, transform_value in self.transforms.items():
                if transform_key in key and key != "youtube_video_id":
                    if "audio" in key:
                        value = value.unsqueeze(0)

                    if key == "wikipedia_caption_image":
                        if value.to(torch.float32).max() > 255:
                            value = value.to(torch.float32) / 65535.0

                    sample[key] = transform_value(value)
                    sample[key] = (
                        torch.stack(sample[key], dim=0)
                        if isinstance(sample[key], list)
                        else sample[key]
                    )

                    break

        if self.dummy_batch_mode:
            if self.dummy_batch is None:
                self.dummy_batch = sample
            return self.dummy_batch

        return sample

    def __len__(self) -> int:
        return self.num_samples


class CustomConcatDataset(Dataset):
    def __init__(self, datasets):
        self.datasets = datasets

    def __len__(self):
        return sum(len(d) for d in self.datasets)

    def __getitem__(self, idx):
        # logger.info(f"dataset {idx % len(self.datasets)}, sample {idx // len(self.datasets)}")
        return self.datasets[idx % len(self.datasets)][idx // len(self.datasets)]


if __name__ == "__main__":
    import tqdm
    from rich import print
    from rich.traceback import install
    import cProfile
    import pstats
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
