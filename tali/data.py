import logging
import multiprocessing as mp
import pathlib
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from math import floor
from typing import Any, Callable, Dict, List, Optional, Union

import datasets
import numpy as np
import PIL
import torchaudio.transforms as TA
import yaml
from pytorchvideo.transforms import ApplyTransformToKey, ShortSideScale
from rich import print
from torchvision.transforms import Compose
from torchvision.transforms._transforms_video import CenterCropVideo
from tqdm import tqdm

from tali.frames import FrameSelectionMethod, extract_frames_pyav
from tali.utils import enrichen_logger

logger = logging.getLogger(__name__)
logger = enrichen_logger(logger)
logger.setLevel(logging.INFO)

WIKIPEDIA_ENTRY_KEYS = [
    "caption_alt_text_description",
    "caption_reference_description",
    "caption_title_and_reference_description",
    "context_page_description",
    "context_section_description",
    "hierarchical_section_title",
    "page_title",
    "section_title",
]


class ModalityTypes(Enum):
    image = "image"
    audio = "audio"
    video = "video"
    text = "text"


class VideoFramesFormat(Enum):
    PIL = "PIL"
    TENSOR = "TENSOR"


@dataclass
class SubModality:
    parent: ModalityTypes
    name: str


class SubModalityTypes(Enum):
    wikipedia_caption_image = SubModality(
        ModalityTypes.image, "wikipedia_caption_image"
    )
    youtube_random_video_frame = SubModality(
        ModalityTypes.image, "youtube_random_video_sample_image"
    )
    youtube_thumbnail_image = SubModality(
        ModalityTypes.image, "youtube_thumbnail_image"
    )

    wikipedia_caption_text = SubModality(
        ModalityTypes.text, "wikipedia_caption_text"
    )
    wikipedia_title_text = SubModality(
        ModalityTypes.text, "wikipedia_title_text"
    )
    wikipedia_main_body_text = SubModality(
        ModalityTypes.text, "wikipedia_main_body_text"
    )
    youtube_subtitle_text = SubModality(
        ModalityTypes.text, "youtube_subtitle_text"
    )
    youtube_description_text = SubModality(
        ModalityTypes.text, "youtube_description_text"
    )
    youtube_title_text = SubModality(ModalityTypes.text, "youtube_title_text")

    youtube_content_audio = SubModality(
        ModalityTypes.audio, "youtube_content_audio"
    )

    youtube_content_video = SubModality(
        ModalityTypes.video, "youtube_content_video"
    )


class TALIKeys(Enum):
    image = "image"
    image_url = "image_url"
    item_idx = "item_idx"
    wit_features = "wit_features"
    wit_idx = "wit_idx"
    youtube_title_text = "youtube_title_text"
    youtube_description_text = "youtube_description_text"
    youtube_video_content = "youtube_video_content"
    youtube_video_starting_time = "youtube_video_starting_time"
    youtube_subtitle_text = "youtube_subtitle_text"
    youtube_video_size = "youtube_video_size"
    youtube_video_file_path = "youtube_video_file_path"
    wikipedia_caption_text = "wikipedia_caption_text"


@dataclass
class TALIBaseTransformConfig:
    root_filepath: Union[str, pathlib.Path]
    modality_list: List
    rng_seed: int = 42
    image_size: int = 224
    num_video_frames: int = 30
    num_audio_frames: int = 44100
    clip_duration_in_seconds: float = 3
    priority_caption_language: Optional[str] = "en"
    video_frame_duration: int = 30
    video_frames_format: str = VideoFramesFormat.TENSOR


@dataclass
class TALIBaseTransformConfig:
    root_filepath: Union[str, pathlib.Path]
    modality_list: List
    rng_seed: int = 42
    image_size: int = 224
    num_video_frames: int = 30
    num_audio_frames: int = 44100
    clip_duration_in_seconds: float = 3
    priority_caption_language: Optional[str] = "en"
    video_frame_duration: int = 30
    video_frames_format: str = VideoFramesFormat.TENSOR.value


def select_subtitles_between_timestamps(
    subtitle_dict: Dict[str, str],
    starting_timestamp: float,
    ending_timestamp: float,
):
    subtitle_dict = yaml.safe_load(subtitle_dict)
    subtitle_dict = {float(key): value for key, value in subtitle_dict.items()}
    subtitle_dict = dict(sorted(subtitle_dict.items(), key=lambda x: x[0]))
    selected_subtitles = ""
    for subtitle_timestamp, subtitle_text in subtitle_dict.items():
        subtitle_timestamp = float(subtitle_timestamp)
        if (
            float(subtitle_timestamp) >= starting_timestamp
            and float(subtitle_timestamp) <= ending_timestamp
        ):
            subtitle_text = "".join(subtitle_text)
            selected_subtitles += subtitle_text + " "

    return selected_subtitles


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
    video_data: Union[pathlib.Path, bytes, str],
    rng: Optional[np.random.Generator] = None,
    return_video: bool = True,
    return_audio: bool = False,
    return_image: bool = False,
    image_size: int = 224,
    starting_second: int = 0,
    ending_second: Optional[int] = None,
    num_audio_frames: int = 1 * 16000,
    num_video_frames: int = 10,
    video_frame_format: str = VideoFramesFormat.TENSOR,
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
            video_data=video_data,
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
        output["video"] = (
            [convert_to_pil(frame) for frame in video]
            if video_frame_format == VideoFramesFormat.PIL
            else video
            if video_frame_format == VideoFramesFormat.TENSOR
            else None
        )
        if output["video"] is None:
            raise ValueError(
                f"Unknown video frame format {video_frame_format}, "
                f"must be one of {VideoFramesFormat.PIL} or "
                f"{VideoFramesFormat.TENSOR}"
            )

    if return_image:
        if image is None:
            image = extract_frames_pyav(
                video_data=video_data,
                starting_second=starting_second,
                ending_second=ending_second,
                num_frames=1,
                rng=rng,
                modality="video",
                frame_selection_method=FrameSelectionMethod.RANDOM,
                single_image_frame=True,
            )
            image = get_video_tensors(image, image_size)[0]

        output["image"] = (
            convert_to_pil(image)
            if video_frame_format == VideoFramesFormat.PIL
            else image
            if video_frame_format == VideoFramesFormat.TENSOR
            else None
        )

        if output["image"] is None:
            raise ValueError(
                f"Unknown video frame format {video_frame_format}, "
                f"must be one of {VideoFramesFormat.PIL} or "
                f"{VideoFramesFormat.TENSOR}"
            )

    if return_audio:
        audio = extract_frames_pyav(
            video_data=video_data,
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
    resampler = TA.Resample(
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


def download_dataset_via_hub(
    dataset_name: str,
    dataset_download_path: pathlib.Path,
    num_download_workers: int = mp.cpu_count(),
):
    import huggingface_hub as hf_hub

    download_folder = hf_hub.snapshot_download(
        repo_id=dataset_name,
        repo_type="dataset",
        cache_dir=dataset_download_path,
        resume_download=True,
        max_workers=num_download_workers,
        ignore_patterns=[],
    )

    return pathlib.Path(download_folder) / "data"


def load_dataset_via_hub(
    dataset_download_path: pathlib.Path,
    num_download_workers: int = mp.cpu_count(),
    dataset_name: Optional[str] = None,
    streaming: bool = False,
):
    from datasets import Features, Image, Sequence, Value

    if not streaming:
        dataset_path = download_dataset_via_hub(
            dataset_download_path=dataset_download_path,
            num_download_workers=num_download_workers,
            dataset_name=dataset_name,
        )
        # Building a list of file paths for validation set
    else:
        dataset_path = dataset_download_path

    train_files = [
        file.as_posix()
        for file in pathlib.Path(dataset_path).glob("*.parquet")
        if "train" in file.as_posix()
    ]
    val_files = [
        file.as_posix()
        for file in pathlib.Path(dataset_path).glob("*.parquet")
        if "val" in file.as_posix()
    ]
    test_files = [
        file.as_posix()
        for file in pathlib.Path(dataset_path).glob("*.parquet")
        if "test" in file.as_posix()
    ]
    print(
        f"Found {len(train_files)} for training set, {len(val_files)} for validation set and {len(test_files)} files for testing set"
    )
    data_files = {
        "test": test_files,
        "val": val_files,
        "train": train_files,
    }

    features = Features(
        {
            "image": Image(
                decode=True
            ),  # Set `decode=True` if you want to decode the images, otherwise `decode=False`
            "image_url": Value("string"),
            "item_idx": Value("int64"),
            "wit_features": Sequence(
                {
                    "attribution_passes_lang_id": Value("bool"),
                    "caption_alt_text_description": Value("string"),
                    "caption_reference_description": Value("string"),
                    "caption_title_and_reference_description": Value("string"),
                    "context_page_description": Value("string"),
                    "context_section_description": Value("string"),
                    "hierarchical_section_title": Value("string"),
                    "is_main_image": Value("bool"),
                    "language": Value("string"),
                    "page_changed_recently": Value("bool"),
                    "page_title": Value("string"),
                    "page_url": Value("string"),
                    "section_title": Value("string"),
                }
            ),
            "wit_idx": Value("int64"),
            "youtube_title_text": Value("string"),
            "youtube_description_text": Value("string"),
            "youtube_video_content": Value("binary"),
            "youtube_video_starting_time": Value("string"),
            "youtube_subtitle_text": Value("string"),
            "youtube_video_size": Value("int64"),
            "youtube_video_file_path": Value("string"),
        }
    )

    dataset = datasets.load_dataset(
        "parquet" if dataset_name is None else dataset_name,
        data_files=data_files,
        features=features,
        num_proc=mp.cpu_count(),
        cache_dir=dataset_download_path / "cache",
        streaming=streaming,
    )
    return dataset


def default_transforms():
    from transformers import CLIPProcessor, WhisperProcessor

    image_text_model_name = "openai/clip-vit-base-patch16"
    audio_model_name = "openai/whisper-base"
    image_text_processor = CLIPProcessor.from_pretrained(image_text_model_name)
    audio_processor = WhisperProcessor.from_pretrained(audio_model_name)

    def image_transforms(x):
        if isinstance(x, PIL.Image.Image):
            temp_x = np.array(x)
            if temp_x.max() > 255:
                temp_x = temp_x / 65535.0
                temp_x = (temp_x * 255).astype(np.uint8)
                x = PIL.Image.fromarray(temp_x)

        return image_text_processor(
            images=x, return_tensors="pt"
        ).pixel_values.squeeze(1)

    def text_transforms(x):
        return image_text_processor(
            text=x, return_tensors="pt", padding=True, truncation=True
        ).input_ids.squeeze(0)

    def audio_transforms(x):
        return torch.cat(
            [
                audio_processor(
                    item.view(-1),
                    sampling_rate=16000,
                    return_tensors="pt",
                ).input_features
                for item in x
            ]
        )

    def video_transforms(x):
        return torch.stack(
            [image_transforms(image) for image in x],
            dim=0,
        )

    return (
        image_transforms,
        text_transforms,
        audio_transforms,
        video_transforms,
    )


class TALIBaseTransform:
    def __init__(
        self,
        cache_dir: pathlib.Path,
        config: TALIBaseTransformConfig,
        text_tokenizer: Optional[Callable] = None,
        image_tokenizer: Optional[Callable] = None,
        audio_tokenizer: Optional[Callable] = None,
        video_tokenizer: Optional[Callable] = None,
    ):
        self.cache_dir = cache_dir
        self.config = config
        self.text_tokenizer = text_tokenizer
        self.image_tokenizer = image_tokenizer
        self.audio_tokenizer = audio_tokenizer
        self.video_tokenizer = video_tokenizer

        self.select_subtitles_between_timestamps = (
            select_subtitles_between_timestamps
        )
        self.video_transform = self.build_video_loader()

    def build_video_loader(self):
        def loader(
            x: bytes | str | pathlib.Path,
            start: int,
            end: int,
            seed: int,
            return_video: bool = False,
            return_audio: bool = False,
            return_image: bool = False,
        ):
            return videoclip_to_video_audio_tensors(
                video_data=x,
                image_size=self.config.image_size,
                starting_second=start,
                ending_second=end,
                return_video=return_video,
                return_audio=return_audio,
                return_image=return_image,
                num_audio_frames=self.config.num_audio_frames,
                num_video_frames=self.config.num_video_frames,
                rng=np.random.RandomState(seed),
                video_frame_format=self.config.video_frames_format,
            )

        return loader

    def _process_wikipedia_text(self, wikipedia_features: dict):
        output_dict = dict()
        for language in wikipedia_features["language"]:
            language_idx = wikipedia_features["language"].index(language)
            wit_text = {
                key: wikipedia_features[key][language_idx]
                for key in WIKIPEDIA_ENTRY_KEYS
                if wikipedia_features[key][language_idx] is not None
            }
            output_dict[language] = wit_text
        return output_dict

    def _process_youtube_subtitles(
        self, youtube_subtitle_text: str, youtube_video_starting_time: int
    ):
        return (
            "<ysub> "
            + select_subtitles_between_timestamps(
                subtitle_dict=youtube_subtitle_text,
                starting_timestamp=int(youtube_video_starting_time),
                ending_timestamp=int(youtube_video_starting_time)
                + int(self.config.video_frame_duration),
            )
            + " </ysub>"
        )

    def _process_text(self, input_dict: Dict[str, Any]):
        wikipedia_text_content = self._process_wikipedia_text(
            input_dict[TALIKeys.wit_features.value]
        )
        output_dict = {
            SubModalityTypes.wikipedia_caption_text.value.name: wikipedia_text_content,
            SubModalityTypes.youtube_description_text.value.name: input_dict[
                TALIKeys.youtube_description_text.value
            ],
            SubModalityTypes.youtube_title_text.value.name: input_dict[
                TALIKeys.youtube_title_text.value
            ],
            SubModalityTypes.youtube_subtitle_text.value.name: self._process_youtube_subtitles(
                youtube_subtitle_text=input_dict[
                    TALIKeys.youtube_subtitle_text.value
                ],
                youtube_video_starting_time=input_dict[
                    TALIKeys.youtube_video_starting_time.value
                ],
            ),
        }

        if self.text_tokenizer is not None:
            for key, value in output_dict.items():
                if isinstance(value, str):
                    output_dict[key] = self.text_tokenizer(value)
                elif isinstance(value, dict):
                    item_dict = {}
                    for sub_key, sub_value in value.items():
                        if isinstance(sub_value, str):
                            item_dict[sub_key] = self.text_tokenizer(sub_value)
                        elif isinstance(sub_value, dict):
                            item_dict[sub_key] = {}
                            for (
                                sub_sub_key,
                                sub_sub_value,
                            ) in sub_value.items():
                                item_dict[sub_key][
                                    sub_sub_key
                                ] = self.text_tokenizer(sub_sub_value)
                    output_dict[key] = item_dict

        return output_dict

    def _process_audio(
        self, input_dict: Dict[str, Any], audio: Optional[None]
    ):
        output_dict = {}
        if SubModalityTypes.youtube_content_audio in self.config.modality_list:
            output_dict[SubModalityTypes.youtube_content_audio.value.name] = (
                audio
                if audio is not None
                else self.video_transform(
                    x=input_dict[TALIKeys.youtube_video_content.value],
                    start=0,
                    end=30,
                    seed=int(input_dict[TALIKeys.item_idx.value]),
                    return_audio=True,
                )["audio"],
            )

        if self.audio_tokenizer is not None:
            for key, value in output_dict.items():
                output_dict[key] = self.audio_tokenizer(value)

        return output_dict

    def _process_image(
        self, input_dict: Dict[str, Any], image: Optional[None]
    ):
        output_dict = {}
        if (
            SubModalityTypes.youtube_random_video_frame
            in self.config.modality_list
        ):
            output_dict[
                SubModalityTypes.youtube_random_video_frame.value.name
            ] = (
                image
                if image is not None
                else self.video_transform(
                    x=input_dict[TALIKeys.youtube_video_content.value],
                    start=0,
                    end=30,
                    seed=int(input_dict[TALIKeys.item_idx.value]),
                    return_image=True,
                )["image"]
            )

        if (
            SubModalityTypes.wikipedia_caption_image
            in self.config.modality_list
        ):
            output_dict[
                SubModalityTypes.wikipedia_caption_image.value.name
            ] = input_dict[TALIKeys.image.value]

        if self.image_tokenizer is not None:
            for key, value in output_dict.items():
                output_dict[key] = self.image_tokenizer(value)

        return output_dict

    def _process_video(
        self, input_dict: Dict[str, Any], video: [Optional] = None
    ):
        output_dict = {}
        if SubModalityTypes.youtube_content_video in self.config.modality_list:
            output_dict = {
                SubModalityTypes.youtube_content_video.value.name: video
                if video is not None
                else self.video_transform(
                    x=input_dict[TALIKeys.youtube_video_content.value],
                    start=0,
                    end=30,
                    seed=int(input_dict[TALIKeys.item_idx.value]),
                    return_video=True,
                )["video"],
            }
        if self.video_tokenizer is not None:
            for key, value in output_dict.items():
                output_dict[key] = self.video_tokenizer(value)

        return output_dict

    def _apply_transform(self, input_dict: Dict[str, Any]):
        output_dict = {}

        output_dict[TALIKeys.wit_idx.value] = [
            input_dict[TALIKeys.wit_idx.value]
        ]

        output_dict[TALIKeys.item_idx.value] = [
            input_dict[TALIKeys.item_idx.value]
        ]

        youtube_video = None
        youtube_audio = None
        youtube_image = None
        if SubModalityTypes.youtube_content_video in self.config.modality_list:
            youtube_features = self.video_transform(
                x=input_dict[TALIKeys.youtube_video_content.value],
                start=0,
                end=30,
                seed=int(input_dict[TALIKeys.item_idx.value]),
                return_video=True,
                return_audio=SubModalityTypes.youtube_content_audio.value
                in self.config.modality_list,
                return_image=SubModalityTypes.youtube_random_video_frame.value
                in self.config.modality_list,
            )
            youtube_video = youtube_features["video"]
            youtube_audio = (
                youtube_features["audio"]
                if "audio" in youtube_features
                else None
            )
            youtube_image = (
                youtube_features["image"]
                if "image" in youtube_features
                else None
            )

        output_dict.update(self._process_text(input_dict=input_dict))
        output_dict.update(
            self._process_audio(
                input_dict=input_dict,
                audio=youtube_audio,
            )
        )
        output_dict.update(
            self._process_image(
                input_dict=input_dict,
                image=youtube_image,
            )
        )
        output_dict.update(
            self._process_video(
                input_dict=input_dict,
                video=youtube_video,
            )
        )

        return output_dict

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

        if isinstance(input_dict["item_idx"], list):
            output_dict = defaultdict(list)
            for idx in range(len(input_dict["item_idx"])):
                input_dict_ = {
                    key: input_dict[key][idx] for key in input_dict.keys()
                }
                output_dict_ = self._apply_transform(input_dict_)
                for key in output_dict_.keys():
                    output_dict[key].append(output_dict_[key])
        else:
            output_dict = self._apply_transform(input_dict)

        return output_dict


if __name__ == "__main__":
    import torch

    dataset_cache = pathlib.Path("/disk/scratch_fast0/tali/")
    dataset = load_dataset_via_hub(dataset_cache, dataset_name="Antreas/TALI")[
        "test"
    ]

    (
        image_transforms,
        text_transforms,
        audio_transforms,
        video_transforms,
    ) = default_transforms()

    demo_transform = TALIBaseTransform(
        cache_dir=dataset_cache / "cache",
        text_tokenizer=text_transforms,
        image_tokenizer=image_transforms,
        audio_tokenizer=audio_transforms,
        video_tokenizer=video_transforms,
        config=TALIBaseTransformConfig(
            root_filepath=dataset_cache,
            modality_list=[
                SubModalityTypes.youtube_content_video,
                SubModalityTypes.youtube_content_audio,
                SubModalityTypes.youtube_random_video_frame,
                SubModalityTypes.youtube_subtitle_text,
                SubModalityTypes.youtube_description_text,
                SubModalityTypes.youtube_title_text,
                SubModalityTypes.wikipedia_caption_image,
                SubModalityTypes.wikipedia_caption_text,
                SubModalityTypes.wikipedia_main_body_text,
                SubModalityTypes.wikipedia_title_text,
            ],
            video_frames_format=VideoFramesFormat.PIL,
        ),
    )

    for sample in tqdm(dataset):
        sample = demo_transform(sample)
        print(list(sample.keys()))
        for key, value in sample.items():
            if hasattr(value, "shape"):
                print(key, value.shape)
            elif isinstance(value, torch.Tensor):
                print(key, value.shape)
            elif hasattr(value, "__len__"):
                print(key, len(value))
            print(key, type(value))

        break
