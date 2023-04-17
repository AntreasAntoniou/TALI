import json
import os
import pathlib
import random
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from math import floor
from typing import Any, Callable, Dict, List, Optional, Union
import datasets

import numpy as np
import PIL
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import torch
import torch.nn as nn
import tqdm
from hydra_zen import builds
from pytorchvideo.data.encoded_video import EncodedVideo
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    ShortSideScale,
    UniformTemporalSubsample,
)
from rich import print
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torchvision.transforms import Compose, RandomCrop, Resize, ToTensor
from torchvision.transforms._transforms_video import CenterCropVideo
from transformers import CLIPProcessor
import datasets

from tali_wit.decorators import configurable
from tali_wit.utils import get_logger, load_json

logger = get_logger(__name__)


@dataclass
class SplitType:
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


@dataclass
class MultiModalInput:
    image: Any = None
    audio: Any = None
    video: Any = None
    text: Any = None


def find_filepaths_with_extension(
    dir_path: str, extension: str, limit_num_files: Optional[int]
):
    filepaths = []

    with tqdm.tqdm(total=12000000) as pbar:
        for path in pathlib.Path(dir_path).iterdir():
            if path.suffix == extension and path.is_file():
                filepaths.append(str(path))
                if limit_num_files is not None:
                    if len(filepaths) >= limit_num_files:
                        break
            pbar.update(1)

    return filepaths


def extract_captions_from_file(filepath: str):
    info_dict = load_json(filepath=filepath)
    return info_dict["edge_media_to_caption"]["edges"][0]["node"]["text"]


def check_if_image_has_matching_info_file(image_path: str):
    if isinstance(image_path, pathlib.Path):
        image_path = str(image_path)
    info_file_path = pathlib.Path(
        image_path.replace("image", "info")
    ).with_suffix(".info")
    return info_file_path.exists()


def get_user_and_post_id_from_image_path(image_path: str):
    username, post_id = image_path.split("/")[-1].split("-")
    post_id = post_id.split(".")[0]

    return username, post_id


def generate_post_paths_from_user_name_and_post_id(
    username: str,
    post_id: str,
    post_image_dir: str,
    post_info_dir: str,
):
    image_path = os.path.join(post_image_dir, f"{username}-{post_id}.jpg")
    info_path = os.path.join(post_info_dir, f"{username}-{post_id}.info")

    return image_path, info_path


@dataclass
class ChallengeSamplesSourceTypes:
    WITHIN_USER: str = "within_user"
    ACROSS_USERS: str = "across_users"


def rank_user_items_by_clip_score(username_filepath: pathlib.Path):
    user_table = pq.read_table(username_filepath).to_pandas()
    return user_table.sort_values(by="similarity", ascending=False)


def get_ranked_filepaths_from_user(
    username_filepath: pathlib.Path, top_k_percent_to_return: int
):
    ranked_user_items = rank_user_items_by_clip_score(username_filepath)
    ranked_filepath_list = ranked_user_items["filepath"].tolist()
    top_k_percent_to_return = int(
        len(ranked_filepath_list) * top_k_percent_to_return / 100
    )
    return ranked_filepath_list[:top_k_percent_to_return]


class ToThreeChannels(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, image):
        if image.shape[0] == 1:
            return image.repeat(3, 1, 1)
        elif image.shape[0] == 2:
            return torch.cat([image, image[0].unsqueeze(0)], dim=0)
        elif image.shape[0] == 3:
            return image
        elif image.shape[0] == 4:
            return image[:3]
        else:
            raise ValueError(f"Image shape {image.shape} not supported")


def default_image_transforms(image_size: int = 224):
    return Compose(
        [
            Resize(image_size),
            RandomCrop(image_size),
            ToTensor(),
            ToThreeChannels(),
        ]
    )


default_image_transforms_config = builds(default_image_transforms)


def dict_to_summary(batch: Dict):
    summary_dict = defaultdict(list)

    if not isinstance(batch, dict) and not isinstance(batch, list):
        batch = [batch.__dict__]

    if isinstance(batch, dict):
        batch = [batch]

    for item in batch:
        for key, value in item.items():
            # print(value)
            if hasattr(value, "shape"):
                summary_dict[key].append((str(value.shape), str(value.dtype)))
            elif hasattr(value, "__len__"):
                summary_dict[key].append(len(value))
            elif value is None:
                summary_dict[key].append(None)
            else:
                summary_dict[key].append(value)

    return summary_dict


def dataclass_collate(batch):
    """Collate data from a list of dataclass objects.

    Args:
        batch (list): List of dataclass objects.

    Returns:
        dict: Dictionary of values from the dataclass objects.
    """
    batch = list(filter(lambda x: x is not None, batch))
    batch = list(filter(lambda x: len(list(x.keys())) != 0, batch))

    # for sample in batch:
    #     for key in list(sample.keys()):
    #         if (
    #             "text" in key
    #             and len(sample[key]) < 77
    #             and isinstance(sample[key], torch.Tensor)
    #         ):
    #             sample[key] = torch.cat(
    #                 [sample[key], 49407 * torch.ones(77 - len(sample[key])).long()]
    #             )

    try:
        if isinstance(batch[0], dict) or not hasattr(
            batch[0], "__dataclass_fields__"
        ):
            batch = default_collate(batch)
            batch = {key: batch[key][0] for key in batch.keys()}
            return batch
        else:
            batched_dict = {
                key: default_collate([getattr(sample, key) for sample in batch])
                if getattr(batch[0], key) != None
                else None
                for key in batch[0].__dict__.keys()
            }
            batched_dict = {key: batched_dict[key][0] for key in batched_dict}
            return batch[0].__class__(**batched_dict)
    except Exception as e:
        print(
            f"Current batch we fucked up on {json.dumps(dict_to_summary(batch), indent=4)}"
        )
        raise e


def get_image_transforms_instait():
    return Compose([Resize((224, 224)), ToTensor(), ToThreeChannels()])


import pathlib
from dataclasses import dataclass
from typing import List
from tali_wit.data import dataclass_collate


@dataclass
class TALISchema:
    wit_idx: pa.int64()
    term_idx: pa.int64()
    sort_type: pa.string()
    age_restricted: pa.bool_()
    author: pa.string()
    channel_id: pa.string()
    channel_url: pa.string()
    description: pa.string()
    embed_url: pa.string()
    keywords: pa.list_(pa.string())
    length: pa.int64()
    publish_date: pa.timestamp("us")
    thumbnail_url: pa.string()
    title: pa.string()
    video_id: pa.string()
    video_store_filepath: pa.string()
    views: pa.string()
    watch_url: pa.string()


tali_schema = list(TALISchema.__dict__["__annotations__"].items())
tali_schema = pa.schema(tali_schema)


@dataclass
class VideoCLIPScoreSchema:
    wit_idx: pa.int32()
    term_idx: pa.int32()
    video_id: pa.string()
    filepath: pa.string()
    reference_text: pa.string()
    scores_sorted_idx: pa.list_(pa.int32())
    scores_sorted: pa.list_(pa.float32())


video_score_schema = list(
    VideoCLIPScoreSchema.__dict__["__annotations__"].items()
)
video_score_schema = pa.schema(video_score_schema)


@dataclass
class CrossModalityTypes:
    image_to_text = "image_to_text"
    image_to_audio = "image_to_audio"
    image_to_audio = "image_to_video"
    text_to_audio = "text_to_audio"
    text_to_video = "text_to_video"
    audio_to_video = "audio_to_video"


class BaseModalityTypes(str, Enum):
    image = "image"
    audio = "audio"
    video = "video"
    text = "text"


class SubModalityTypes(str, Enum):
    wikipedia_caption_image = "wikipedia_caption_image"
    youtube_random_video_sample_image = "youtube_random_video_sample_image"
    youtube_thumbnail_image = "youtube_thumbnail_image"

    wikipedia_caption_text = "wikipedia_caption_text"
    wikipedia_title_text = "wikipedia_title_text"
    wikipedia_main_body_text = "wikipedia_main_body_text"
    youtube_subtitle_text = "youtube_subtitle_text"
    youtube_description_text = "youtube_description_text"
    youtube_title_text = "youtube_title_text"

    youtube_content_audio = "youtube_content_audio"

    youtube_content_video = "youtube_content_video"


@dataclass
class AnyModalSample:
    modality: str
    sub_modality: str
    shape: tuple


class ModalityTypes(Enum):
    wit_image = AnyModalSample(
        modality=BaseModalityTypes.image,
        sub_modality=SubModalityTypes.wikipedia_caption_image,
        shape=("batch_size", "channel", "height", "width"),
    )
    youtube_image = AnyModalSample(
        modality=BaseModalityTypes.image,
        sub_modality=SubModalityTypes.youtube_random_video_sample_image,
        shape=("batch_size", "channel", "height", "width"),
    )
    youtube_thumbnail = AnyModalSample(
        modality=BaseModalityTypes.image,
        sub_modality=SubModalityTypes.youtube_thumbnail_image,
        shape=("batch_size", "channel", "height", "width"),
    )

    wit_caption = AnyModalSample(
        modality=BaseModalityTypes.text,
        sub_modality=SubModalityTypes.wikipedia_caption_text,
        shape=("batch_size", "sequence_length"),
    )
    wit_title = AnyModalSample(
        modality=BaseModalityTypes.text,
        sub_modality=SubModalityTypes.wikipedia_title_text,
        shape=("batch_size", "sequence_length"),
    )
    wit_main_body = AnyModalSample(
        modality=BaseModalityTypes.text,
        sub_modality=SubModalityTypes.wikipedia_main_body_text,
        shape=("batch_size", "sequence_length"),
    )
    youtube_subtitles = AnyModalSample(
        modality=BaseModalityTypes.text,
        sub_modality=SubModalityTypes.youtube_subtitle_text,
        shape=("batch_size", "sequence_length"),
    )
    youtube_description = AnyModalSample(
        modality=BaseModalityTypes.text,
        sub_modality=SubModalityTypes.youtube_description_text,
        shape=("batch_size", "sequence_length"),
    )
    youtube_title = AnyModalSample(
        modality=BaseModalityTypes.text,
        sub_modality=SubModalityTypes.youtube_title_text,
        shape=("batch_size", "sequence_length"),
    )

    youtube_audio = AnyModalSample(
        modality=BaseModalityTypes.audio,
        sub_modality=SubModalityTypes.youtube_content_audio,
        shape=("batch_size", "sequence_length", "channel", "audio_stream"),
    )

    youtube_video = AnyModalSample(
        modality=BaseModalityTypes.video,
        sub_modality=SubModalityTypes.youtube_content_video,
        shape=("batch_size", "sequence_length", "channel", "height", "width"),
    )

    def __str__(self):
        return self.value


@dataclass
class ModalityDataSample:
    data: Any
    modality_type: Any


def videoclip_to_video_audio_tensors(
    video_path: pathlib.Path,
    return_video: bool,
    return_audio: bool,
    image_size: int,
    clip_duration_in_seconds: float,
    rng: np.random.Generator,
):
    video: EncodedVideo = EncodedVideo.from_path(video_path)
    video_duration_in_seconds = float(video.duration)
    try:
        clip_start_sec = rng.randint(
            0, int(floor(video_duration_in_seconds - clip_duration_in_seconds))
        )
    except ValueError:
        raise ValueError(
            (
                f"Video duration is less than clip duration {video_duration_in_seconds, clip_duration_in_seconds}"
            )
        )

    # Get clip
    video_data = video.get_clip(
        start_sec=clip_start_sec,
        end_sec=clip_start_sec + clip_duration_in_seconds,
    )

    audio_sample_rate = floor(
        video_data["audio"].shape[0] / video_duration_in_seconds
    )
    video_sample_rate = floor(
        video_data["video"].shape[1] / video_duration_in_seconds
    )
    video_tensors = None
    audio_tensors = None
    output_dict = {
        "full_video_shape": video_data["video"].shape,
        "full_audio_shape": video_data["audio"].shape,
    }

    if return_video:
        # Compose video data transforms
        video_transform = ApplyTransformToKey(
            key="video",
            transform=Compose(
                [
                    ShortSideScale(size=image_size),
                    CenterCropVideo(crop_size=(image_size, image_size)),
                    UniformTemporalSubsample(
                        num_samples=int(floor(video_duration_in_seconds * 5)),
                        temporal_dim=1,
                    ),
                ]
            ),
        )

        video_data = video_transform(video_data)
        video_tensors = video_data["video"].permute(1, 0, 2, 3) / 255.0
        output_dict["full_video_shape"] = video_data["video"].shape

        output_dict[
            ModalityTypes.youtube_video.value.sub_modality
        ] = ModalityDataSample(
            data=video_tensors, modality_type=ModalityTypes.youtube_video.value
        )

    if return_audio:
        audio_transform = ApplyTransformToKey(
            key="audio",
            transform=Compose(
                [
                    UniformTemporalSubsample(
                        num_samples=floor(video_duration_in_seconds * 16000),
                        temporal_dim=0,
                    ),
                ]
            ),
        )

        audio_tensors = audio_transform(video_data)["audio"]

        output_dict[
            ModalityTypes.youtube_audio.value.sub_modality
        ] = ModalityDataSample(
            data=audio_tensors,
            modality_type=ModalityTypes.youtube_audio.value,
        )

    return output_dict


@dataclass
class TermIDTranslation:
    title_prompted = 0
    caption_prompted = 1


@dataclass
class WitSample:
    wikipedia_caption_image: Any
    wikipedia_caption_text: Any
    wikipedia_main_body_text: Any
    wikipedia_title_text: Any


def get_wit_sample(
    dataset: Any,
    wit_index: int,
    language_id: str = "en",
    image_size: int = 224,
    modality_list: List[ModalityTypes] = None,
    if_none_return_random: bool = False,
):
    data_dict = get_language_specific_entries(
        wit_idx=wit_index, wit_entry=dataset[int(wit_index)]
    )

    image = ModalityDataSample(
        data=data_dict.image, modality_type=ModalityTypes.wit_image.value
    )

    image_caption = ModalityDataSample(
        data=data_dict.caption_reference_description,
        modality_type=ModalityTypes.wit_caption.value,
    )

    title_short_description = ModalityDataSample(
        data=data_dict.context_page_description,
        modality_type=ModalityTypes.wit_main_body.value,
    )
    title = ModalityDataSample(
        data=data_dict.page_title,
        modality_type=ModalityTypes.wit_title.value,
    )

    text_not_none = []
    for item in [image_caption, title_short_description, title]:
        if item.data is not None:
            text_not_none.append(item)

    for item in [image_caption, title_short_description, title]:
        if any(
            item.modality_type == modality_type
            for modality_type in modality_list
        ):
            if item.data is None:
                item.data = random.choice(text_not_none).data

    output = WitSample(
        wikipedia_caption_image=image,
        wikipedia_caption_text=image_caption,
        wikipedia_main_body_text=title_short_description,
        wikipedia_title_text=title,
    )

    output_dict = output.__dict__
    output_dict_cache = output_dict.copy()
    for key, value in output_dict_cache.items():
        if not any(
            key == modality.sub_modality.replace("SubModalityTypes.", "")
            for modality in modality_list
        ):
            del output_dict[key]

    if "wikipedia_caption_image" in output_dict:
        output_dict["wikipedia_caption_image"].data = default_image_transforms(
            image_size=image_size
        )(output_dict["wikipedia_caption_image"].data)

    if "wikipedia_caption_text" in output_dict:
        output_dict["wikipedia_caption_text"].data = (
            "<wcap> " + output_dict["wikipedia_caption_text"].data + " </wcap>"
        )

    if "wikipedia_main_body_text" in output_dict:
        output_dict["wikipedia_main_body_text"].data = (
            "<wbody> "
            + output_dict["wikipedia_main_body_text"].data
            + " </wbody>"
        )

    if "wikipedia_title_text" in output_dict:
        output_dict["wikipedia_title_text"].data = (
            "<wtitle> "
            + output_dict["wikipedia_title_text"].data
            + " </wtitle>"
        )

    return output_dict


def select_subtitles_between_timestamps(
    subtitle_dict: Dict[str, str],
    starting_timestamp: float,
    ending_timestamp: float,
):
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


def get_base_modality(submodality: str):
    for item in list(ModalityTypes):
        if item.value.sub_modality == submodality:
            return item.value.modality


def get_tali_sample(
    video_id: int,
    modality_list: List[ModalityTypes],
    rng_seed: int = 42,
    root_filepath: pathlib.Path = pathlib.Path("/data/"),
    top_k: int = 10,
    image_size: int = 224,
    clip_duration_in_seconds: float = 30,
):
    output_dict = {}
    rng = np.random.RandomState(rng_seed)

    if isinstance(root_filepath, str):
        root_filepath = pathlib.Path(root_filepath)

    video_filepath = root_filepath / pathlib.Path(
        f"video_clip_scores.parquet/relevance/{video_id}.parquet"
    )

    taliwit_table = ds.dataset(
        video_filepath, schema=video_score_schema
    ).to_table()
    table: VideoCLIPScoreSchema = taliwit_table.to_pandas()

    table_idx = table.video_id.tolist().index(video_id)
    video_path = table.filepath[table_idx]
    scores_sorted_idx = table.scores_sorted_idx[table_idx]
    reference_text = table.reference_text[table_idx]
    wit_index = table.wit_idx[table_idx]
    term_idx = table.term_idx[table_idx]

    #######################################################################################################################
    ## Get video clip that includes image frames and audio frames

    video_data_root = root_filepath / "video_data.parquet"

    video_data_filepath = pathlib.Path(video_data_root / video_path).parent
    subclip_filepaths = list(video_data_filepath.rglob("*.mp4"))[:top_k]
    selected_subclip = rng.choice(subclip_filepaths)  # youtube videoclip

    if (
        ModalityTypes.youtube_video.value in modality_list
        or ModalityTypes.youtube_audio.value in modality_list
        or ModalityTypes.youtube_image.value in modality_list
    ):
        output_dict = videoclip_to_video_audio_tensors(
            selected_subclip,
            return_video=ModalityTypes.youtube_video.value in modality_list
            or ModalityTypes.youtube_image.value in modality_list,
            return_audio=ModalityTypes.youtube_audio.value in modality_list,
            image_size=image_size,
            clip_duration_in_seconds=clip_duration_in_seconds,
            rng=rng,
        )

    #######################################################################################################################
    ## Get youtube subtitles
    if ModalityTypes.youtube_subtitles.value in modality_list:
        clip_subtitles_filepath = (
            root_filepath
            / pathlib.Path("captions.parquet/relevance/")
            / str(int(wit_index / 1000))
            / str(wit_index)
            / str(term_idx)
            / str(video_id)
            / "captions.json"
        )

        subtitles = load_json(clip_subtitles_filepath)
        timestamp = float(
            pathlib.Path(selected_subclip).stem.replace("360p_", "")
        )
        subtitles = select_subtitles_between_timestamps(
            subtitle_dict=subtitles,
            starting_timestamp=timestamp,
            ending_timestamp=timestamp + 30,
        )

        output_dict[
            ModalityTypes.youtube_subtitles.value.sub_modality
        ] = ModalityDataSample(
            data="<sub> " + subtitles + "</sub>",
            modality_type=ModalityTypes.youtube_subtitles.value,
        )

    if (
        ModalityTypes.youtube_title.value in modality_list
        or ModalityTypes.youtube_description.value in modality_list
    ):
        wit_to_tali_entry_filepath = (
            root_filepath
            / "wit_to_video_paths.parquet/relevance/"
            / f"{wit_index // 1000}/{wit_index}/"
        )

        wit_to_tali_entry_table = (
            ds.dataset(wit_to_tali_entry_filepath, schema=tali_schema)
            .to_table()
            .to_pandas()
        )

        video_ids = wit_to_tali_entry_table.video_id.tolist()
        video_idx = video_ids.index(video_id)

        if ModalityTypes.youtube_title.value in modality_list:
            output_dict[
                ModalityTypes.youtube_title.value.sub_modality
            ] = ModalityDataSample(
                data="<ytitle> "
                + wit_to_tali_entry_table.title[video_idx]
                + " </ytitle>",
                modality_type=ModalityTypes.youtube_title.value,
            )

        if ModalityTypes.youtube_description.value in modality_list:
            output_dict[
                ModalityTypes.youtube_description.value.sub_modality
            ] = ModalityDataSample(
                data="<ydesc>"
                + wit_to_tali_entry_table.description[video_idx]
                + "</ydesc>",
                modality_type=ModalityTypes.youtube_description.value,
            )
    # print(list(output_dict.keys()))
    return wit_index, output_dict


def get_sample_from_wit_index(
    dataset: Any,
    wit_index: int,
    rng_seed: int,
    modality_list: List[ModalityTypes],
    root_filepath: pathlib.Path = pathlib.Path("/data/"),
    top_k_wit: int = 10,
    language_id: str = "en",
    image_size: int = 224,
    clip_duration_in_seconds: float = 30,
):
    if isinstance(root_filepath, str):
        root_filepath = pathlib.Path(root_filepath)

    rng = np.random.RandomState(rng_seed)
    output_dict = {}
    wit_output = get_wit_sample(
        dataset=dataset,
        wit_index=wit_index,
        language_id=language_id,
        image_size=image_size,
    )
    for key, value in wit_output.__dict__.items():
        # print([modality.sub_modality for modality in modality_list], key)
        if any(
            key == modality.sub_modality.replace("SubModalityTypes.", "")
            for modality in modality_list
        ):
            output_dict[key] = value

    if any(
        "youtube" in modality.sub_modality.replace("SubModalityTypes.", "")
        for modality in modality_list
    ):
        wit_to_tali_entry_filepath = (
            root_filepath
            / "wit_to_video_paths.parquet/relevance/"
            / f"{wit_index // 1000}/{wit_index}/"
        )

        wit_to_tali_entry_table = (
            ds.dataset(wit_to_tali_entry_filepath, schema=tali_schema)
            .to_table()
            .to_pandas()
        )

        num_videos = len(wit_to_tali_entry_table.wit_idx)

        video_selected_idx = rng.randint(0, num_videos)

        wit_index, data_dict = get_tali_sample(
            video_id=wit_to_tali_entry_table.video_id[video_selected_idx],
            rng_seed=rng_seed,
            root_filepath=root_filepath,
            modality_list=modality_list,
            top_k=top_k_wit,
            image_size=image_size,
            clip_duration_in_seconds=clip_duration_in_seconds,
        )

        for key, value in data_dict.items():
            output_dict[key] = value

    return wit_index, output_dict


def get_sample_from_video_id(
    dataset: Any,
    video_id: str,
    rng_seed: int,
    modality_list: List[ModalityTypes],
    root_filepath: pathlib.Path = pathlib.Path("/data/"),
    top_k_tali: int = 10,
    language_id: str = "en",
    image_size: int = 224,
    clip_duration_in_seconds: float = 30,
):
    rng = np.random.RandomState(rng_seed)
    output_dict = {}

    if isinstance(root_filepath, str):
        root_filepath = pathlib.Path(root_filepath)

    ### Fetch video data ###
    if (
        ModalityTypes.youtube_audio.value in modality_list
        or ModalityTypes.youtube_image.value in modality_list
        or ModalityTypes.youtube_video.value in modality_list
        or ModalityTypes.youtube_subtitles.value in modality_list
        or ModalityTypes.youtube_description.value in modality_list
        or ModalityTypes.youtube_thumbnail.value in modality_list
        or ModalityTypes.youtube_title.value in modality_list
        or ModalityTypes.youtube_image.value in modality_list
    ):
        wit_index, output_dict = get_tali_sample(
            video_id=video_id,
            rng_seed=rng_seed,
            root_filepath=root_filepath,
            modality_list=modality_list,
            top_k=top_k_tali,
            image_size=image_size,
            clip_duration_in_seconds=clip_duration_in_seconds,
        )

    if (
        ModalityTypes.wit_caption.value in modality_list
        or ModalityTypes.wit_main_body.value in modality_list
        or ModalityTypes.wit_image.value in modality_list
        or ModalityTypes.wit_title.value in modality_list
    ):
        video_filepath = root_filepath / pathlib.Path(
            f"video_clip_scores.parquet/relevance/{video_id}.parquet"
        )

        taliwit_table = ds.dataset(
            video_filepath, schema=video_score_schema
        ).to_table()
        table: VideoCLIPScoreSchema = taliwit_table.to_pandas()

        video_id = table.video_id[0]
        wit_index = table.wit_idx[0]

        wit_output = get_wit_sample(
            dataset=dataset,
            wit_index=wit_index,
            language_id=language_id,
            image_size=image_size,
            modality_list=modality_list,
        )

        for key, value in wit_output.items():
            if any(
                key == modality.sub_modality.replace("SubModalityTypes.", "")
                for modality in modality_list
            ):
                output_dict[key] = value

    return wit_index, output_dict


@dataclass
class WITFeature:
    item_idx: int
    image: PIL.Image.Image
    image_url: str
    caption_alt_text_description: Optional[str] = None
    caption_reference_description: Optional[str] = None
    caption_title_and_reference_description: Optional[str] = None
    context_page_description: Optional[str] = None
    context_section_description: Optional[str] = None
    hierarchical_section_title: Optional[str] = None
    is_main_image: Optional[bool] = None
    page_changed_recently: Optional[bool] = None
    page_title: Optional[str] = None
    section_title: Optional[str] = None
    text: Optional[Dict[str, str]] = None


def get_language_specific_entries(
    wit_idx: int, wit_entry: Any, language_id: List[str] = None
):
    if language_id is None:
        language_id = ["en", "uk", "fi", "da", "pl", "fr"]

    output_dict = {
        "image": wit_entry["image"],
        "image_url": wit_entry["image_url"],
        "item_idx": wit_idx,
    }
    wit_features = wit_entry["wit_features"]

    if language_id in wit_features["language"]:
        choose_language_id = np.random.choice(language_id)
        language_idx = wit_features["language"].index(choose_language_id)
    else:
        language_idx = np.random.choice(len(wit_features["language"]))
    for key, value in wit_features.items():
        output_dict[key] = value[language_idx]

    return WITFeature(**output_dict)


def get_wit_sample_idx_with_video_available(
    root_path: pathlib.Path = pathlib.Path("/data/"),
):
    wit_idx_to_tali_wit_table_path = defaultdict(list)
    wit_to_video_paths_table_path = (
        root_path / "wit_to_video_paths.parquet" / "relevance"
    )
    # /data/wit_to_video_paths.parquet/relevance/
    with tqdm.tqdm(total=558) as top_pbar:
        for top_level_dir in pathlib.Path(
            wit_to_video_paths_table_path
        ).iterdir():
            with tqdm.tqdm(total=600) as inner_pbar:
                for file in pathlib.Path(top_level_dir).rglob("*.parquet"):
                    if file.is_file():
                        table = (
                            ds.dataset(file, schema=video_score_schema)
                            .to_table()
                            .to_pydict()
                        )
                        video_clip_score_table_path = (
                            pathlib.Path(
                                "/data/video_clip_scores.parquet/relevance/"
                            )
                            / f"{table['video_id'][0]}.parquet"
                        )
                        if video_clip_score_table_path.exists():
                            wit_idx_to_tali_wit_table_path[
                                str(table.get("wit_idx")[0])
                            ].append(file.as_posix())
                    inner_pbar.update(1)
                top_pbar.update(1)
                top_pbar.set_description(
                    f"Size of dict: {len(wit_idx_to_tali_wit_table_path)}"
                )
    return wit_idx_to_tali_wit_table_path


def get_dataset_both_way_dictionaries(
    wit_idx_with_videos_dict_filepath: Union[
        str, pathlib.Path
    ] = "/data/wit_idx_with_videos_dict.json"
):
    wit_idx_with_videos_dict = load_json(
        filepath=wit_idx_with_videos_dict_filepath
    )

    video_id_to_wit_idx_dict = defaultdict(str)
    with tqdm.tqdm(total=len(wit_idx_with_videos_dict)) as pbar:
        for key, value in wit_idx_with_videos_dict.items():
            for item in value:
                video_id = pathlib.Path(item).parts[-2]
                video_id_to_wit_idx_dict[video_id] = item
                pbar.update(1)

    return {
        "wit_idx_to_tali_wit_dict": wit_idx_with_videos_dict,
        "video_id_to_wit_idx_dict": video_id_to_wit_idx_dict,
    }


class DefaultVideoTransforms:
    def __init__(self) -> None:
        self.processor: CLIPProcessor = CLIPProcessor.from_pretrained(
            "openai/clip-vit-large-patch14"
        )

    def __call__(self, x) -> Any:
        x = x.unbind(0)
        x = self.processor(images=x, return_tensors="pt")["pixel_values"]
        return x


@configurable
class TALIDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        set_name: str,
        root_filepath: Union[str, pathlib.Path],
        modality_list: List[AnyModalSample],
        language_id: str = "en",
        rng_seed: int = 42,
        top_k_tali: int = 10,
        image_size: int = 224,
        transforms: Optional[Dict[str, Callable]] = None,
        num_video_frames: int = 30,
        num_audio_frames: int = 44100,
        clip_duration_in_seconds: float = 3,
    ):
        if isinstance(root_filepath, str):
            root_filepath = pathlib.Path(root_filepath)

        self.wit_dataset = datasets.load_dataset(
            "wikimedia/wit_base",
            split="train",
            cache_dir=root_filepath / "wit_cache",
        )

        self.wit_idx_to_tali_wit_dict = load_json(
            filepath=root_filepath / "wit_idx_to_video_id_dict_cleaned.json"
        )
        self.video_id_list = load_json(
            filepath=root_filepath / "video_id_to_wit_idx_dict_cleaned.json"
        )
        self.set_name = set_name

        self.root_filepath = root_filepath
        self.modality_list = modality_list
        self.language_id = language_id
        self.rng_seed = rng_seed
        self.top_k_tali = top_k_tali
        self.image_size = image_size
        self.num_video_frames = num_video_frames
        self.num_audio_frames = num_audio_frames
        self.clip_duration_in_seconds = clip_duration_in_seconds

        if transforms is None:
            self.processor: CLIPProcessor = CLIPProcessor.from_pretrained(
                "openai/clip-vit-large-patch14"
            )
            self.transforms = {
                # "text": lambda x: self.processor(
                #     text=x, return_tensors="pt", padding=True, truncation=True
                # )["input_ids"],
                # "image": lambda x: self.processor(
                #     images=x, return_tensors="pt"
                # )["pixel_values"],
                # "video": DefaultVideoTransforms(),
            }
        else:
            self.transforms = transforms

        self.requested_youtube_data = any(
            "youtube" in modality.sub_modality.replace("SubModalityTypes.", "")
            for modality in modality_list
        )

        self.getitem_fnc = (
            get_sample_from_video_id
            if self.requested_youtube_data
            else get_sample_from_wit_index
        )

        self.video_id_list = [key for key in self.video_id_list.keys()]

        train_video_id_list = self.video_id_list[
            : int(len(self.video_id_list) * 0.9)
        ]

        val_video_id_list = self.video_id_list[
            len(train_video_id_list) : len(train_video_id_list)
            + int(len(self.video_id_list) * 0.05)
        ]

        test_video_list = self.video_id_list[
            len(train_video_id_list) + len(val_video_id_list) :
        ]

        if set_name == "train":
            self.dataset_list = train_video_id_list

            if not self.requested_youtube_data:
                self.total_items = len(self.wit_dataset)
            else:
                self.total_items = len(self.dataset_list) * self.top_k_tali
        elif set_name == "val":
            self.dataset_list = val_video_id_list
            self.total_items = len(self.dataset_list)
        elif set_name == "test":
            self.dataset_list = test_video_list
            self.total_items = len(self.dataset_list)
        else:
            raise ValueError(
                f"Set name {set_name} not supported, choose one of train, val, test"
            )

        self.broken_idx = set()
        self.started_sampling = False
        # create two get item top level methods,
        # one for wit index and one for video id,
        # these will also need different ways to sample the dataset_dict

    def __len__(self):
        if self.set_name == "train":
            return 99999999
        return self.total_items

    def __getitem__(self, idx):
        # if not self.started_sampling:
        #     self.wit_dataset = datasets.load_dataset(
        #         "wikimedia/wit_base",
        #         split="train",
        #         cache_dir=self.root_filepath / "wit_cache",
        #     )
        #     self.started_sampling = True

        if idx in self.broken_idx:
            return self.__getitem__(idx + 1)
        try:
            if self.requested_youtube_data:
                actual_idx = int(idx % self.total_items) // self.top_k_tali
                video_idx = self.dataset_list[actual_idx]
                wit_idx, output_dict = get_sample_from_video_id(
                    dataset=self.wit_dataset,
                    video_id=video_idx,
                    rng_seed=self.rng_seed + idx % self.top_k_tali,
                    root_filepath=self.root_filepath,
                    modality_list=self.modality_list,
                    language_id=self.language_id,
                    top_k_tali=self.top_k_tali,
                    image_size=self.image_size,
                    clip_duration_in_seconds=self.clip_duration_in_seconds,
                )
            else:
                idx = idx % self.total_items
                output_dict = get_wit_sample(
                    dataset=self.wit_dataset,
                    wit_index=idx,
                    language_id=self.language_id,
                    image_size=self.image_size,
                    modality_list=self.modality_list,
                )
                wit_idx = idx

            data_dict = {}
            shape_dict = {}
            for key, value in output_dict.items():
                if not isinstance(value, torch.Size):
                    modality_type = value.modality_type.modality
                    sub_modality_type = value.modality_type.sub_modality
                    data_dict[sub_modality_type] = value.data
                else:
                    shape_dict[key] = value

            rng = np.random.RandomState(self.rng_seed + idx)
            if ModalityTypes.youtube_image.value in self.modality_list:
                sample_frame_idx = rng.randint(
                    low=0, high=data_dict["youtube_content_video"].shape[0]
                )
                data_dict[
                    ModalityTypes.youtube_image.value.sub_modality
                ] = data_dict["youtube_content_video"][sample_frame_idx]
                if not ModalityTypes.youtube_video.value in self.modality_list:
                    del data_dict[
                        ModalityTypes.youtube_video.value.sub_modality
                    ]

            if (
                "youtube_content_video" in data_dict
                and "youtube_content_audio" in data_dict
            ):
                if (
                    data_dict["youtube_content_video"].shape[0]
                    < self.num_video_frames
                ):
                    data_dict["youtube_content_video"] = torch.cat(
                        [
                            data_dict["youtube_content_video"],
                            torch.zeros(
                                self.num_video_frames
                                - data_dict["youtube_content_video"].shape[0],
                                3,
                                self.image_size,
                                self.image_size,
                            ),
                        ]
                    )
                    starting_video_frame = 0
                else:
                    upper_bound = (
                        data_dict["youtube_content_video"].shape[0]
                        - self.num_video_frames
                        + 1
                    )

                    if upper_bound < 0:
                        return self.__getitem__(idx + 1)

                    starting_video_frame = rng.randint(
                        0,
                        upper_bound,
                    )

                    data_dict["youtube_content_video"] = data_dict[
                        "youtube_content_video"
                    ][
                        starting_video_frame : starting_video_frame
                        + self.num_video_frames
                    ]
                starting_audio_frame = int(
                    floor(
                        data_dict["youtube_content_audio"].shape[0]
                        * (
                            starting_video_frame
                            / shape_dict["full_video_shape"][1]
                        )
                    )
                )

                data_dict["youtube_content_audio"] = data_dict[
                    "youtube_content_audio"
                ].view(-1)

                if (
                    data_dict["youtube_content_audio"].shape[0]
                    < self.num_audio_frames
                ):
                    data_dict["youtube_content_audio"] = torch.cat(
                        [
                            data_dict["youtube_content_audio"],
                            torch.zeros(
                                self.num_audio_frames
                                - data_dict["youtube_content_audio"].shape[0]
                            ),
                        ]
                    )
                else:
                    # print(
                    #     data_dict["youtube_content_audio"].shape,
                    #     starting_audio_frame,
                    #     starting_audio_frame + self.num_audio_frames,
                    # )
                    data_dict["youtube_content_audio"] = data_dict[
                        "youtube_content_audio"
                    ][
                        starting_audio_frame : starting_audio_frame
                        + self.num_audio_frames
                    ]

            elif "youtube_content_video" in data_dict:
                if (
                    data_dict["youtube_content_video"].shape[0]
                    < self.num_video_frames
                ):
                    data_dict["youtube_content_video"] = torch.cat(
                        [
                            data_dict["youtube_content_video"],
                            torch.zeros(
                                self.num_video_frames
                                - data_dict["youtube_content_video"].shape[0],
                                3,
                                self.image_size,
                                self.image_size,
                            ),
                        ]
                    )
                else:
                    upper_bound = (
                        data_dict["youtube_content_video"].shape[0]
                        - self.num_video_frames
                        + 1
                    )

                    if upper_bound < 0:
                        return self.__getitem__(idx + 1)

                    starting_video_frame = rng.randint(
                        0,
                        data_dict["youtube_content_video"].shape[0]
                        - self.num_video_frames
                        + 1,
                    )

                    data_dict["youtube_content_video"] = data_dict[
                        "youtube_content_video"
                    ][
                        starting_video_frame : starting_video_frame
                        + self.num_video_frames
                    ]

            elif "youtube_content_audio" in data_dict:
                data_dict["youtube_content_audio"] = data_dict[
                    "youtube_content_audio"
                ].view(-1)

                if (
                    data_dict["youtube_content_audio"].shape[0]
                    < self.num_audio_frames
                ):
                    data_dict["youtube_content_audio"] = torch.cat(
                        [
                            data_dict["youtube_content_audio"],
                            torch.zeros(
                                self.num_audio_frames
                                - data_dict["youtube_content_audio"].shape[0]
                            ),
                        ]
                    )
                else:
                    upper_bound = (
                        data_dict["youtube_content_audio"].shape[0]
                        - self.num_audio_frames
                    )

                    if upper_bound < 0:
                        return self.__getitem__(idx + 1)

                    starting_audio_frame = rng.randint(
                        0,
                        upper_bound,
                    )

                    data_dict["youtube_content_audio"] = data_dict[
                        "youtube_content_audio"
                    ].view(-1)[
                        starting_audio_frame : starting_audio_frame
                        + self.num_audio_frames
                    ]
            keys = list(data_dict.keys())
            output_dict = {}
            for sub_modality_name in keys:
                modality_type = get_base_modality(sub_modality_name).value
                sub_modality_name = sub_modality_name.value
                if modality_type not in output_dict:
                    output_dict[modality_type] = {}

                if modality_type in self.transforms:
                    output_dict[modality_type][
                        sub_modality_name
                    ] = self.transforms[modality_type](
                        data_dict[sub_modality_name]
                    )
                else:
                    output_dict[modality_type][sub_modality_name] = data_dict[
                        sub_modality_name
                    ]

            output_dict["wit_idx"] = wit_idx

        except Exception as e:
            logger.exception(
                f"{e} {self.requested_youtube_data}, {self.modality_list}"
            )
            self.broken_idx.add(idx)
            return self.__getitem__(idx + 1)

        return output_dict


if __name__ == "__main__":
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
            # ModalityTypes.youtube_video.value,
            # ModalityTypes.youtube_subtitles.value,
            # ModalityTypes.youtube_audio.value,
            # ModalityTypes.youtube_description.value,
        ],
        language_id="en",
        rng_seed=42,
        top_k_tali=10,
        image_size=224,
        transforms=None,
        num_video_frames=5,
        num_audio_frames=1 * 16000,
        clip_duration_in_seconds=5.0,
    )

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=16,
        num_workers=8,
        shuffle=True,
        pin_memory=False,
        collate_fn=dataclass_collate,
        prefetch_factor=2,
    )

    def get_batch(data, batch_size):
        for i in range(0, len(data), batch_size):
            batch = []
            for i in range(i, min(i + batch_size, len(data))):
                batch.append(data[i])
            yield dataclass_collate(batch)

    with tqdm.tqdm(total=len(dataloader)) as pbar:
        for idx, item in enumerate(dataloader):
            # print(item)

            pbar.update(1)

    # print("check2")
    # with tqdm.tqdm(total=len(dataset) / 16) as pbar:
    #     for idx, item in enumerate(dataset):
    #         # print(item)
    #         if idx % 16 == 0:
    #             pbar.update(1)
