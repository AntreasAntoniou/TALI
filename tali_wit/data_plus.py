import logging
import os
import pathlib
import time
from dataclasses import dataclass
from math import floor
from typing import Any, Dict, List, Optional, Union
import datasets

import numpy as np
import torch
import tqdm
from pytorchvideo.data.encoded_video import EncodedVideo
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    ShortSideScale,
    UniformTemporalSubsample,
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
    default_image_transforms,
    ModalityTypes,
    get_base_modality,
    select_subtitles_between_timestamps,
)

from tali_wit.decorators import configurable
from tali_wit.utils import get_logger, load_json

logger = get_logger(__name__)
pytorchvideo_logger = get_logger("pytorchvideo", logging_level=logging.CRITICAL)


def get_video_clip(video, starting_second, ending_second):
    return video.get_clip(start_sec=starting_second, end_sec=ending_second)


def get_video_tensors(video_clip, image_size, num_video_frames):
    video_transform = Compose(
        [
            UniformTemporalSubsample(
                num_samples=num_video_frames,
                temporal_dim=1,
            ),
            ShortSideScale(size=image_size),
            CenterCropVideo(crop_size=(image_size, image_size)),
        ]
    )
    video_clip = ApplyTransformToKey("video", video_transform)(video_clip)
    return video_clip["video"].permute(1, 0, 2, 3) / 255.0


def get_image_tensor(video_clip, image_size, rng):
    input_dict = {
        "video": video_clip["video"][
            :, rng.randint(0, video_clip["video"].shape[1]), :, :
        ].unsqueeze(1)
    }
    video_transform = Compose(
        [
            ShortSideScale(size=image_size),
            CenterCropVideo(crop_size=(image_size, image_size)),
        ]
    )
    video_clip = ApplyTransformToKey("video", video_transform)(input_dict)
    return video_clip["video"].permute(1, 0, 2, 3) / 255.0


def videoclip_to_video_audio_tensors(
    video_path: pathlib.Path,
    rng: np.random.Generator,
    return_video: bool = True,
    return_audio: bool = False,
    return_image: bool = False,
    image_size: int = 224,
    starting_second: int = 0,
    ending_second: int = None,
    num_audio_frames: int = 1 * 16000,
    num_video_frames: int = 10,
):
    video = EncodedVideo.from_path(video_path)
    video_duration = float(video.duration)
    if ending_second is None:
        ending_second = video_duration
    if ending_second > video_duration:
        starting_second = video_duration - (ending_second - starting_second)
        starting_second = max(0, starting_second)
        ending_second = video_duration
    video_clip = get_video_clip(video, starting_second, ending_second)

    output = {}

    if return_video:
        output["video"] = get_video_tensors(
            video_clip=video_clip,
            image_size=image_size,
            num_video_frames=num_video_frames,
        )
        if output["video"].shape[0] < num_video_frames:
            output["video"] = torch.cat(
                [
                    output["video"],
                    torch.zeros(
                        num_video_frames - output["video"].shape[0],
                        output["video"].shape[1],
                        output["video"].shape[2],
                        output["video"].shape[3],
                        output["video"].shape[4],
                    ),
                ],
                dim=0,
            )
    if return_image:
        output["image"] = get_image_tensor(
            video_clip=video_clip, image_size=image_size, rng=rng
        )[0]

    if return_audio:
        output["audio"] = extract_audio(
            num_audio_frames=num_audio_frames, video_clip=video_clip
        )
    return output


def extract_audio(num_audio_frames, video_clip):
    audio_duration_target = float(num_audio_frames) / 16000.0
    source_sample_rate = 44100
    target_sample_rate = 16000
    video_clip["audio"] = video_clip["audio"][
        : int(floor(float(44100 * float(audio_duration_target))))
    ]
    resampler = T.Resample(
        source_sample_rate, target_sample_rate, dtype=video_clip["audio"].dtype
    )
    audio = resampler(video_clip["audio"])
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
                    "/data/datasets/tali-wit-2-1-buckets/", self.config.root_filepath
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
                ] = (
                    f"<ydesc> " + input_dict["youtube_description_text"] + f" </ydesc>"
                )

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
                                "/data/datasets/tali-wit-2-1-buckets/",
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
                    "/data/datasets/tali-wit-2-1-buckets/", self.config.root_filepath
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
                ] = (
                    f"<ydesc> " + input_dict["youtube_description_text"] + f" </ydesc>"
                )

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
                                "/data/datasets/tali-wit-2-1-buckets/",
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


@configurable
class TALIBase(Dataset):
    def __init__(
        self,
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
        infinite_sampling: bool = False,
    ):
        super().__init__()
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
            )
        )
        self.infinite_sampling = infinite_sampling
        self.dataset = datasets.load_from_disk(
            pathlib.Path(hf_tali_root_filepath) / f"{set_name}-set"
        )
        self.dataset = self.dataset.with_transform(transform)

        if infinite_sampling:
            self.num_samples = 10**8
        else:
            self.num_samples = len(self.dataset)

    def __getitem__(self, idx):
        if self.infinite_sampling:
            idx = idx % len(self.dataset)

        sample = self.dataset[idx]

        return sample

    def __len__(self) -> int:
        return self.num_samples


if __name__ == "__main__":
    import tqdm
    from rich import print
    from rich.traceback import install
    import cProfile
    import pstats

    install()
    os.environ["HYDRA_FULL_ERROR"] = "1"

    def sample():
        transform = TALIBaseTransform(
            config=TALIBaseTransformConfig(
                root_filepath="/data/datasets/tali-wit-2-1-buckets/",
                modality_list=[
                    ModalityTypes.wit_image.value,
                    ModalityTypes.wit_caption.value,
                    ModalityTypes.wit_title.value,
                    ModalityTypes.wit_main_body.value,
                    ModalityTypes.youtube_image.value,
                    ModalityTypes.youtube_video.value,
                    ModalityTypes.youtube_subtitles.value,
                    ModalityTypes.youtube_audio.value,
                    ModalityTypes.youtube_description.value,
                ],
                rng_seed=42,
                top_k_tali=10,
                image_size=224,
                num_video_frames=5,
                num_audio_frames=16000,
                clip_duration_in_seconds=3.0,
                deterministic_sampling=True,
            )
        )
        dataset = datasets.load_from_disk(
            "/home/evolvingfungus/forge/workspaces/tali-2-2/train-set"
        )
        dataset = dataset.with_transform(transform)
        dataset = TALIBase(
            set_name="train",
            tali_root_filepath="/data/datasets/tali-wit-2-1-buckets/",
            hf_tali_root_filepath="/home/evolvingfungus/forge/workspaces/tali-2-2/",
            modality_list=[
                ModalityTypes.wit_image.value,
                ModalityTypes.wit_caption.value,
                ModalityTypes.wit_title.value,
                ModalityTypes.wit_main_body.value,
                ModalityTypes.youtube_image.value,
                ModalityTypes.youtube_video.value,
                ModalityTypes.youtube_subtitles.value,
                ModalityTypes.youtube_audio.value,
                ModalityTypes.youtube_description.value,
            ],
            rng_seed=42,
            top_k_tali=10,
            image_size=224,
            num_video_frames=5,
            num_audio_frames=16000,
            clip_duration_in_seconds=3.0,
            deterministic_sampling=False,
            infinite_sampling=True,
        )
        # dataloader = DataLoader(
        #     dataset,
        #     batch_size=4,
        #     num_workers=1,
        #     shuffle=True,
        #     collate_fn=dataclass_collate,
        # )
        num_samples = 100
        with tqdm.tqdm(total=len(dataset)) as pbar:
            for i, example in enumerate(dataset):
                # example = generate_hierarchical_data_dict(example)
                print(example)
                shape_dict = {}
                for modality, modality_value in example.items():
                    # for submodality, submodality_value in modality_value.items():
                    if isinstance(modality_value, torch.Tensor):
                        modality_value = modality_value.to(torch.float32)
                        print(
                            f"{modality} {modality_value.mean()} {modality_value.std()}, {modality_value.min()}, {modality_value.max()}"
                        )
                    else:
                        print(f"{modality} {modality_value}")
                pbar.set_description(f"Processing {i}th example")
                pbar.update(1)

    # sample()
    pr = cProfile.Profile()
    pr.runcall(sample)

    ps = pstats.Stats(pr).sort_stats("tottime")
    ps.print_stats()
# write a transform for the wit dataset, and, add an option for a youtube image sampling process
