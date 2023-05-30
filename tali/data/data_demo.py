import pathlib
import time
from typing import Any, Dict
import numpy as np
from tali.data.data import (
    ModalityTypes,
    default_image_transforms,
    select_subtitles_between_timestamps,
)
from tali.data.data_plus import (
    TALIBaseTransformConfig,
    get_submodality_name,
    videoclip_to_video_audio_tensors,
)
from tali.utils import get_logger, load_json

logger = get_logger(__name__)


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
                return_video=get_submodality_name(
                    ModalityTypes.youtube_video.value
                )
                in self.modality_list,
                return_audio=get_submodality_name(
                    ModalityTypes.youtube_audio.value
                )
                in self.modality_list,
                return_image=get_submodality_name(
                    ModalityTypes.youtube_image.value
                )
                in self.modality_list,
                num_audio_frames=self.config.num_audio_frames,
                num_video_frames=self.config.num_video_frames,
                rng=rng,
            )
        )

        self.select_subtitles_between_timestamps = (
            select_subtitles_between_timestamps
        )

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
                        key: wit_sample[key][language_idx]
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
                    get_submodality_name(
                        ModalityTypes.youtube_description.value
                    )
                ] = (
                    "<ydesc> "
                    + input_dict["youtube_description_text"]
                    + " </ydesc>"
                )

            if (
                get_submodality_name(ModalityTypes.youtube_subtitles.value)
                in self.modality_list
            ):
                output_dict[
                    get_submodality_name(
                        ModalityTypes.youtube_description.value
                    )
                ] = (
                    "<ysub> "
                    + select_subtitles_between_timestamps(
                        subtitle_dict=load_json(
                            input_dict["youtube_subtitle_text"].replace(
                                "/data/",
                                self.config.root_filepath.as_posix()
                                if isinstance(
                                    self.config.root_filepath, pathlib.Path
                                )
                                else self.config.root_filepath,
                            )
                        ),
                        starting_timestamp=video_starting_second
                        + clip_starting_second,
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
