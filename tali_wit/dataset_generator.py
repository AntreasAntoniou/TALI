import pathlib
from collections import defaultdict
from math import floor
from typing import Any, List, Union
import datasets

import numpy as np
import pyarrow as pa
import pyarrow.dataset as ds
import tqdm
from pytorchvideo.data.encoded_video import EncodedVideo
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    ShortSideScale,
    UniformTemporalSubsample,
)
from torchvision.transforms import Compose
from torchvision.transforms._transforms_video import CenterCropVideo
import datasets
from tali_wit.data import (
    AnyModalSample,
    ModalityDataSample,
    ModalityTypes,
    TALISchema,
    VideoCLIPScoreSchema,
)

from tali_wit.utils import get_logger, load_json

logger = get_logger(__name__)

tali_schema = list(TALISchema.__dict__["__annotations__"].items())
tali_schema = pa.schema(tali_schema)

video_score_schema = list(
    VideoCLIPScoreSchema.__dict__["__annotations__"].items()
)
video_score_schema = pa.schema(video_score_schema)


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


def get_wit_sample(
    dataset: Any,
    wit_index: int,
):
    wit_index = int(wit_index)
    data_dict = get_language_specific_entries(
        wit_idx=wit_index, wit_entry=dataset[wit_index]
    )

    return data_dict


def get_tali_sample(
    video_id: int,
    root_filepath: pathlib.Path = pathlib.Path("/data/"),
):
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
    wit_index = table.wit_idx[table_idx]
    term_idx = table.term_idx[table_idx]

    #######################################################################################################################
    ## Get video clip that includes image frames and audio frames

    video_data_root = root_filepath / "video_data.parquet"

    video_data_filepath = pathlib.Path(video_data_root / video_path).parent
    subclip_filepaths = list(video_data_filepath.rglob("*.mp4"))[:10]
    subclip_filepaths = [
        str(subclip_filepath.as_posix())
        for subclip_filepath in subclip_filepaths
    ]

    output_dict = {
        str(ModalityTypes.youtube_video.value.sub_modality): subclip_filepaths
    }
    #######################################################################################################################
    ## Get youtube subtitles
    clip_subtitles_filepath = (
        root_filepath
        / pathlib.Path("captions.parquet/relevance/")
        / str(int(wit_index / 1000))
        / str(wit_index)
        / str(term_idx)
        / str(video_id)
        / "captions.json"
    )

    # subtitles = load_json(clip_subtitles_filepath)

    output_dict[
        str(ModalityTypes.youtube_subtitles.value.sub_modality)
    ] = clip_subtitles_filepath.as_posix()

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

    output_dict[
        str(ModalityTypes.youtube_title.value.sub_modality)
    ] = wit_to_tali_entry_table.title[video_idx]

    output_dict[
        str(ModalityTypes.youtube_description.value.sub_modality)
    ] = wit_to_tali_entry_table.description[video_idx]

    for key in list(output_dict.keys()):
        if output_dict[key] == None:
            output_dict[key] = ""

    return wit_index, output_dict


def get_sample_from_video_id(
    dataset: Any,
    video_id: str,
    modality_list: List[ModalityTypes],
    root_filepath: pathlib.Path = pathlib.Path("/data/"),
):
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
    ):
        wit_index, output_dict = get_tali_sample(
            video_id=video_id,
            root_filepath=root_filepath,
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
        )

        for key, value in wit_output.items():
            output_dict[key] = value

    return wit_index, output_dict


def get_language_specific_entries(wit_idx: int, wit_entry: Any):
    output_dict = {
        "image": wit_entry["image"],
        "image_url": wit_entry["image_url"],
        "item_idx": wit_idx,
        "wit_features": wit_entry["wit_features"].copy(),
    }

    return output_dict


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


class TALIDatasetGenerator:
    def __init__(
        self,
        set_name: str,
        root_filepath: Union[str, pathlib.Path],
        modality_list: List[AnyModalSample],
    ):
        super().__init__()
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

        self.requested_youtube_data = any(
            "youtube" in modality.sub_modality.replace("SubModalityTypes.", "")
            for modality in modality_list
        )

        self.video_id_list = list(self.video_id_list.keys())

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
            self.total_items = len(self.dataset_list)
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

    def __len__(self):
        return self.total_items

    def __getitem__(self, idx):
        try:
            actual_idx = int(idx)
            video_idx = self.dataset_list[actual_idx]
            wit_idx, output_dict = get_sample_from_video_id(
                dataset=self.wit_dataset,
                video_id=video_idx,
                root_filepath=self.root_filepath,
                modality_list=self.modality_list,
            )

            output_dict["wit_idx"] = wit_idx
            keys = list(output_dict.keys())
            for key in keys:
                if "SubModalityTypes." in key:
                    new_key = key.replace("SubModalityTypes.", "")
                    output_dict[new_key] = output_dict[key]
                    del output_dict[key]
            return output_dict

        except Exception as e:
            # logger.exception(
            #     f"{e} {self.requested_youtube_data}, {self.modality_list}"
            # )
            return False


# import tqdm
# from rich import print
# from rich.traceback import install

# install()
# os.environ["HYDRA_FULL_ERROR"] = "1"


def tali_generator(set_name):
    failed_samples = 0
    dataset = TALIDatasetGenerator(
        set_name=set_name,
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
    )
    sample_idx = 0
    with tqdm.tqdm(total=len(dataset), smoothing=0.0) as pbar:
        for idx, item in enumerate(dataset):
            pbar.update(1)

            if idx >= len(dataset):
                break

            if item is False:
                failed_samples += 1
                continue

            pbar.set_description(
                f"idx: {idx} Failed samples: {failed_samples}, total length: {len(dataset)}"
            )
            sample_idx += 1

            yield item
