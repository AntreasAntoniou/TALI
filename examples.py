import pathlib
from enum import Enum

import torch
from tqdm.auto import tqdm

from tali.data import (
    SubModalityTypes,
    TALIBaseTransform,
    TALIBaseTransformConfig,
    VideoFramesFormat,
    default_transforms,
    load_dataset_via_hub,
)


def tali_with_transforms_no_streaming(
    dataset_storage_path: pathlib.Path | str,
):
    if isinstance(dataset_storage_path, str):
        dataset_storage_path = pathlib.Path(dataset_storage_path)

    dataset = load_dataset_via_hub(
        dataset_storage_path, dataset_name="Antreas/TALI"
    )["train"]

    (
        image_transforms,
        text_transforms,
        audio_transforms,
        video_transforms,
    ) = default_transforms()

    preprocessing_transform = TALIBaseTransform(
        cache_dir=dataset_storage_path / "cache",
        text_tokenizer=text_transforms,
        image_tokenizer=image_transforms,
        audio_tokenizer=audio_transforms,
        video_tokenizer=video_transforms,
        config=TALIBaseTransformConfig(
            root_filepath=dataset_storage_path,
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
        sample = preprocessing_transform(sample)
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


def tali_without_transforms_no_streaming(
    dataset_storage_path: pathlib.Path | str,
):
    if isinstance(dataset_storage_path, str):
        dataset_storage_path = pathlib.Path(dataset_storage_path)

    dataset = load_dataset_via_hub(
        dataset_storage_path, dataset_name="Antreas/TALI"
    )["train"]

    preprocessing_transform = TALIBaseTransform(
        cache_dir=dataset_storage_path / "cache",
        text_tokenizer=None,
        image_tokenizer=None,
        audio_tokenizer=None,
        video_tokenizer=None,
        config=TALIBaseTransformConfig(
            root_filepath=dataset_storage_path,
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
        sample = preprocessing_transform(sample)
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


def tali_with_transforms_streaming(
    dataset_storage_path: pathlib.Path | str,
):
    if isinstance(dataset_storage_path, str):
        dataset_storage_path = pathlib.Path(dataset_storage_path)

    dataset = load_dataset_via_hub(
        dataset_storage_path, dataset_name="Antreas/TALI", streaming=True
    )["train"]

    (
        image_transforms,
        text_transforms,
        audio_transforms,
        video_transforms,
    ) = default_transforms()

    preprocessing_transform = TALIBaseTransform(
        cache_dir=dataset_storage_path / "cache",
        text_tokenizer=text_transforms,
        image_tokenizer=image_transforms,
        audio_tokenizer=audio_transforms,
        video_tokenizer=video_transforms,
        config=TALIBaseTransformConfig(
            root_filepath=dataset_storage_path,
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
        sample = preprocessing_transform(sample)
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


def tali_without_transforms_streaming(
    dataset_storage_path: pathlib.Path | str,
):
    if isinstance(dataset_storage_path, str):
        dataset_storage_path = pathlib.Path(dataset_storage_path)

    dataset = load_dataset_via_hub(
        dataset_storage_path, dataset_name="Antreas/TALI", streaming=True
    )["train"]

    preprocessing_transform = TALIBaseTransform(
        cache_dir=dataset_storage_path / "cache",
        text_tokenizer=None,
        image_tokenizer=None,
        audio_tokenizer=None,
        video_tokenizer=None,
        config=TALIBaseTransformConfig(
            root_filepath=dataset_storage_path,
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
        sample = preprocessing_transform(sample)
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


class ExampleOption(Enum):
    WITH_TRANSFORMS_NO_STREAMING = "with_transforms_no_streaming"
    WITHOUT_TRANSFORMS_NO_STREAMING = "without_transforms_no_streaming"
    WITH_TRANSFORMS_STREAMING = "with_transforms_streaming"
    WITHOUT_TRANSFORMS_STREAMING = "without_transforms_streaming"


def main(option: ExampleOption, dataset_storage_path: pathlib.Path | str):
    if option == ExampleOption.WITH_TRANSFORMS_NO_STREAMING:
        tali_with_transforms_no_streaming(
            dataset_storage_path=dataset_storage_path
        )
    elif option == ExampleOption.WITHOUT_TRANSFORMS_NO_STREAMING:
        tali_without_transforms_no_streaming(
            dataset_storage_path=dataset_storage_path
        )
    elif option == ExampleOption.WITH_TRANSFORMS_STREAMING:
        tali_with_transforms_streaming(
            dataset_storage_path=dataset_storage_path
        )
    elif option == ExampleOption.WITHOUT_TRANSFORMS_STREAMING:
        tali_without_transforms_streaming(
            dataset_storage_path=dataset_storage_path
        )


if __name__ == "__main__":
    import fire

    fire.Fire(main)
