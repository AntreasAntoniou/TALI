import pathlib
import time
from typing import Tuple

import torch
from torch.utils.data import DataLoader
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
    dataset_cache_path: pathlib.Path | str,
):
    if isinstance(dataset_storage_path, str):
        dataset_storage_path = pathlib.Path(dataset_storage_path)

    dataset = load_dataset_via_hub(
        dataset_download_path=dataset_storage_path,
        dataset_cache_path=dataset_cache_path / "tali",
        dataset_name="Antreas/TALI",
    )["train"]

    (
        image_transforms,
        text_transforms,
        audio_transforms,
        video_transforms,
    ) = default_transforms()

    preprocessing_transform = TALIBaseTransform(
        cache_dir=dataset_cache_path,
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
            if hasattr(value, "shape") or isinstance(value, torch.Tensor):
                print(key, value.shape)
            elif hasattr(value, "__len__"):
                print(key, len(value))
            print(key, type(value))

        break


def tali_without_transforms_no_streaming(
    dataset_storage_path: pathlib.Path | str,
    dataset_cache_path: pathlib.Path | str,
):
    if isinstance(dataset_storage_path, str):
        dataset_storage_path = pathlib.Path(dataset_storage_path)

    dataset = load_dataset_via_hub(
        dataset_download_path=dataset_storage_path,
        dataset_cache_path=dataset_cache_path / "tali",
        dataset_name="Antreas/TALI",
    )["train"]

    preprocessing_transform = TALIBaseTransform(
        cache_dir=dataset_cache_path,
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
            if hasattr(value, "shape") or isinstance(value, torch.Tensor):
                print(key, value.shape)
            elif hasattr(value, "__len__"):
                print(key, len(value))
            print(key, type(value))

        break


def tali_with_transforms_streaming(
    dataset_storage_path: pathlib.Path | str,
    dataset_cache_path: pathlib.Path | str,
):
    if isinstance(dataset_storage_path, str):
        dataset_storage_path = pathlib.Path(dataset_storage_path)

    dataset = load_dataset_via_hub(
        dataset_download_path=dataset_storage_path,
        dataset_cache_path=dataset_cache_path / "tali",
        dataset_name="Antreas/TALI",
        streaming=True,
    )["train"]

    (
        image_transforms,
        text_transforms,
        audio_transforms,
        video_transforms,
    ) = default_transforms()

    preprocessing_transform = TALIBaseTransform(
        cache_dir=dataset_cache_path,
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
            if hasattr(value, "shape") or isinstance(value, torch.Tensor):
                print(key, value.shape)
            elif hasattr(value, "__len__"):
                print(key, len(value))
            print(key, type(value))


def tali_without_transforms_streaming(
    dataset_storage_path: pathlib.Path | str,
    dataset_cache_path: pathlib.Path | str,
):
    if isinstance(dataset_storage_path, str):
        dataset_storage_path = pathlib.Path(dataset_storage_path)

    dataset = load_dataset_via_hub(
        dataset_download_path=dataset_storage_path,
        dataset_cache_path=dataset_cache_path / "tali",
        dataset_name="Antreas/TALI",
        streaming=True,
    )["train"]

    preprocessing_transform = TALIBaseTransform(
        cache_dir=dataset_cache_path,
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
            if hasattr(value, "shape") or isinstance(value, torch.Tensor):
                print(key, value.shape)
            elif hasattr(value, "__len__"):
                print(key, len(value))
            print(key, type(value))

        break


def measure_dataloader_speed(
    dataloader: DataLoader,
    num_batches: int = 100,
) -> Tuple[float, float]:
    """Measure the speed of a dataloader.

    Args:
        dataloader: DataLoader to measure
        num_batches: Number of batches to process

    Returns:
        Tuple of (average time per batch, samples per second)
    """
    total_time = 0
    total_samples = 0

    for i, batch in enumerate(tqdm(dataloader, total=num_batches)):
        if i >= num_batches:
            break

        start_time = time.time()
        # Simulate processing by accessing the batch
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                _ = value.shape
        end_time = time.time()

        batch_time = end_time - start_time
        total_time += batch_time
        total_samples += dataloader.batch_size

    avg_time = total_time / num_batches
    samples_per_sec = total_samples / total_time

    return avg_time, samples_per_sec


def tali_dataloader_speed_test(
    dataset_storage_path: pathlib.Path | str,
    dataset_cache_path: pathlib.Path | str,
    num_workers: int = 0,
    num_batches: int = 100,
):
    """Test TALI dataset loading speed with DataLoader.

    Args:
        dataset_storage_path: Path to dataset storage
        dataset_cache_path: Path to dataset cache
        num_workers: Number of workers for DataLoader
        num_batches: Number of batches to process
    """
    if isinstance(dataset_storage_path, str):
        dataset_storage_path = pathlib.Path(dataset_storage_path)

    # Load dataset with transforms
    dataset = load_dataset_via_hub(
        dataset_download_path=dataset_storage_path,
        dataset_cache_path=dataset_cache_path / "tali",
        dataset_name="Antreas/TALI",
    )["train"]

    (
        image_transforms,
        text_transforms,
        audio_transforms,
        video_transforms,
    ) = default_transforms()

    preprocessing_transform = TALIBaseTransform(
        cache_dir=dataset_cache_path,
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

    # Create DataLoader with batch size 1
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=lambda x: preprocessing_transform(
            x[0]
        ),  # Apply transform in collate
    )

    print(f"\nTesting DataLoader with {num_workers} workers:")
    avg_time, samples_per_sec = measure_dataloader_speed(
        dataloader, num_batches
    )
    print(f"Average time per batch: {avg_time:.4f} seconds")
    print(f"Samples per second: {samples_per_sec:.2f}")


class ExampleOption:
    WITH_TRANSFORMS_NO_STREAMING = "with_transforms_no_streaming"
    WITHOUT_TRANSFORMS_NO_STREAMING = "without_transforms_no_streaming"
    WITH_TRANSFORMS_STREAMING = "with_transforms_streaming"
    WITHOUT_TRANSFORMS_STREAMING = "without_transforms_streaming"
    DATALOADER_SPEED_TEST = "dataloader_speed_test"


def main(
    option: ExampleOption,
    dataset_storage_path: pathlib.Path | str,
    dataset_cache_path: pathlib.Path | str,
    num_workers: int = 0,
    num_batches: int = 100,
):
    if isinstance(dataset_storage_path, str):
        dataset_storage_path = pathlib.Path(dataset_storage_path)

    if isinstance(dataset_cache_path, str):
        dataset_cache_path = pathlib.Path(dataset_cache_path)

    if option == ExampleOption.WITH_TRANSFORMS_NO_STREAMING:
        tali_with_transforms_no_streaming(
            dataset_storage_path=dataset_storage_path,
            dataset_cache_path=dataset_cache_path,
        )
    elif option == ExampleOption.WITHOUT_TRANSFORMS_NO_STREAMING:
        tali_without_transforms_no_streaming(
            dataset_storage_path=dataset_storage_path,
            dataset_cache_path=dataset_cache_path,
        )
    elif option == ExampleOption.WITH_TRANSFORMS_STREAMING:
        tali_with_transforms_streaming(
            dataset_storage_path=dataset_storage_path,
            dataset_cache_path=dataset_cache_path,
        )
    elif option == ExampleOption.WITHOUT_TRANSFORMS_STREAMING:
        tali_without_transforms_streaming(
            dataset_storage_path=dataset_storage_path,
            dataset_cache_path=dataset_cache_path,
        )
    elif option == ExampleOption.DATALOADER_SPEED_TEST:
        tali_dataloader_speed_test(
            dataset_storage_path=dataset_storage_path,
            dataset_cache_path=dataset_cache_path,
            num_workers=num_workers,
            num_batches=num_batches,
        )


if __name__ == "__main__":
    import fire

    fire.Fire(main)
