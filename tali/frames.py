import contextlib
import functools
import io
import os
import time
from typing import Union

import av
import numpy as np
import torch


class FrameSelectionMethod:
    """
    Enum-like class for frame selection methods 🎞
    """

    RANDOM: str = "random"  # 🎲
    UNIFORM: str = "uniform"  # 📏
    SEQUENTIAL: str = "sequential"  #


def seek_to_second(container, stream, second):
    # Convert the second to the stream's time base
    timestamp = int(second * stream.time_base.denominator / stream.time_base.numerator)
    # Seek to the timestamp
    container.seek(timestamp, stream=stream)
    return container


def duration_in_seconds(stream):
    return float(stream.duration * stream.time_base)


def frame_timestamp_in_seconds(frame, stream):
    return float(frame.pts * stream.time_base)


def duration_in_seconds_from_path(video_path, modality):
    with av.open(video_path) as container:
        stream = next(s for s in container.streams if s.type == modality)
        return duration_in_seconds(stream)


def suppress_stderr(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with open(os.devnull, "w") as devnull:
            with contextlib.redirect_stderr(devnull):
                return func(*args, **kwargs)

    return wrapper


@suppress_stderr
def extract_frames_pyav(
    video_data: Union[str, bytes],
    modality: str,
    starting_second: float,
    ending_second: float,
    num_frames: int,
    rng: np.random.Generator,
    frame_selection_method: str = "RANDOM",
    key_frames_only: bool = False,
    stereo_audio_if_available: bool = False,
    single_image_frame: bool = False,
) -> torch.Tensor:
    frame_dict = {}

    video_source = io.BytesIO(video_data) if isinstance(video_data, bytes) else video_data

    with av.open(video_source) as container:
        stream = next(s for s in container.streams if s.type == modality)
        if key_frames_only:
            stream.codec_context.skip_frame = "NONKEY"

        container = seek_to_second(container, stream, starting_second)

        for frame in container.decode(stream):
            # logger.info(f"Frame timestamp: {frame}")
            frame_timestamp = frame_timestamp_in_seconds(frame, stream)
            # logger.info(f"Frame timestamp: {frame_timestamp}")
            array_frame = torch.from_numpy(
                frame.to_ndarray(format="rgb24" if modality == "video" else None)
            )

            if modality == "video" and len(array_frame.shape) == 2:
                array_frame = array_frame.unsqueeze(0)

            if modality == "audio" and not stereo_audio_if_available:
                array_frame = array_frame[0].unsqueeze(0)

            if frame_timestamp > ending_second:
                break
            frame_dict[frame_timestamp] = array_frame
            # logger.info(f"Frame dict: {frame_dict}")
            if single_image_frame:
                break

    frame_values = (
        torch.stack(list(frame_dict.values()))
        if modality == "video"
        else torch.cat(list(frame_dict.values()), dim=1).permute(1, 0)
    )

    if frame_selection_method == FrameSelectionMethod.RANDOM:
        frame_indices = rng.choice(
            len(frame_values),
            min(num_frames, len(frame_values)),
            replace=key_frames_only,
        )
    elif frame_selection_method == FrameSelectionMethod.UNIFORM:
        frame_indices = np.linspace(
            0,
            len(frame_values),
            min(num_frames, len(frame_values)),
            endpoint=False,
            dtype=int,
        )
    elif frame_selection_method == FrameSelectionMethod.SEQUENTIAL:
        frame_indices = np.arange(0, min(num_frames, len(frame_values)))

    frame_indices = sorted(set(frame_indices))
    output = frame_values[frame_indices]

    if modality == "video" and len(output.shape) == 3:
        output = output.unsqueeze(0)

    return output


def test_extract_frames_video_pyav():
    video_path = (
        "/data/datasets/tali-wit-2-1-buckets/video_data.parquet/550/550321/4chLRYT8ylY/360p_90.mp4"
    )
    video_path = (
        "/data/datasets/tali-wit-2-1-buckets//video_data.parquet/10/10586/SA7bKo4HRTg/360p_0.mp4"
    )
    modality = "video"
    start_time = 10
    end_time = 20
    num_frames = 30
    rng = np.random.default_rng()

    for selection_method in [
        FrameSelectionMethod.RANDOM,
        FrameSelectionMethod.UNIFORM,
        FrameSelectionMethod.SEQUENTIAL,
    ]:
        for i in range(5):
            time_list = []
            for key_frames_only in [False]:
                start_fn_time = time.time()
                frames = extract_frames_pyav(
                    video_path=video_path,
                    modality=modality,
                    starting_second=start_time,
                    ending_second=end_time,
                    num_frames=num_frames,
                    rng=rng,
                    frame_selection_method=selection_method,
                    key_frames_only=key_frames_only,
                )
                end_fn_time = time.time()
                time_list.append(end_fn_time - start_fn_time)
        print(
            f"Using {selection_method} frame selection method 🎲, with key_frames_only: {key_frames_only}, have extracted {frames.shape}, mean time {np.mean(time_list)} seconds, std time {np.std(time_list)} seconds"
        )


def test_extract_frames_audio_pyav():
    video_path = (
        "/data/datasets/tali-wit-2-1-buckets/video_data.parquet/550/550321/4chLRYT8ylY/360p_90.mp4"
    )
    video_path = (
        "/data/datasets/tali-wit-2-1-buckets//video_data.parquet/10/10586/SA7bKo4HRTg/360p_0.mp4"
    )
    modality = "audio"
    start_time = 10
    end_time = 20
    num_frames = 88200
    rng = np.random.default_rng()

    for selection_method in [
        FrameSelectionMethod.RANDOM,
        FrameSelectionMethod.UNIFORM,
        FrameSelectionMethod.SEQUENTIAL,
    ]:
        for i in range(5):
            time_list = []
            for key_frames_only in [False]:
                start_fn_time = time.time()
                frames = extract_frames_pyav(
                    video_path=video_path,
                    modality=modality,
                    starting_second=start_time,
                    ending_second=end_time,
                    num_frames=num_frames,
                    rng=rng,
                    frame_selection_method=selection_method,
                    key_frames_only=key_frames_only,
                    stereo_audio_if_available=False,
                )
                end_fn_time = time.time()
                time_list.append(end_fn_time - start_fn_time)
        print(
            f"Using {selection_method} frame selection method 🎲, with key_frames_only: {key_frames_only}, have extracted {frames.shape}, mean time {np.mean(time_list)} seconds, std time {np.std(time_list)} seconds"
        )


if __name__ == "__main__":
    # test_extract_frames_torchvision()
    # test_extract_frames_video_pyav()
    test_extract_frames_audio_pyav()
