import os
import torch

from tqdm.auto import tqdm
from transformers import CLIPProcessor
from tali.data.data import ModalityTypes
from tali.data.data_plus import TALIBase
from tali.wit import WITBase
from rich import print
from rich.traceback import install

install()

tali_dataset = TALIBase(
    set_name="train",
    tali_dataset_dir=os.environ["TALI_DATASET_DIR"],
    modality_list=[
        ModalityTypes.wit_image.value,
        ModalityTypes.wit_caption.value,
        ModalityTypes.wit_title.value,
        ModalityTypes.wit_main_body.value,
    ],
    num_samples_per_episode=32,
    rng_seed=42,
    top_k_tali=10,
    image_size=224,
    num_video_frames=10,
    num_audio_frames=32000,
    clip_duration_in_seconds=3.0,
    deterministic_sampling=True,
    dummy_batch_mode=False,
    image_text_model_name="openai/clip-vit-base-patch16",
    audio_model_name="openai/whisper-base",
    use_model_preprocessing=True,
    total_num_samples=None,
    cache_generated_samples_in_memory=False,
    cache_num_samples=10,
)
image_text_model_name: str = "openai/clip-vit-base-patch16"

clip_preprocessor = CLIPProcessor.from_pretrained(image_text_model_name)


def test_equal_tali_dataset():
    tali_dataset = TALIBase(
        set_name="train",
        tali_dataset_dir=os.environ["TALI_DATASET_DIR"],
        modality_list=[
            ModalityTypes.wit_image.value,
            ModalityTypes.wit_caption.value,
            ModalityTypes.wit_title.value,
            ModalityTypes.wit_main_body.value,
        ],
        num_samples_per_episode=32,
        rng_seed=42,
        top_k_tali=10,
        image_size=224,
        num_video_frames=10,
        num_audio_frames=32000,
        clip_duration_in_seconds=3.0,
        deterministic_sampling=True,
        dummy_batch_mode=False,
        image_text_model_name="openai/clip-vit-base-patch16",
        audio_model_name="openai/whisper-base",
        use_model_preprocessing=True,
        total_num_samples=None,
        cache_generated_samples_in_memory=False,
        cache_num_samples=10,
    )
    image_text_model_name: str = "openai/clip-vit-base-patch16"

    clip_preprocessor = CLIPProcessor.from_pretrained(image_text_model_name)

    sample = tali_dataset[0]

    image = sample["wikipedia_caption_image"]
    text = sample["wikipedia_caption_text"]

    core_dataset = tali_dataset.dataset
    samples = core_dataset[:32]

    wit_idx = samples["wit_idx"]
    wit_image = samples["wikipedia_caption_image"]
    wit_caption = samples["wikipedia_caption_text"]

    wit_image_tokens = clip_preprocessor(
        images=wit_image, return_tensors="pt"
    ).pixel_values
    wit_text_tokens = clip_preprocessor(
        text=wit_caption, padding=True, truncation=True, return_tensors="pt"
    ).input_ids

    diff_image = torch.sum(torch.abs(image - wit_image_tokens))
    diff_text = torch.sum(torch.abs(text - wit_text_tokens))

    assert diff_image == 0
    assert diff_text == 0


def test_equal_wit_dataset():
    wit_dataset = WITBase(
        wit_dataset_dir=os.environ["WIT_DATASET_DIR"],
        tali_dataset_dir=os.environ["TALI_DATASET_DIR"],
        image_size=224,
        set_name="train",
        num_samples_per_episode=32,
        deterministic_sampling=False,
        total_num_samples=100,
        priority_caption_language="en",
        dummy_batch_mode=False,
        image_text_model_name="openai/clip-vit-base-patch32",
        audio_model_name="openai/whisper-base",
    )
    image_text_model_name: str = "openai/clip-vit-base-patch16"

    clip_preprocessor = CLIPProcessor.from_pretrained(image_text_model_name)

    sample = wit_dataset[0]

    image = sample["wikipedia_caption_image"]
    text = sample["wikipedia_caption_text"]

    core_dataset = wit_dataset.dataset
    samples = core_dataset[:32]

    wit_image = samples["image"]

    wit_image_tokens = clip_preprocessor(
        images=wit_image, return_tensors="pt"
    ).pixel_values
    print(wit_image_tokens.shape, image.shape)
    diff_image = torch.sum(torch.abs(image - wit_image_tokens))

    assert diff_image == 0
    print(text)


def test_video_equal():
    import os

    from tqdm.auto import tqdm
    from transformers import CLIPProcessor
    from tali.data.data import ModalityTypes
    from tali.data.data_plus import TALIBase
    from rich import print
    from rich.traceback import install

    install()

    tali_dataset = TALIBase(
        set_name="train",
        tali_dataset_dir=os.environ["TALI_DATASET_DIR"],
        modality_list=[
            ModalityTypes.wit_image.value,
            ModalityTypes.wit_caption.value,
            ModalityTypes.wit_title.value,
            ModalityTypes.wit_main_body.value,
            ModalityTypes.youtube_audio.value,
            ModalityTypes.youtube_video.value,
            ModalityTypes.youtube_description.value,
            ModalityTypes.youtube_title.value,
            ModalityTypes.youtube_image.value,
        ],
        num_samples_per_episode=32,
        rng_seed=42,
        top_k_tali=10,
        image_size=224,
        num_video_frames=10,
        num_audio_frames=32000,
        clip_duration_in_seconds=3.0,
        deterministic_sampling=True,
        dummy_batch_mode=False,
        image_text_model_name="openai/clip-vit-base-patch16",
        audio_model_name="openai/whisper-base",
        use_model_preprocessing=True,
        total_num_samples=None,
        cache_generated_samples_in_memory=False,
        cache_num_samples=10,
    )
    image_text_model_name: str = "openai/clip-vit-base-patch16"

    clip_preprocessor = CLIPProcessor.from_pretrained(image_text_model_name)

    sample = tali_dataset[0]

    image = sample["wikipedia_caption_image"]
    text = sample["wikipedia_caption_text"]

    print(list(sample.keys()))
    video = sample["youtube_content_video"]
    youtube_image = sample["youtube_random_video_sample_image"]
    print(
        f"image: {image.shape}, mean: {image.mean()}, std: {image.std()}, min: {image.min()}, max: {image.max()}"
    )
    print(
        f"video: {video.shape}, mean: {video.mean()}, std: {video.std()}, min: {video.min()}, max: {video.max()}"
    )
    print(
        f"youtube_image: {youtube_image.shape}, mean: {youtube_image.mean()}, std: {youtube_image.std()}, min: {youtube_image.min()}, max: {youtube_image.max()}"
    )
