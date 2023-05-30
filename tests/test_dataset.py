import os

from transformers import CLIPProcessor
from tali.data.data import ModalityTypes
from tali.data.data_plus import TALIBase


dataset = TALIBase(
    set_name="train",
    tali_dataset_dir=os.environ["TALI_DATASET_DIR"],
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
