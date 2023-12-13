
# TALI: A Large Scale Quadra-Modal Dataset consisting of Temporally and Semantically Aligned Audio, Language and Images

Welcome to TALI, a large-scale quadra-modal dataset consisting of temporally and semantically aligned audio, language, and images. This dataset is assembled from YouTube and Wikipedia, offering rich multimodal data points for various research areas.

## Characteristics of TALI

- TALI integrates YouTube media components (video and audio), YouTube text components (title, description, subtitles), and Wikipedia components (image and context). These components have been temporally aligned and semantically integrated.
- Multiple language support enhances the global comprehension and capabilities of the TALI dataset.

## For a Gradio visualization of the full dataset please go to this [link](https://antreas.io/demos/tali)

## Getting Started

### Installation

For the default install use:

```bash
pip install git+https://github.com/AntreasAntoniou/TALI
```

For the dev install use:

```bash
pip install git+https://github.com/AntreasAntoniou/TALI[dev]
```


To get started with TALI, you can load the dataset via Hugging Face's `datasets` library through our helper functions. The reason we don't use `datasets` directly is because we found huggingface_hub downloads much faster and reliable. For a full set of possible configurations look at [examples.py](examples.py). Here's a basic usage example:

```python
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

    dataset_cache = pathlib.Path("/my/path/to/data")

    dataset = load_dataset_via_hub(dataset_cache, dataset_name="Antreas/TALI")[
        "train"
    ]

    (
        image_transforms,
        text_transforms,
        audio_transforms,
        video_transforms,
    ) = default_transforms()

    preprocessing_transform = TALIBaseTransform(
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
```

## Significance of TALI

Previously, self-supervised learning has primarily relied on uni-modal datasets or multi-modal datasets that lack temporal alignment across different modalities. TALI addresses this gap by offering a dataset with temporal alignment across four modalities. This dataset empowers models to perceive real-world dynamics and comprehend temporal sequencing, paramount in various applications. Furthermore, semantically aligned Wikipedia text and images provide added context, creating a richer learning environment for various multi-modal research areas, including contextual understanding, pattern recognition, and contrastive learning.

## Dataset Collection Methodology

TALI's assembly involved semantic alignment techniques using CLIP models to extract multilingual content from YouTube, starting from the Wikipedia Image to Text (WiT) dataset. Videos were segmented into 30-second clips and ranked for relevance. Additional metadata was retained for rich multimodal data points. More specifically:

1. We start from the [WiT dataset](https://huggingface.co/datasets/wikimedia/wit_base) use either the context_page_description or page_title, which we refer to as source-query, search youtube with it. Return top 100 result titles.
2. Compare the returning titles, which we'll call youtube-titles, with the source-query using the CLIP text embeddings of the largest CLIP model (patch-14, large) we had available whose alignments were the closest we've seen with human perception.
3. Choose top-1 title’s video based on the CLIP ranking.
4. Download video, break into 30 second segments. Apply CLIP image embedding to the first image of each segment, and compare with the video’s title text. Rank the segments based on this distance.
5. Choose the top-10 segments for each video. Extract image, audio and subtitle frames.

## Detailed Description of Components

- **Media Components**: YouTube media components (audio and video) are temporally and weakly semantically aligned with the Wikipedia image.
- **Textual Components**: YouTube subtitles are temporally and weakly semantically aligned with the Wikipedia image and text context. In contrast, the YouTube title and description text are strongly semantically aligned with their corresponding video and audio and weakly semantically aligned with the Wikipedia image and text context.

## Language and Caption Information

The TALI dataset encapsulates multi-language captions. While these captions are ideally semantically aligned with both the YouTube and Wikipedia components, alignment's extent can fluctuate due to the auto-selection process of the dataset.

## Usage

TALI can be used for a wide range of tasks. Its quadra-modal nature makes it suitable for both uni-modal and multi-modal tasks. Here are a few examples:

- **Uni-modal tasks**: Language modelling, image classification, audio classification, video classification
- **Bi-modal tasks**: Image captioning, text-to-image synthesis, audio-visual correspondence learning
- **Multi-modal tasks**: Cross-modal retrieval, multi-modal fusion for classification or prediction tasks

## Limitations and Challenges

While TALI is a highly versatile dataset, it also poses some challenges:

- The semantic alignment between YouTube and Wikipedia components is not perfect. There might be some mismatches.
- The quality of YouTube subtitles can vary greatly. Some videos might have professionally produced subtitles while others might only have auto-generated captions.
- Some videos might contain multiple languages, and the language might not always align with the language of the Wikipedia page.

## Citation

If you use TALI in your research, please cite our (not yet public) paper:

```bibtex
@article{antoniou2023tali,
  title={TALI: A Large Scale Tetra-Modal Dataset consisting of Temporally and Semantically Aligned Audio, Language and Images},
  author={Antoniou, Antreas, Eleni Triantafillou, Justin Engelmann, Fady Rezk, Hugo Larochelle, Jeff Pan, Yi Liu, and Amos Storkey},
  year={2023}
}
```