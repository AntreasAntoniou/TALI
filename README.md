
# TALI: Temporal and Semantic Alignment of Language, Image, Video, and Audio

Welcome to TALI, a large-scale quadra-modal dataset consisting of temporally and semantically aligned audio, language, and images. This dataset is assembled from YouTube and Wikipedia, offering rich multimodal data points for various research areas.

## Characteristics of TALI

- TALI integrates YouTube media components (video and audio), YouTube text components (title, description, subtitles), and Wikipedia components (image and context). These components have been temporally aligned and semantically integrated.
- Multiple language support enhances the global comprehension and capabilities of the TALI dataset.

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

## Getting Started

To get started with TALI, you can load the dataset via Hugging Face's `datasets` library. Here's a basic usage example:

```python
from tali.data import load_dataset_via_hub

# Path to your local dataset cache
dataset_cache = "/path/to/your/dataset/cache"

# Load the TALI dataset
dataset = load_dataset_via_hub(dataset_cache, dataset_name="Antreas/TALI")["train"]
