# Demo of TALI: A Large Scale Tetra-Modal Dataset consisting of Temporally and Semantically Aligned Audio, Language and Images

Exploring the dynamics of real-world settings often calls for models that understand and incorporate temporal and semantic cues across multiple modalities. The TALI (Temporal and Semantic Alignment of Language, Image, Video, and Audio) dataset addresses this need. Assembled from YouTube and Wikipedia, TALI dataset offers temporally and semantically aligned data across four modalities - language, image, video, and audio.

## What should I expect to see here? 
This demo is intended to show you what the TALI dataset looks like. It is a dataset that used the [Wikipedia Image Text (WIT)](https://huggingface.co/datasets/wikimedia/wit_base) captions and article titles to search Youtube for videos that match the captions, and then subsquently downloads the video, audio, and subtitles from such videos.
The result is a rich multi modal dataset that has multiple caption types related to both the WiT Images, and the Youtube videos. This means learning can take place between either temporally or semantically aligned text, images, audio and video.
        

<details>
 <summary>Characteristics of TALI</summary>

- TALI integrates YouTube media components (video and audio), YouTube text components (title, description, subtitles), and Wikipedia components (image and context). These components have been temporally aligned and semantically integrated.
- Multiple language support enhances the global comprehension and capabilities of the TALI dataset.
</details>


<details>
  <summary>Significance of TALI</summary>

  Previously, self-supervised learning has primarily relied on uni-modal datasets or multi-modal datasets that lack temporal alignment across different modalities. TALI addresses this gap by offering a dataset with temporal alignment across four modalities. This dataset empowers models to perceive real-world dynamics and comprehend temporal sequencing, paramount in various applications. Furthermore, semantically aligned Wikipedia text and images provide added context, creating a richer learning environment for various multi-modal research areas, including contextual understanding, pattern recognition, and contrastive learning.
  
</details>

<details>
  <summary>Dataset Collection Methodology</summary>

  TALI's assembly involved semantic alignment techniques using CLIP models to extract multilingual content from YouTube, starting from the Wikipedia Image to Text (WiT) dataset. Videos were segmented into 30-second clips and ranked for relevance. Additional metadata was retained for rich multimodal data points. More specifically:
  
  1. We start from the [WiT dataset](https://huggingface.co/datasets/wikimedia/wit_base) use either the context_page_description or page_title, which we refer to as source-query, search youtube with it. Return top 100 result titles.
  2. Compare the returning titles, which we'll call youtube-titles, with the source-query using the CLIP text embeddings of the largest CLIP model (patch-14, large) we had available whose alignments were the closest we've seen with human perception.
  3. Choose top-1 title’s video based on the CLIP ranking.
  4. Download video, break into 30 second segments. Apply CLIP image embedding to the first image of each segment, and compare with the video’s title text. Rank the segments based on this distance.
  5. Choose the top-10 segments for each video. Extract image, audio and subtitle frames.

</details>

<details>
  <summary>Detailed Description of Components</summary>

  - **Media Components**: YouTube media components (audio and video) are temporally and weakly semantically aligned with the Wikipedia image.
  - **Textual Components**: YouTube subtitles are temporally and weakly semantically aligned with the Wikipedia image and text context. In contrast, the YouTube title and description text are strongly semantically aligned with their corresponding video and audio and weakly semantically aligned with the Wikipedia image and text context.
</details>


<details>
  <summary>Language and Caption Information</summary>

  The TALI dataset encapsulates multi-language captions. While these captions are ideally semantically aligned with both the YouTube and Wikipedia components, alignment's extent can fluctuate due to the auto-selection process of the dataset.

</details>

## Your Guide to Navigating the Demo
Embark on an exploratory journey through TALI with these easy steps:
1. **Set Selection:** Commence your exploration in the 'train', 'val', or 'test' sets.
2. **Sampling Challenges:** Opt for a selected index or embrace randomness with a surprise fetch.
3. **Savor the Media Fusion:** Experience how Wikipedia context and imagery aligns with YouTube's rich video and audio media.
4. **Unraveling Layered Text:** Dive deep into contextual narratives, explore language nuances, and synthesize multi-dimensional relationships between text and media.

<details>

<summary>Participate in TALI's Refinement</summary>
Your contributions can profoundly shape TALI:
1. **Detail Your Observations:** Should you identify improvements or note aspects that demand attention, document your observations.
2. **Submit Reports for Dataset Enrichment:** Your detailed feedback spurs enhancements and sculpt the dataset's ongoing growth.
</details>

