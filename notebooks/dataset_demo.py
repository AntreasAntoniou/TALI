import pathlib
import random

import datasets
import gradio as gr
import torch
import torchvision.transforms as T
from fastapi import concurrency
from rich import print
from rich.traceback import install

from tali.data.data_new import load_dataset_via_hub
from tali.utils import get_logger

logger = get_logger(__name__)

from tali.data.data_new import TALIBaseDemoTransform

dataset_cache = pathlib.Path("/disk/scratch_fast0/tali/")
dataset_dict = load_dataset_via_hub(dataset_cache)
demo_transform = TALIBaseDemoTransform(cache_dir=dataset_cache / "cache")
dataset_length_dict = {
    "train": len(dataset_dict["train"]),
    "val": len(dataset_dict["val"]),
    "test": len(dataset_dict["test"]),
}


dataset_dict.set_transform(demo_transform)


num_samples = 0

# [
#     'image',
#     'image_url',
#     'item_idx',
#     'wit_features',
#     'wit_idx',
#     'youtube_title_text',
#     'youtube_description_text',
#     'youtube_video_content',
#     'youtube_video_starting_time',
#     'youtube_subtitle_text',
#     'youtube_video_size',
#     'youtube_video_file_path',
#     'captions'
# ]


def update_length_options(set_name):
    max_idx = dataset_length_dict[set_name] - 1
    return gr.update(minimum=0, maximum=max_idx, step=1)


def get_random_sample(set_name):
    sample_index = random.randint(0, dataset_length_dict[set_name] - 1)
    return sample_index


def generate_caption_output(caption_dict):
    with gr.Column() as output:
        for language_key, language_captions in caption_dict.items():
            with gr.Tab(language_key):
                for caption_key, caption in language_captions.items():
                    gr.Textbox(value=caption, label=caption_key)

    return gr.update(children=[output])


def update_captions(language, set_name, sample_index):
    dataset = dataset_dict[set_name]
    sample = dataset[int(sample_index)]
    caption_dict = sample["captions"][language]

    for key in [
        "caption_alt_text_description",
        "caption_reference_description",
        "caption_title_and_reference_description",
        "context_page_description",
        "context_section_description",
        "hierarchical_section_title",
        "page_title",
        "section_title",
    ]:
        if key not in caption_dict:
            caption_dict[key] = "<Unavailable/>"

    return [
        gr.update(value=caption_dict["caption_alt_text_description"]),
        gr.update(value=caption_dict["caption_reference_description"]),
        gr.update(
            value=caption_dict["caption_title_and_reference_description"]
        ),
        gr.update(value=caption_dict["context_page_description"]),
        gr.update(value=caption_dict["context_section_description"]),
        gr.update(value=caption_dict["hierarchical_section_title"]),
        gr.update(value=caption_dict["page_title"]),
        gr.update(value=caption_dict["section_title"]),
    ]


def update_language_choices(set_name, sample_index):
    languages = list(
        dataset_dict[set_name][int(sample_index)]["captions"].keys()
    )
    return gr.update(choices=languages, value=languages[0]), *update_captions(
        languages[0], set_name, sample_index
    )


def load_sample(set_name, sample_index):
    # Load the dataset based on the set name (you'll need to implement this part)
    dataset = dataset_dict[set_name]

    # Retrieve the sample at the given index
    sample = dataset[int(sample_index)]
    if sample_index == -1:
        return load_random_sample(set_name)
    # Extract the text, image, video, and audio from the sample (you'll need to adapt this to your specific dataset)
    subtitles = sample["youtube_subtitle_text"]

    wit_image = sample["image"]

    temp_file_path = "/dev/shm/{set_name}_{sample_index}.mp4"

    # Write the bytes to the file
    with open(temp_file_path, "wb") as temp_file:
        temp_file.write(sample["youtube_video_content"])

    video_path = temp_file_path
    audio_path = temp_file_path

    youtube_description_text = sample["youtube_description_text"]
    youtube_title_text = sample["youtube_title_text"]

    return (
        *update_language_choices(set_name=set_name, sample_index=sample_index),
        subtitles,
        wit_image,
        video_path,
        audio_path,
        youtube_description_text,
        youtube_title_text,
    )


def load_random_sample(set_name):
    sample_idx = get_random_sample(set_name)
    return gr.update(value=sample_idx), *load_sample(set_name, sample_idx)


if __name__ == "__main__":
    callback = gr.CSVLogger()

    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
        # TALI (Temporally and semantically Aligned Audio, Language and Images) Dataset Demo v-0.4.1 🖼️ 🔊 🎦 📝
        ## What should I expect to see here? 
        This demo is intended to show you what the TALI dataset looks like. It is a dataset that used the [Wikipedia Image Text (WIT)](https://huggingface.co/datasets/wikimedia/wit_base) captions and article titles to search Youtube for videos that match the captions, and then subsquently downloads the video, audio, and subtitles from such videos.
        The result is a rich multi modal dataset that has multiple caption types related to both the WiT Images, and the Youtube videos. This means learning can take place between either temporally or semantically aligned text, images, audio and video.
        ## How was this collected?
        1. We start from the [WiT dataset](https://huggingface.co/datasets/wikimedia/wit_base) use either the context_page_description or page_title, which we refer to as source-query, search youtube with it. Return top 100 result titles.
        2. Compare the returning titles, which we'll call youtube-titles, with the source-query using the CLIP text embeddings of the largest CLIP model (patch-14, large) we had available whose alignments were the closest we've seen with human perception.
        3. Choose top-1 title’s video based on the CLIP ranking.
        4. Download video, break into 30 second segments. Apply CLIP image embedding to the first image of each segment, and compare with the video’s title text. Rank the segments based on this distance.
        5. Choose the top-10 segments for each video. Extract image, audio and subtitle frames.
        At sampling time:
        Randomly select one of these 10 segments, choose a 10 second segment out of the 30 second clip. Return 200 video frames (spread throughout the 10 second segment), and, 160000 audio frames (10 seconds).
        """
        )

        gr.Markdown(
            """
        ### First select the set to sample from, and the index of the sample to load, and click the "Fetch sample" button OR simply click the "Fetch random sample" button to load a random sample.
        """
        )
        with gr.Row():
            with gr.Column():
                input_set_name = gr.Dropdown(
                    choices=["train", "val", "test"],
                    value="train",
                    label="Set name",
                    info="Select the set to sample from",
                )
            with gr.Column():
                input_sample_index = gr.Slider(
                    minimum=0,
                    maximum=130000,
                    randomize=True,
                    step=1,
                    interactive=True,
                    label="Datapoint idx to sample",
                    info="Select the idx to sample",
                )

            with gr.Column():
                fetch_btn = gr.Button("Fetch sample")
                fetch_random_btn = gr.Button("Fetch random sample")

        input_set_name.change(
            update_length_options, input_set_name, input_sample_index
        )
        gr.Markdown(
            """
        ### The wikipedia image and caption should be (weakly) semantically aligned to the youtube components, while the youtube components are temporally aligned to each other.
        """
        )
        youtube_subtitle_text = gr.Text(label="Youtube Subtitles")
        youtube_title_text = gr.Textbox(label="Youtube Title")
        youtube_description_text = gr.Textbox(label="Youtube Description")

        with gr.Row():
            with gr.Column():
                wikipedia_image = gr.Image(label="Wikipedia Image")
            with gr.Column():
                youtube_video = gr.Video(label="Youtube Video")
            with gr.Column():
                youtube_audio = gr.Audio(label="Youtube Audio")

        gr.Markdown(
            """
        ### Choose what language to display captions in (the captions are in multiple languages)
        """
        )
        wikipedia_language = gr.Dropdown(label="Wiki language ID")

        gr.Markdown(
            """
        ### These captions are semantically aligned to the wikipedia image, and should ideally be semantically aligned to the youtube components, however the dataset was selected automatically and this is not always the case. Overall however, the captions are very good at describing the youtube components.
        """
        )
        with gr.Row():
            page_title = gr.Textbox(label="Wikipedia Page Title")
        with gr.Row():
            caption_title_and_reference_description = gr.Textbox(
                label="Wikipedia Caption Title and Reference Description"
            )
        with gr.Row():
            section_title = gr.Textbox(label="Wikipedia Section Title")
        with gr.Row():
            hierarchical_section_title = gr.Textbox(
                label="Wikipedia Hierarchical Section Title"
            )
        with gr.Row():
            caption_alt_text_description = gr.Textbox(
                label="Wikipedia Caption Alt Text Description"
            )
        with gr.Row():
            caption_reference_description = gr.Textbox(
                label="Wikipedia Caption Reference Description"
            )
        with gr.Row():
            context_section_description = gr.Textbox(
                label="Wikipedia Context Section Description"
            )
        with gr.Row():
            context_page_description = gr.Textbox(
                label="Wikipedia Context Page Description"
            )

        wikipedia_language.change(
            update_captions,
            [wikipedia_language, input_set_name, input_sample_index],
            [
                caption_alt_text_description,
                caption_reference_description,
                caption_title_and_reference_description,
                context_page_description,
                context_section_description,
                hierarchical_section_title,
                page_title,
                section_title,
            ],
        )

        report_textbox = gr.Textbox(
            info="Please describe the issue you found with the sample",
            label="Issue description",
        )
        callback.setup(
            [
                input_set_name,
                input_sample_index,
                wikipedia_language,
                report_textbox,
            ],
            "flagged_data_points",
        )
        report_button = gr.Button("Report Issue")
        report_button.click(
            lambda *args: callback.flag(args),
            [
                input_set_name,
                input_sample_index,
                wikipedia_language,
                report_textbox,
            ],
            None,
            preprocess=False,
        )
        report_button.click(
            lambda x, y: [
                gr.update(
                    value="Issue has been reported. Thank you for your help!",
                    interactive=False,
                ),
                gr.update(
                    value="Issue has been reported. Thank you for your help!",
                    visible=False,
                ),
            ],
            [report_button, report_textbox],
            [report_button, report_textbox],
        )

        # fetch_random_btn.click(update_language_choices, [input_set_name, input_sample_index],
        # [output_language, caption_alt_text_description, caption_reference_description,
        # caption_title_and_reference_description, context_page_description, context_section_description,
        # hierarchical_section_title, page_title, section_title])
        fetch_random_btn.click(
            fn=load_random_sample,
            inputs=[input_set_name],
            outputs=[
                input_sample_index,
                wikipedia_language,
                caption_alt_text_description,
                caption_reference_description,
                caption_title_and_reference_description,
                context_page_description,
                context_section_description,
                hierarchical_section_title,
                page_title,
                section_title,
                youtube_subtitle_text,
                wikipedia_image,
                youtube_video,
                youtube_audio,
                youtube_description_text,
                youtube_title_text,
            ],
        )

        # fetch_btn.click(update_language_choices, [input_set_name, input_sample_index],
        # [output_language, caption_alt_text_description, caption_reference_description,
        # caption_title_and_reference_description, context_page_description, context_section_description,
        # hierarchical_section_title, page_title, section_title])
        fetch_btn.click(
            fn=load_sample,
            inputs=[input_set_name, input_sample_index],
            outputs=[
                wikipedia_language,
                caption_alt_text_description,
                caption_reference_description,
                caption_title_and_reference_description,
                context_page_description,
                context_section_description,
                hierarchical_section_title,
                page_title,
                section_title,
                youtube_subtitle_text,
                wikipedia_image,
                youtube_video,
                youtube_audio,
                youtube_description_text,
                youtube_title_text,
            ],
        )

    demo.queue()
    demo.launch(share=True, debug=True)
