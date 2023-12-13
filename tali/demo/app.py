import pathlib
import random

import gradio as gr

from tali.data.data_new import load_dataset_via_hub
from tali.utils import get_logger

logger = get_logger(__name__)

from tali.data.data_new import TALIBaseDemoTransform

dataset_cache = pathlib.Path("/disk/scratch_fast0/tali/")
dataset_dict = load_dataset_via_hub(dataset_cache, dataset_name="Antreas/TALI")
demo_transform = TALIBaseDemoTransform(cache_dir=dataset_cache / "cache")
dataset_length_dict = {
    "train": len(dataset_dict["train"]),
    "val": len(dataset_dict["val"]),
    "test": len(dataset_dict["test"]),
}


dataset_dict.set_transform(demo_transform)


num_samples = 0

# Sample keys:
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
    flagged_logger_path = pathlib.Path(__file__).parent / "flagged.csv"

    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        # load markdown from intro.md file
        intro_md_path = pathlib.Path(__file__).parent / "intro.md"
        with open(intro_md_path, "r") as f:
            gr.Markdown(f.read())

        gr.Markdown(
            """
            ### Introduction:
            This multi-modal dataset combines YouTube and Wikipedia data, which are aligned temporally and semantically to different extents. Let's explore them!
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
                    maximum=dataset_length_dict["train"] - 1,
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

        gr.Markdown("## Explore the Multi-modal Components")

        with gr.Row():
            with gr.Column():  # Wikipedia Components Column
                gr.Markdown(
                    """
            ## Wikipedia Components - Text and Images
            - The Wikipedia page title is *not temporally aligned* but is *strongly semantically aligned* with the broader topic or context of the YouTube content.
            - The Wikipedia sections and descriptions are *weakly temporally aligned* but *strongly semantically aligned*. They provide a structured textual representation and interpretation of the content.
            - The Wikipedia images, titles and references' descriptions provide *weak temporal alignment* but *strong semantic alignment* with the YouTube content.
            """
                )

                # All the Wikipedia related components go here
                wikipedia_image = gr.Image(label="Wikipedia Image")
                wikipedia_language = gr.Dropdown(label="Wiki language ID")

                page_title = gr.Textbox(label="Wikipedia Page Title")
                section_title = gr.Textbox(label="Wikipedia Section Title ")
                hierarchical_section_title = gr.Textbox(
                    label="Hierarchical Section Title "
                )
                caption_title_and_reference_description = gr.Textbox(
                    label="Caption Title Reference "
                )
                caption_alt_text_description = gr.Textbox(
                    label="Caption Alt-Text "
                )
                caption_reference_description = gr.Textbox(
                    label="Caption Reference Description "
                )
                context_section_description = gr.Textbox(
                    label="Context Section Description "
                )
                context_page_description = gr.Textbox(
                    label="Context Page Description "
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

            with gr.Column():  # YouTube Components Column
                gr.Markdown(
                    """
                ## YouTube Components - Video, Audio and Text
                - The video and audio are *strongly temporally and semantically aligned*, as they originate from the same source.
                - The YouTube subtitles are *strongly temporally and semantically aligned* with the video/audio content, transcribing the spoken content.
                - The YouTube description is *weakly temporally aligned* but *strongly semantically aligned*. It describes the overall video content but doesn't map to specific timestamps.
                """
                )

                # All the YouTube related components go here

                youtube_video = gr.Video(label="Youtube Video ")
                youtube_subtitle_text = gr.Text(label="Youtube Subtitles ")

                youtube_audio = gr.Audio(label="Youtube Audio ")
                youtube_title_text = gr.Textbox(label="Youtube Title ")

                youtube_description_text = gr.Textbox(
                    label="Youtube Description "
                )

        gr.Markdown(
            """
            ## Issue Reporting
            
            Should you encounter samples that aren't appropriately aligned or notice other issues, please write a description in the 'Issue description' textbox and press the 'Report issue' button.
            """
        )
        report_textbox = gr.Textbox(
            info="Please describe the issue you found with the sample",
            label="Issue description",
        )
        callback.setup(
            components=[
                input_set_name,
                input_sample_index,
                wikipedia_language,
                report_textbox,
            ],
            flagging_dir=flagged_logger_path,
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
