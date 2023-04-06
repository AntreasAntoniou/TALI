import os

os.environ["HYDRA_FULL_ERROR"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import pathlib
import random

import tqdm
from rich import print
from rich.traceback import install

import datasets
from tali_wit.data import ModalityTypes


install()

from tali_wit.decorators import configurable
from tali_wit.utils import get_logger, load_json, save_json
from tali_wit.models import ModalityConfig

logger = get_logger(__name__)

from tali_wit.data_plus import TALIBaseDemoTransform, TALIBaseTransformConfig

data_root = "/data_large/datasets/tali-wit-2-1-buckets/"
transform = TALIBaseDemoTransform(
    config=TALIBaseTransformConfig(
        root_filepath=data_root,
        modality_list=[
            ModalityTypes.wit_image.value,
            ModalityTypes.wit_caption.value,
            ModalityTypes.wit_title.value,
            ModalityTypes.wit_main_body.value,
            ModalityTypes.youtube_image.value,
            ModalityTypes.youtube_video.value,
            ModalityTypes.youtube_subtitles.value,
            ModalityTypes.youtube_audio.value,
            ModalityTypes.youtube_description.value,
        ],
        rng_seed=42,
        top_k_tali=10,
        image_size=224,
        num_video_frames=100,
        num_audio_frames=5 * 16000,
        clip_duration_in_seconds=5.0,
        deterministic_sampling=True,
    )
)
train_dataset = datasets.load_from_disk(data_root + "train-set")
train_dataset = train_dataset.with_transform(transform)
val_dataset = datasets.load_from_disk(data_root + "val-set")
val_dataset = val_dataset.with_transform(transform)
test_dataset = datasets.load_from_disk(data_root + "test-set")
test_dataset = test_dataset.with_transform(transform)
num_samples = 0

from collections import defaultdict
from distutils.command.upload import upload
from importlib.resources import path
import gradio as gr
import torchvision
import torchaudio

dataset_dict = {
    "train": train_dataset,
    "val": val_dataset,
    "test": test_dataset,
}


# {
#     'wit_idx': 502620,
#     'wikipedia_caption_image': torch.Size([3, 224, 224]),
#     'wikipedia_text': '<section_title> Station layout </section_title>',
#     'youtube_video_id': '6e7RO-o6u6w',
#     'youtube_content_video': torch.Size([10, 3, 224, 224]),
#     'youtube_content_audio': torch.Size([16000]),
#     'youtube_description_text': "<ysub> it's open somebody's gonna be building something there so the neighborhood
# isn't is in change i don't think shintomicho has much of a personality when they took away the kabuki theater it
# really did change  </ysub>"
# }
from collections import defaultdict
from distutils.command.upload import upload
from importlib.resources import path
import gradio as gr
import torchvision
import torchaudio

dataset_dict = {
    "train": train_dataset,
    "val": val_dataset,
    "test": test_dataset,
}


def update_length_options(set_name):
    max_idx = len(dataset_dict[set_name]) - 1
    return gr.update(minimum=0, maximum=max_idx, step=1)


def get_random_sample(set_name):
    dataset = dataset_dict[set_name]
    sample_index = random.randint(0, len(dataset) - 1)
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
    # print(dataset_dict[set_name][int(sample_index)])
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
    # Extract the text, image, video, and audio from the sample (you'll need to adapt this to your specific dataset)
    subtitles = sample["youtube_description_text"]
    print(
        f"shapes {sample['youtube_content_video'].shape}, {sample['youtube_content_audio'].shape}, {sample['wikipedia_caption_image'].shape}, {sample['youtube_random_video_sample_image'].shape}"
    )
    wit_image = sample["wikipedia_caption_image"].permute(1, 2, 0).numpy()
    youtube_image = (
        sample["youtube_random_video_sample_image"].permute(1, 2, 0).numpy()
    )
    video = sample["youtube_content_video"].permute(0, 2, 3, 1).numpy() * 255
    audio = sample["youtube_content_audio"]

    video_path = f"../demo_cache/temp_data/video-{set_name}-{sample_index}.mp4"
    audio_path = f"../demo_cache/temp_data/audio-{set_name}-{sample_index}.mp3"
    if not pathlib.Path(video_path).parent.exists():
        pathlib.Path(video_path).parent.mkdir(parents=True, exist_ok=True)

    if not pathlib.Path(video_path).exists():
        torchvision.io.write_video(video_path, video, fps=20)
    if not pathlib.Path(audio_path).exists():
        torchaudio.save(audio_path, audio.view(-1).unsqueeze(0), 16000)
    return (
        *update_language_choices(set_name=set_name, sample_index=sample_index),
        subtitles,
        wit_image,
        youtube_image,
        video_path,
        audio_path,
    )


def load_random_sample(set_name):
    sample_idx = get_random_sample(set_name)
    return gr.update(value=sample_idx), *load_sample(set_name, sample_idx)


if __name__ == "__main__":
    callback = gr.CSVLogger()

    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
        # TALI (Temporally and semantically Aligned Audio, Language and Images) Dataset Demo v-0.3.0 🖼️ 🔊 🎦 📝
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
        ### The wikipedia image is semantically aligned to the youtube components, while the youtube components are temporally aligned to each other.
        """
        )
        output_subtitle = gr.Text(label="Youtube Subtitles")
        caption_title_and_reference_description = gr.Textbox(
            label="caption_title_and_reference_description"
        )
        page_title = gr.Textbox(label="page_title")
        with gr.Row():
            with gr.Column():
                output_wit_image = gr.Image(label="Wikipedia Image")
            with gr.Column():
                output_youtube_image = gr.Image(label="Youtube Image")
            with gr.Column():
                output_video = gr.Video(label="Youtube Video")
            with gr.Column():
                output_audio = gr.Audio(label="Youtube Audio")

        gr.Markdown(
            """
        ### Choose what language to display captions in (the captions are in multiple languages)
        """
        )
        output_language = gr.Dropdown(label="Wiki language ID")

        gr.Markdown(
            """
        ### These captions are semantically aligned to the wikipedia image, and should ideally be semantically aligned to the youtube components, however the dataset was selected automatically and this is not always the case. Overall however, the captions are very good at describing the youtube components.
        """
        )

        with gr.Row():
            section_title = gr.Textbox(label="section_title")
            hierarchical_section_title = gr.Textbox(
                label="hierarchical_section_title"
            )
        with gr.Row():
            caption_alt_text_description = gr.Textbox(
                label="caption_alt_text_description"
            )
            caption_reference_description = gr.Textbox(
                label="caption_reference_description"
            )
        with gr.Row():
            context_section_description = gr.Textbox(
                label="context_section_description"
            )
            context_page_description = gr.Textbox(
                label="context_page_description"
            )

        output_language.change(
            update_captions,
            [output_language, input_set_name, input_sample_index],
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
                output_language,
                report_textbox,
            ],
            "flagged_data_points",
        )
        report_button = gr.Button(
            "Report Issue", info="Report an issue with the sample"
        )
        report_button.click(
            lambda *args: callback.flag(args),
            [
                input_set_name,
                input_sample_index,
                output_language,
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

        # fetch_random_btn.click(update_language_choices, [input_set_name, input_sample_index], [output_language, caption_alt_text_description, caption_reference_description, caption_title_and_reference_description, context_page_description, context_section_description, hierarchical_section_title, page_title, section_title])
        fetch_random_btn.click(
            fn=load_random_sample,
            inputs=[input_set_name],
            outputs=[
                input_sample_index,
                output_language,
                caption_alt_text_description,
                caption_reference_description,
                caption_title_and_reference_description,
                context_page_description,
                context_section_description,
                hierarchical_section_title,
                page_title,
                section_title,
                output_subtitle,
                output_wit_image,
                output_youtube_image,
                output_video,
                output_audio,
            ],
        )

        # fetch_btn.click(update_language_choices, [input_set_name, input_sample_index], [output_language, caption_alt_text_description, caption_reference_description, caption_title_and_reference_description, context_page_description, context_section_description, hierarchical_section_title, page_title, section_title])
        fetch_btn.click(
            fn=load_sample,
            inputs=[input_set_name, input_sample_index],
            outputs=[
                output_language,
                caption_alt_text_description,
                caption_reference_description,
                caption_title_and_reference_description,
                context_page_description,
                context_section_description,
                hierarchical_section_title,
                page_title,
                section_title,
                output_subtitle,
                output_wit_image,
                output_youtube_image,
                output_video,
                output_audio,
            ],
        )

    demo.queue(concurrency_count=8)
    demo.launch(share=True, debug=True, enable_queue=True)
