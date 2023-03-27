from huggingface_hub import HfApi

hf_api = HfApi()

# files = hf_api.list_repo_files(
#     repo_id="Antreas/tali-2-tali_omni_base_patch16_224-wit_tali_image_text_audio_video_dataset-42"
# )

# commits = hf_api.list_repo_commits(
#     repo_id="Antreas/tali-2-tali_omni_base_patch16_224-wit_tali_image_text_audio_video_dataset-42",
# )

# refs = hf_api.list_repo_refs(
#     repo_id="Antreas/tali-2-tali_omni_base_patch16_224-wit_tali_image_text_audio_video_dataset-42",
# )

# info = hf_api.repo_info(
#     repo_id="Antreas/tali-2-tali_omni_base_patch16_224-wit_tali_image_text_audio_video_dataset-42"
# )

files = hf_api.list_repo_files(
    repo_id="Antreas/tali-2-tali_omni_base_patch16_224-wit_tali_image_text_audio_video_dataset-42"
)

ckpt_dict = {}
for file in files:
    if "checkpoints/ckpt" in file:
        ckpt_global_step = int(file.split("/")[-2].split("_")[-1])
        ckpt_dict[ckpt_global_step] = "/".join(file.split("/")[:-1])

print(ckpt_dict)
