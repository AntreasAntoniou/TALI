import os

import huggingface_hub as hf_hub

client = hf_hub.HfApi(token=os.environ["HF_TOKEN"])
repo = hf_hub.Repository(
    local_dir="/data-fast1/datasets/",
    clone_from="Antreas/TALI",
    token=client.token,
    repo_type="dataset",
)
repo.git_pull()

repo.git_add("data/**", auto_lfs_track=True)
repo.git_commit("Update TALI dataset")
repo.git_push()
