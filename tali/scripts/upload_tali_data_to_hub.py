import os
import glob
import pathlib
import huggingface_hub as hf_hub
from tqdm import tqdm

os.environ["HF_TOKEN"] = "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"

dataset_dir = "/data-unified/TALI"
client = hf_hub.HfApi(token=os.environ["HF_TOKEN"])
repo = hf_hub.Repository(
    local_dir=dataset_dir,
    clone_from="Antreas/TALI",
    token=client.token,
    repo_type="dataset",
    client=client,
)
repo.git_pull()

# Find files with "video_data_part" in their name
files = glob.glob(f"{dataset_dir}/**")

# Iterate through the files with tqdm progress bar, adding and committing each one
count = 1
upload_processes = []
for file in tqdm(files):
    print(f"Uploading {os.path.basename(file)}")
    try:
        repo.git_add(file, auto_lfs_track=True)
        repo.git_commit(f"Upload {os.path.basename(file)}")
        count += 1
    except Exception as e:
        print(f"Error: {e}")

    if count % 1 == 0:
        push_process = repo.git_push(blocking=True)
        upload_processes.append(push_process)

# while len(upload_processes) > 0:
#     for process in upload_processes:
#         if process.is_done is True:
#             upload_processes.remove(process)
