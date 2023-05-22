import os
from tqdm.auto import tqdm

# os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

from huggingface_hub import HfApi

target_dir = "/data-fast1/TALI/data/"
api = HfApi()
# video_data_part.7z.001"


for subdir, dirs, files in tqdm(os.walk(target_dir)):
    if "subdir" not in subdir:
        continue
    print(f"Uploading {subdir}, to, data/{subdir.split('/')[-1]}")
    api.upload_folder(
        folder_path=subdir,
        path_in_repo=f"data/{subdir.split('/')[-1]}",
        repo_id="Antreas/TALI",
        repo_type="dataset",
    )

# target_dir = "/data-fast1/TALI/data/subdir_7"
# api.upload_folder(
#     folder_path=target_dir,
#     path_in_repo=f"data/{target_dir.split('/')[-1]}",
#     repo_id="Antreas/TALI",
#     repo_type="dataset",
# )
