import multiprocessing as mp

from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Antreas/TALI",
    repo_type="dataset",
    cache_dir="/data/",
    local_dir="/data/TALI/",
    resume_download=True,
    max_workers=mp.cpu_count(),
)

# from huggingface_hub import hf_hub_download

# hf_hub_download(
#     repo_id="Antreas/TALI",
#     repo_type="dataset",
#     subfolder="test-set",
#     filename="data-00000-of-00001.arrow",
#     cache_dir="/tali-data/",
#     local_dir="/tali-data/",
#     local_dir_use_symlinks=True,
# )

# hf_hub_download(
#     repo_id="Antreas/TALI",
#     repo_type="dataset",
#     subfolder="test-set",
#     filename="dataset_info.json",
#     cache_dir="/tali-data/",
#     local_dir="/tali-data/",
#     local_dir_use_symlinks=True,
# )

# hf_hub_download(
#     repo_id="Antreas/TALI",
#     repo_type="dataset",
#     subfolder="test-set",
#     filename="state.json",
#     cache_dir="/tali-data/",
#     local_dir="/tali-data/",
#     local_dir_use_symlinks=True,
# )
# ####################
# hf_hub_download(
#     repo_id="Antreas/TALI",
#     repo_type="dataset",
#     subfolder="val-set",
#     filename="data-00000-of-00001.arrow",
#     cache_dir="/tali-data/",
#     local_dir="/tali-data/",
#     local_dir_use_symlinks=True,
# )

# hf_hub_download(
#     repo_id="Antreas/TALI",
#     repo_type="dataset",
#     subfolder="val-set",
#     filename="dataset_info.json",
#     cache_dir="/tali-data/",
#     local_dir="/tali-data/",
#     local_dir_use_symlinks=True,
# )

# hf_hub_download(
#     repo_id="Antreas/TALI",
#     repo_type="dataset",
#     subfolder="val-set",
#     filename="state.json",
#     cache_dir="/tali-data/",
#     local_dir="/tali-data/",
#     local_dir_use_symlinks=True,
# )
# ####################

# for i in range(9):
#     hf_hub_download(
#         repo_id="Antreas/TALI",
#         repo_type="dataset",
#         subfolder="train-set",
#         filename=f"data-0000{i}-of-00009.arrow",
#         cache_dir="/tali-data/",
#         local_dir="/tali-data/",
#         local_dir_use_symlinks=True,
#     )

# hf_hub_download(
#     repo_id="Antreas/TALI",
#     repo_type="dataset",
#     subfolder="train-set",
#     filename="dataset_info.json",
#     cache_dir="/tali-data/",
#     local_dir="/tali-data/",
#     local_dir_use_symlinks=True,
# )

# hf_hub_download(
#     repo_id="Antreas/TALI",
#     repo_type="dataset",
#     subfolder="train-set",
#     filename="state.json",
#     cache_dir="/tali-data/",
#     local_dir="/tali-data/",
#     local_dir_use_symlinks=True,
# )
