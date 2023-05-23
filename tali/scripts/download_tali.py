import multiprocessing as mp
from huggingface_hub import snapshot_download, hf_hub_download

snapshot_download(
    repo_id="Antreas/TALI",
    repo_type="dataset",
    cache_dir="/tali-data/TALI",
    local_dir="/tali-data/",
    resume_download=True,
    max_workers=mp.cpu_count(),
)

hf_hub_download(
    filename="caption_data.7z",
    subfolder="data",
    repo_id="Antreas/TALI",
    repo_type="dataset",
    cache_dir="/tali-data/TALI",
    local_dir="/tali-data/TALI",
)
