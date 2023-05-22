import multiprocessing as mp
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Antreas/TALI",
    repo_type="dataset",
    cache_dir="/tali-data/TALI",
    local_dir="/tali-data/TALI",
    resume_download=True,
    max_workers=mp.cpu_count(),
)
