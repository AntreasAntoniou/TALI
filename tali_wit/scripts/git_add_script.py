import os
import subprocess
from tqdm import tqdm
import fire


def add_files_with_progress(
    directory="video_data.parquet", file_list="files_to_add.txt"
):
    # Generate file list
    subprocess.run(
        ["find", directory, "-type", "f"],
        stdout=open(file_list, "w"),
        env=os.environ.copy(),
    )

    # Read file list
    with open(file_list, "r") as f:
        files = [line.strip() for line in f.readlines()]

    # Add files with progress bar using tqdm
    for file in tqdm(files, desc="Adding files to Git"):
        subprocess.run(["git", "add", file], env=os.environ.copy())


if __name__ == "__main__":
    fire.Fire(add_files_with_progress)
