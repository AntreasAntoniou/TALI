import pathlib
import datasets
import tqdm
import shutil
from rich import print
import fire


if __name__ == "__main__":
    from huggingface_hub import Repository
    import os

    tali_dataset_dir = "/data_fast/TALI/"

    # Initialize the repository in the existing dataset folder
    repo = Repository(
        local_dir=tali_dataset_dir,
        clone_from="Antreas/TALI",
        repo_type="dataset",
    )
    batch_size_in_mb = 1
    cur_idx = 0
    commited_files = []
    cur_commit_size = 0
    # Iterate through the dataset folder and add all existing files
    with tqdm.tqdm(total=25_000_000) as pbar:
        for root, _, files in os.walk(tali_dataset_dir):
            for file in files:
                if (
                    # file.endswith(".parquet")
                    file.endswith(".mp4")
                    # or file.endswith(".json")
                    # or file.endswith(".arrow")
                ):
                    file_path = os.path.join(root, file)
                    repo_path = os.path.relpath(file_path, tali_dataset_dir)

                    # Add the existing file to the repository
                    repo.git_add(repo_path)
                    commited_files.append(repo_path)
                    # get file size in mega bytes
                    size = os.path.getsize(file_path) / 1e6
                    cur_commit_size += size

                    cur_idx += 1
                    pbar.update(size)
                try:
                    if cur_commit_size >= batch_size_in_mb and len(commited_files) > 0:
                        # Commit and push all files to the repository
                        repo.git_commit(f"Add existing dataset files {cur_idx}")
                        repo.git_push()
                        commited_files = []
                        cur_commit_size = 0
                        print(f"PUSHED {cur_idx} files")
                except Exception as e:
                    print(e)
                    print("Failed to commit and push files to the hub!")
                    print(f"Current index: {cur_idx}")
                    # print(f"Current repo files: {commited_files}")

    print("Existing dataset files uploaded to the hub!")
