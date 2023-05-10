from pathlib import Path
import subprocess
from tqdm import tqdm
import fire


def upload_lfs_files(target_dir, upload_batch_size=10):
    # Convert to a Path object
    target_dir = Path(target_dir)

    # Git LFS needs to be initialized per repository
    subprocess.run(["git", "lfs", "install"], cwd=target_dir)

    # Find all files in the directory and its subdirectories
    files = list(target_dir.glob("**/*"))

    # Filter out directories, hidden files and files inside hidden directories
    files = [f for f in files if f.is_file() and not ".git" in f.parts]

    total = len(files)

    # Initialize the progress bar
    pbar = tqdm(total=total, ncols=70)

    for count, file in enumerate(files, 1):
        # Track the file with Git LFS
        subprocess.run(["git", "lfs", "track", str(file)], cwd=target_dir)

        # Add the file to the repository
        subprocess.run(["git", "add", str(file)], cwd=target_dir)

        # If we've processed 10 files, commit and push
        if count % upload_batch_size == 0:
            subprocess.run(
                [
                    "git",
                    "commit",
                    "-m",
                    f"Adding files {count - upload_batch_size} to {count}",
                ],
                cwd=target_dir,
            )
            subprocess.run(["git", "push", "origin", "main"], cwd=target_dir)

            # Delete the last 10 files from the local system
            for i in range(count - upload_batch_size + 1, count + 1):
                files[i].unlink()

            # Update the progress bar
            pbar.update(10)

    # If there are any remaining files (less than 10), commit and push those
    if count % upload_batch_size != 0:
        subprocess.run(
            ["git", "commit", "-m", "Adding remaining files"], cwd=target_dir
        )
        subprocess.run(["git", "push", "origin", "main"], cwd=target_dir)

        # Delete the remaining files from the local system
        for i in range(count - count % upload_batch_size + 1, count + 1):
            files[i].unlink()

        # Final update to the progress bar
        pbar.update(count % upload_batch_size)

    pbar.close()


def main():
    # Expose the function to the command-line
    fire.Fire(upload_lfs_files)


if __name__ == "__main__":
    main()
