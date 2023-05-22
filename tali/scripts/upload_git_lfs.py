from pathlib import Path
import subprocess
import sys
from tqdm import tqdm
from rich import print
import fire


def upload_lfs_files(target_dir, upload_batch_size: int = 10):
    # Convert to a Path object
    target_dir = Path(target_dir)

    # Git LFS needs to be initialized per repository
    subprocess.run(["git", "lfs", "install"], cwd=target_dir)

    # Find all files in the directory and its subdirectories
    files = list(target_dir.glob("**/*"))

    # Filter out directories, hidden files and files inside hidden directories
    files = [
        f
        for f in files
        if f.is_file() and ".git" not in f.parts and not f.name.startswith(".")
    ]

    total = len(files)

    # Initialize the progress bar
    pbar = tqdm(total=total, ncols=70)

    for count, file in enumerate(files, 1):
        # Track the file with Git LFS
        print(f"Tracking {file}")

        subprocess.run(
            ["git", "lfs", "track", file.as_posix()],
            cwd=target_dir,
            stdout=sys.stdout,
            stderr=sys.stdout,
        )

        # Add the file to the repository
        print(f"Adding {file}")
        subprocess.run(
            ["git", "add", file.as_posix()],
            cwd=target_dir,
            stdout=sys.stdout,
            stderr=sys.stdout,
        )

        # If we've processed 10 files, commit and push
        if count % upload_batch_size == 0:
            print(
                f"Committing and pushing files {count - upload_batch_size} to {count}"
            )
            subprocess.run(
                [
                    "git",
                    "commit",
                    "-m",
                    f"Adding files {count - upload_batch_size} to {count}",
                ],
                cwd=target_dir,
                stdout=sys.stdout,
                stderr=sys.stdout,
            )
            print("Pushing to remote")
            subprocess.run(
                ["git", "push", "origin", "main"],
                cwd=target_dir,
                stdout=sys.stdout,
                stderr=sys.stdout,
            )

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
