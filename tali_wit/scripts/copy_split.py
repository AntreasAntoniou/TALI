import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import List

from fire import Fire
import tqdm

# Set the minimum and maximum chunk size in bytes
min_chunk_size = 4 * 1024**3  # 4 GB
max_chunk_size = 4.5 * 1024**3  # 4.5 GB

# Set the Zstandard compression level
compression_level = 1


def compress_chunk(filepaths: List[Path], output_path: Path) -> None:
    tar_cmd = ["tar", "-cf", "-", *map(str, filepaths)]
    zstd_cmd = ["zstd", f"-{compression_level}", "-o", str(output_path)]

    tar_proc = subprocess.Popen(tar_cmd, stdout=subprocess.PIPE)
    zstd_proc = subprocess.Popen(zstd_cmd, stdin=tar_proc.stdout)
    tar_proc.stdout.close()  # Close the pipe after the subprocess ends
    zstd_proc.communicate()  # Wait for the zstd process to finish


def main(input_folder_dir: str, output_folder_dir: str):
    input_folder_path = Path(input_folder_dir)
    output_folder_path = Path(output_folder_dir)
    output_folder_path.mkdir(parents=True, exist_ok=True)

    filepaths = []
    chunk_number = 0
    chunk_size = 0

    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        with tqdm.tqdm() as pbar:
            for root, _, files in os.walk(input_folder_path):
                for file in files:
                    file_path = Path(root) / file
                    if chunk_size + file_path.stat().st_size <= max_chunk_size:
                        filepaths.append(file_path)
                        chunk_size += file_path.stat().st_size
                    else:
                        output_path = (
                            output_folder_path
                            / f"archive_{chunk_number:04d}.tar.zst"
                        )
                        executor.submit(compress_chunk, filepaths, output_path)
                        filepaths = [file_path]
                        chunk_size = file_path.stat().st_size
                        chunk_number += 1
                    pbar.update(1)

            if filepaths:  # Compress remaining files
                output_path = (
                    output_folder_path / f"archive_{chunk_number:04d}.tar.zst"
                )
                executor.submit(compress_chunk, filepaths, output_path)


if __name__ == "__main__":
    Fire(main)
