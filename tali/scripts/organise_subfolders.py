import os
import shutil
from pathlib import Path

source_dir = Path("/data/TALI/data/")
destination_dir = Path("/data/TALI/data/")
files = list(source_dir.glob("*"))

for i in range(0, len(files), 5):
    subdir = destination_dir / f"subdir_{i//5}"
    subdir.mkdir(exist_ok=True)

    for file in files[i : i + 5]:
        shutil.move(str(file), str(subdir))
