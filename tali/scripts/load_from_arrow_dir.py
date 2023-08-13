import json
import pathlib

import pyarrow as pa
from datasets import Dataset

train_data_dir = pathlib.Path(
    "/data/generator/default-e3d897e3cfea555e/0.0.0/"
)

arrow_files = [
    file_name
    for file_name in train_data_dir.iterdir()
    if file_name.suffix == ".arrow"
]  # Add your Arrow file names here

arrow_tables = [
    pa.ipc.open_file(file_path).read_all() for file_path in arrow_files
]
concatenated_table = pa.concat_tables(arrow_tables)

train_data = Dataset.from_parquet(concatenated_table)

dataset_info = json.load(open(f"{train_data_dir}/dataset_info.json", "r"))

train_data.info = dataset_info
