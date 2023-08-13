import json

import pyarrow as pa
from datasets import Dataset

train_data_dir = "/data/generator/default-e3d897e3cfea555e/0.0.0/"

arrow_files = [
    f"{train_data_dir}/{file_name}"
    for file_name in ["file1.arrow", "file2.arrow"]
]  # Add your Arrow file names here

arrow_tables = [
    pa.ipc.open_file(file_path).read_all() for file_path in arrow_files
]
concatenated_table = pa.concat_tables(arrow_tables)

train_data = Dataset.from_parquet(concatenated_table)

dataset_info = json.load(open(f"{train_data_dir}/dataset_info.json", "r"))

train_data.info = dataset_info
