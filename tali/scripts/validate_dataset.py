import datasets
import pathlib
import tqdm.auto as tqdm

tali_dataset_dir = "/tali-data/TALI"

if __name__ == "__main__":
    set_name = "train"
    dataset = datasets.load_from_disk(
        pathlib.Path(tali_dataset_dir) / f"{set_name}-set",
        keep_in_memory=True,
    )

    for item in tqdm(dataset):
        print(item)
