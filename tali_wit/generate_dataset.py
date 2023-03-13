from datasets import Dataset

from tali_wit.dataset_generator import tali_generator


def train_tali_generator():
    return tali_generator(set_name="train")


def val_tali_generator():
    return tali_generator(set_name="val")


def test_tali_generator():
    return tali_generator(set_name="test")


# ds = Dataset.from_generator(
#     val_tali_generator, cache_dir="/devcode/tali-2-2/val/cache"
# )
# ds.save_to_disk("/devcode/tali-2-2/val.parquet")

# ds = Dataset.from_generator(
#     test_tali_generator, cache_dir="/devcode/tali-2-2/test/cache"
# )
# ds.save_to_disk("/devcode/tali-2-2/test.parquet")

ds = Dataset.from_generator(
    train_tali_generator, cache_dir="/devcode/tali-2-2/train/cache"
)
ds.save_to_disk("/devcode/tali-2-2/train-set")