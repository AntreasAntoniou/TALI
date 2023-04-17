from datasets import Dataset

from tali_wit.dataset_cache_generator import tali_cache_generator


def train_tali_generator():
    return tali_cache_generator(set_name="train")


def val_tali_generator():
    return tali_cache_generator(set_name="val")


def test_tali_generator():
    return tali_cache_generator(set_name="test")


ds = Dataset.from_generator(
    val_tali_generator, cache_dir="/devcode/tali-2-2/val/cache"
)
ds.save_to_disk("/devcode/tali-2-2/val-set")

ds = Dataset.from_generator(
    test_tali_generator, cache_dir="/devcode/tali-2-2/test/cache"
)
ds.save_to_disk("/devcode/tali-2-2/test-set")

# ds = Dataset.from_generator(
#     train_tali_generator, cache_dir="/devcode/tali-2-2/train/cache"
# )
# ds.save_to_disk("/devcode/tali-2-2/train-set")
