import pathlib
import shutil
import datasets

from tali_wit.dataset_cache_generator import tali_cache_generator


datasets.enable_caching()


def train_tali_generator():
    return tali_cache_generator(set_name="train", num_samples=100000000)


def val_tali_generator():
    return tali_cache_generator(set_name="val", num_samples=10000)


def test_tali_generator():
    return tali_cache_generator(set_name="test", num_samples=10000)


if pathlib.Path("/home/evolvingfungus/tali_cache/").exists():
    shutil.rmtree("/home/evolvingfungus/tali_cache/")

ds = datasets.Dataset.from_generator(
    val_tali_generator,
    cache_dir="/home/evolvingfungus/tali_cache/val/",
    writer_batch_size=100,
)

ds = datasets.Dataset.from_generator(
    test_tali_generator,
    cache_dir="/home/evolvingfungus/tali_cache/test/",
    writer_batch_size=100,
)

ds = datasets.Dataset.from_generator(
    train_tali_generator,
    cache_dir="/home/evolvingfungus/tali_cache/train/",
    writer_batch_size=100,
)
