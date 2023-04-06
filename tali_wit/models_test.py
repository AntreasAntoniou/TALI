import os
import hydra
from hydra_zen import instantiate

from tali_wit.config import BaseConfig, collect_config_store
from tali_wit.ctools import get_max_supported_batch_size
from tali_wit.models import TALIModel
from tali_wit.utils import get_logger, pretty_config, set_seed

import tqdm
from rich import print
from rich.traceback import install

logger = get_logger(name=__name__)
config_store = collect_config_store()


@hydra.main(config_path=None, config_name="config", version_base=None)
def test(cfg: BaseConfig) -> None:
    print(pretty_config(cfg, resolve=True))

    set_seed(seed=cfg.seed)

    model: TALIModel = instantiate(cfg.model)

    for dataset_name, (batch_size, dataset) in cfg.dataset.items():
        logger.info(f"Setting up {dataset_name} train dataset")
        train_dataset = instantiate(
            dataset, set_name="train", infinite_sampling=True
        )
        logger.info(f"Setting up {dataset_name} train dataloader")
        train_dataloader = instantiate(
            cfg.dataloader,
            dataset=train_dataset,
            batch_size=1,
            shuffle=True,
        )
        dummy_batch = next(iter(train_dataloader))
        logger.info(
            f"Finding max batch size for {dataset_name} train dataloader"
        )
        optimal_batch_size = get_max_supported_batch_size(
            model=model, batch=dummy_batch, train_mode=True
        )
        if "audio" in dataset_name:
            optimal_batch_size //= 2

        train_dataloader = instantiate(
            cfg.dataloader,
            dataset=train_dataset,
            batch_size=optimal_batch_size,
            shuffle=True,
        )


if __name__ == "__main__":
    # test all possible options

    install()
    os.environ["HYDRA_FULL_ERROR"] = "1"
    test()
    # dummy_input = torch.randn((10000, 10000))
    # print(contrastive_accuracy(dummy_input))
    # print(contrastive_accuracy_top_k(dummy_input, k=5))

    # For 24GB GPU at 16-bit
    # 128 batch size for wit_image to text
    # 18 batch size for wit_image to youtube_audio
    # 24 batch size for wit_image to youtube_video
    # 20 batch size for text to youtube_audio
    # 16 batch size for text to youtube_video
    # 12 batch size for audio to youtube_video
    # set up a config that allows to select modality pairs + batch size and then build unique dataloaders for each pair
    # add accuracy and top-k accuracy metrics
