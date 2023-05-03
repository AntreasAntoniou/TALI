import logging
import shutil
import signal
from functools import wraps

import accelerate
import torch
from huggingface_hub import (
    create_repo,
    hf_hub_download,
    login,
    snapshot_download,
)
from omegaconf import DictConfig, OmegaConf
from rich.logging import RichHandler
from rich.syntax import Syntax
from rich.traceback import install
from rich.tree import Tree


def get_logger(
    name=__name__, logging_level: str = None, set_rich: bool = False
) -> logging.Logger:
    """Initializes multi-GPU-friendly python command line logger."""

    logger = logging.getLogger(name)

    logging_level = logging_level or logging.INFO

    logger.setLevel(logging_level)

    if set_rich:
        ch = RichHandler()

        # create formatter
        formatter = logging.Formatter("%(message)s")

        # add formatter to ch
        ch.setFormatter(formatter)

        # add ch to logger
        logger.addHandler(ch)

    install()

    # this ensures all logging levels get marked with the rank zero decorator
    # otherwise logs would get multiplied for each GPU process in multi-GPU setup

    return logger


def get_hydra_config(logger_level: str = "INFO"):
    return dict(
        job_logging=dict(
            version=1,
            formatters=dict(
                simple=dict(
                    level=logger_level,
                    format="%(message)s",
                    datefmt="[%X]",
                )
            ),
            handlers=dict(
                rich={
                    "class": "rich.logging.RichHandler",
                    # "formatter": "simple",
                }
            ),
            root={"handlers": ["rich"], "level": logger_level},
            disable_existing_loggers=False,
        ),
        hydra_logging=dict(
            version=1,
            formatters=dict(
                simple=dict(
                    level=logging.CRITICAL,
                    format="%(message)s",
                    datefmt="[%X]",
                )
            ),
            handlers={
                "rich": {
                    "class": "rich.logging.RichHandler",
                    # "formatter": "simple",
                }
            },
            root={"handlers": ["rich"], "level": logging.CRITICAL},
            disable_existing_loggers=False,
        ),
        run={
            "dir": "${current_experiment_dir}/hydra-run/${now:%Y-%m-%d_%H-%M-%S}"
        },
        sweep={
            "dir": "${current_experiment_dir}/hydra-multirun/${now:%Y-%m-%d_%H-%M-%S}",
            "subdir": "${hydra.job.num}",
        },
    )


def timeout(timeout_secs: int):
    def wrapper(func):
        @wraps(func)
        def time_limited(*args, **kwargs):
            # Register an handler for the timeout
            def handler(signum, frame):
                raise Exception(f"Timeout for function '{func.__name__}'")

            # Register the signal function handler
            signal.signal(signal.SIGALRM, handler)

            # Define a timeout for your function
            signal.alarm(timeout_secs)

            result = None
            try:
                result = func(*args, **kwargs)
            except Exception as exc:
                logging.error(f"Exploded due to time out on {args, kwargs}")
                raise exc
            finally:
                # disable the signal alarm
                signal.alarm(0)

            return result

        return time_limited

    return wrapper


def demo_logger():
    logger = get_logger(__name__)

    logger.info("Hello World")
    logger.debug("Debugging")
    logger.warning("Warning")
    logger.error("Error")
    logger.critical("Critical")
    logger.exception("Exception")


def set_seed(seed: int):
    accelerate.utils.set_seed(seed)


def pretty_config(
    config: DictConfig,
    resolve: bool = True,
):
    """Prints content of DictConfig using Rich library and its tree structure.

    Args:
        config (DictConfig): Configuration composed by Hydra.
        fields (Sequence[str], optional): Determines which main fields from config will
        be printed and in what order.
        resolve (bool, optional): Whether to resolve reference fields of DictConfig.
    """

    style = "dim"
    tree = Tree("CONFIG", style=style, guide_style=style)

    for field in config.keys():
        branch = tree.add(field, style=style, guide_style=style)

        config_section = config.get(field)
        branch_content = str(config_section)
        if isinstance(config_section, DictConfig):
            branch_content = OmegaConf.to_yaml(config_section, resolve=resolve)

        branch.add(Syntax(branch_content, "yaml"))

    return tree


import os
import os.path
import pathlib
from typing import Any, Dict, Union

import orjson as json


def save_json(
    filepath: Union[str, pathlib.Path], dict_to_store: Dict, overwrite=True
):
    """
    Saves a metrics .json file with the metrics
    :param log_dir: Directory of log
    :param metrics_file_name: Name of .csv file
    :param dict_to_store: A dict of metrics to add in the file
    :param overwrite: If True overwrites any existing files with the same filepath,
    if False adds metrics to existing
    """

    if isinstance(filepath, str):
        filepath = pathlib.Path(filepath)

    if overwrite and filepath.exists():
        filepath.unlink(missing_ok=True)

    if not filepath.parent.exists():
        filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, "wb") as json_file:
        json_file.write(json.dumps(dict_to_store))
    return filepath


def load_json(filepath: Union[str, pathlib.Path]):
    """
    Loads the metrics in a dictionary.
    :param log_dir: The directory in which the log is saved
    :param metrics_file_name: The name of the metrics file
    :return: A dict with the metrics
    """

    if isinstance(filepath, str):
        filepath = pathlib.Path(filepath)

    with open(filepath, "rb") as json_file:
        dict_to_load = json.loads(json_file.read())

    return dict_to_load


logger = get_logger(name=__name__)


def download_model_with_name(
    hf_repo_path, hf_cache_dir, model_name, download_only_if_finished=False
):
    if not pathlib.Path(
        pathlib.Path(hf_cache_dir) / "checkpoints" / f"{model_name}"
    ).exists():
        pathlib.Path(
            pathlib.Path(hf_cache_dir) / "checkpoints" / f"{model_name}"
        ).mkdir(parents=True, exist_ok=True)

    config_filepath = hf_hub_download(
        repo_id=hf_repo_path,
        cache_dir=pathlib.Path(hf_cache_dir),
        resume_download=True,
        filename="config.yaml",
        repo_type="model",
    )

    config_path = pathlib.Path(hf_cache_dir) / "config.yaml"

    shutil.copy(
        pathlib.Path(config_filepath),
        config_path,
    )

    trainer_state_filepath = hf_hub_download(
        repo_id=hf_repo_path,
        cache_dir=pathlib.Path(hf_cache_dir),
        resume_download=True,
        subfolder=f"checkpoints/{model_name}",
        filename="trainer_state.pt",
        repo_type="model",
    )

    trainer_path = (
        pathlib.Path(hf_cache_dir)
        / "checkpoints"
        / f"{model_name}"
        / "trainer_state.pt"
    )

    shutil.copy(
        pathlib.Path(trainer_state_filepath),
        trainer_path,
    )
    logger.info(
        f"Trainer state copied to {trainer_path} from {trainer_state_filepath}."
    )

    if download_only_if_finished:
        state_dict = torch.load(trainer_path)["state_dict"]["eval"][0][
            "auc-macro"
        ]
        global_step_list = list(state_dict.keys())
        if len(global_step_list) < 40:
            return False

    optimizer_filepath = hf_hub_download(
        repo_id=hf_repo_path,
        cache_dir=pathlib.Path(hf_cache_dir),
        resume_download=True,
        subfolder=f"checkpoints/{model_name}",
        filename="optimizer.bin",
        repo_type="model",
    )

    model_filepath = hf_hub_download(
        repo_id=hf_repo_path,
        cache_dir=pathlib.Path(hf_cache_dir),
        resume_download=True,
        subfolder=f"checkpoints/{model_name}",
        filename="pytorch_model.bin",
        repo_type="model",
    )

    random_states_filepath = hf_hub_download(
        repo_id=hf_repo_path,
        cache_dir=pathlib.Path(hf_cache_dir),
        resume_download=True,
        subfolder=f"checkpoints/{model_name}",
        filename="random_states_0.pkl",
        repo_type="model",
    )

    try:
        scaler_state_filepath = hf_hub_download(
            repo_id=hf_repo_path,
            cache_dir=pathlib.Path(hf_cache_dir),
            resume_download=True,
            subfolder=f"checkpoints/{model_name}",
            filename="scaler.pt",
            repo_type="model",
        )
    except Exception as e:
        scaler_state_filepath = None

    target_optimizer_path = (
        pathlib.Path(hf_cache_dir)
        / "checkpoints"
        / f"{model_name}"
        / "optimizer.bin"
    )

    shutil.copy(
        pathlib.Path(optimizer_filepath),
        target_optimizer_path,
    )

    target_model_path = (
        pathlib.Path(hf_cache_dir)
        / "checkpoints"
        / f"{model_name}"
        / "pytorch_model.bin"
    )

    shutil.copy(
        pathlib.Path(model_filepath),
        target_model_path,
    )

    random_states_path = (
        pathlib.Path(hf_cache_dir)
        / "checkpoints"
        / f"{model_name}"
        / "random_states_0.pkl"
    )

    shutil.copy(
        pathlib.Path(random_states_filepath),
        random_states_path,
    )

    if scaler_state_filepath is not None:
        scaler_path = (
            pathlib.Path(hf_cache_dir)
            / "checkpoints"
            / f"{model_name}"
            / "scaler.pt"
        )

        shutil.copy(
            pathlib.Path(scaler_state_filepath),
            scaler_path,
        )

    return {
        "root_filepath": pathlib.Path(hf_cache_dir)
        / "checkpoints"
        / f"{model_name}",
        "optimizer_filepath": target_optimizer_path,
        "model_filepath": target_model_path,
        "random_states_filepath": random_states_path,
        "trainer_state_filepath": trainer_path,
        "config_filepath": config_path,
    }


def create_hf_model_repo_and_download_maybe(cfg: Any):
    import yaml
    from huggingface_hub import HfApi

    if (
        cfg.download_checkpoint_with_name is not None
        and cfg.download_latest is True
    ):
        raise ValueError(
            "Cannot use both continue_from_checkpoint_with_name and continue_from_latest"
        )

    hf_repo_path = cfg.hf_repo_path
    login(token=os.environ["HF_TOKEN"], add_to_git_credential=True)
    print(
        f"Logged in to huggingface with token {os.environ['HF_TOKEN']}, creating repo {hf_repo_path}"
    )
    repo_url = create_repo(hf_repo_path, repo_type="model", exist_ok=True)

    logger.info(f"Created repo {hf_repo_path}, {cfg.hf_cache_dir}")

    if not pathlib.Path(cfg.hf_cache_dir).exists():
        pathlib.Path(cfg.hf_cache_dir).mkdir(parents=True, exist_ok=True)
        pathlib.Path(pathlib.Path(cfg.hf_cache_dir) / "checkpoints").mkdir(
            parents=True, exist_ok=True
        )

    config_dict = OmegaConf.to_container(cfg, resolve=True)

    hf_api = HfApi()

    config_json_path: pathlib.Path = save_json(
        filepath=pathlib.Path(cfg.hf_cache_dir) / "config.json",
        dict_to_store=config_dict,
        overwrite=True,
    )

    hf_api.upload_file(
        repo_id=hf_repo_path,
        path_or_fileobj=config_json_path.as_posix(),
        path_in_repo="config.json",
    )

    config_yaml_path = pathlib.Path(cfg.hf_cache_dir) / "config.yaml"
    with open(config_yaml_path, "w") as file:
        documents = yaml.dump(config_dict, file)

    hf_api.upload_file(
        repo_id=hf_repo_path,
        path_or_fileobj=config_yaml_path.as_posix(),
        path_in_repo="config.yaml",
    )

    try:
        if cfg.resume == False and not (
            cfg.download_checkpoint_with_name is not None
            or cfg.download_latest
        ):
            return None, repo_url

        if cfg.download_checkpoint_with_name is not None:
            logger.info(
                f"Download {cfg.download_checkpoint_with_name} checkpoint, if it exists, from the huggingface hub 👨🏻‍💻"
            )

            path_dict = download_model_with_name(
                hf_repo_path=hf_repo_path,
                hf_cache_dir=cfg.hf_cache_dir,
                model_name=cfg.download_checkpoint_with_name,
            )
            logger.info(
                f"Downloaded checkpoint from huggingface hub to {cfg.hf_cache_dir}"
            )
            return path_dict["root_filepath"], repo_url

        elif cfg.download_latest:
            files = hf_api.list_repo_files(
                repo_id=hf_repo_path,
            )

            ckpt_dict = {}
            for file in files:
                if "checkpoints/ckpt" in file:
                    ckpt_global_step = int(file.split("/")[-2].split("_")[-1])
                    ckpt_dict[ckpt_global_step] = "/".join(
                        file.split("/")[:-1]
                    )

            latest_ckpt = ckpt_dict[max(ckpt_dict.keys())]

            model_dir = pathlib.Path(cfg.hf_cache_dir) / latest_ckpt
            if model_dir.exists():
                logger.info("Checkpoint exists, skipping download")
            else:
                logger.info(
                    "Download latest checkpoint, if it exists, from the huggingface hub 👨🏻‍💻"
                )
                path_dict = download_model_with_name(
                    model_name=latest_ckpt.split("/")[-1],
                    hf_repo_path=hf_repo_path,
                    hf_cache_dir=cfg.hf_cache_dir,
                )
                logger.info(
                    f"Downloaded checkpoint from huggingface hub to {cfg.hf_cache_dir}"
                )
            return (
                model_dir,
                repo_url,
            )
        else:
            logger.info(
                "Download all available checkpoints, if they exist, from the huggingface hub 👨🏻‍💻"
            )

            ckpt_folderpath = snapshot_download(
                repo_id=hf_repo_path,
                cache_dir=pathlib.Path(cfg.hf_cache_dir),
                resume_download=True,
            )
            latest_checkpoint = (
                pathlib.Path(cfg.hf_cache_dir) / "checkpoints" / "latest"
            )

            if pathlib.Path(
                pathlib.Path(cfg.hf_cache_dir) / "checkpoints"
            ).exists():
                pathlib.Path(
                    pathlib.Path(cfg.hf_cache_dir) / "checkpoints"
                ).mkdir(parents=True, exist_ok=True)

            shutil.copy(
                pathlib.Path(ckpt_folderpath), cfg.hf_cache_dir / "checkpoints"
            )

            if latest_checkpoint.exists():
                logger.info(
                    f"Downloaded checkpoint from huggingface hub to {latest_checkpoint}"
                )
            return cfg.hf_cache_dir / "checkpoints" / "latest"

    except Exception as e:
        return None, repo_url
