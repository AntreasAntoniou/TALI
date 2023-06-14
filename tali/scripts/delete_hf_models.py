import requests
from huggingface_hub import HfApi, Repository

from tali.utils import get_logger

logger = get_logger(__name__, set_rich=True)

if __name__ == "__main__":
    import os

    huggingface_username = os.environ.get("HF_USERNAME")
    huggingface_api_token = os.environ.get("HF_TOKEN")

    client = HfApi(token=huggingface_api_token)

    for model_repo in client.list_models(author=huggingface_username):
        if "debug" in model_repo.__dict__["modelId"]:
            logger.info(f"Deleting {model_repo}")
            client.delete_repo(repo_id=model_repo.__dict__["modelId"])
