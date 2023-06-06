import requests
from huggingface_hub import HfApi, Repository

if __name__ == "__main__":
    import os

    from rich import print

    huggingface_username = os.environ.get("HF_USERNAME")
    huggingface_api_token = os.environ.get("HF_TOKEN")

    client = HfApi(token=huggingface_api_token)

    for model_repo in client.list_models(author=huggingface_username):
        print(f"Setting {model_repo} to private")
        client.update_repo_visibility(
            repo_id=model_repo.__dict__["modelId"], private=True
        )
