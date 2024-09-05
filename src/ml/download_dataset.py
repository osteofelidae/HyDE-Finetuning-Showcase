import datasets
from src.utils.path_utils import (path, Path)

def download_dataset(
        dataset_id: str = "",
        output_dir: Path = path("data/downloads")
):
    with open("huggingface_token.private") as file:
        # Load dataset
        data = datasets.load_dataset(dataset_id, use_auth_token=file.read().strip()).filter(lambda e: e["lang"].lower() == "lua")

    # Save train and test split
    data["train"].to_json(output_dir/"raw.json")

if __name__ == "__main__":
    download_dataset(
        dataset_id="bigcode/the-stack",
        output_dir=path("data/lua")
    )