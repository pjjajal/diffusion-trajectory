from datasets import load_dataset
from omegaconf import DictConfig
import os


MODEL_NAME_TO_ID = {
    "drawbench": ("sayakpaul/drawbench", "train", ["Prompts"]),
    "open_image_preferences_36": (
        "csv",
        os.path.join(
            os.getcwd(), "eval_datasets/open_img_pref/open_img_pref_sampled_36.csv"
        ),
        ["prompt"],
    ),
    "open_image_preferences_60": (
        "csv",
        os.path.join(
            os.getcwd(), "eval_datasets/open_img_pref/open_img_pref_sampled_60.csv"
        ),
        ["prompt"],
    ),
}


def create_dataset(dataset_cfg: DictConfig):
    try:
        model_id, split, columns = MODEL_NAME_TO_ID[dataset_cfg.name]
        if model_id == "csv":
            dataset = load_dataset(model_id, data_files=split, split="train")
        else:
            dataset = load_dataset(
                model_id, split=split, cache_dir=dataset_cfg.cache_dir
            )
        dataset = dataset.select_columns(columns)
        if "prompt" not in dataset.column_names:
            dataset = dataset.rename_column(columns[0], "prompt")
    except KeyError:
        raise ValueError(f"Dataset {dataset_cfg.name} not supported.")
    return dataset
