from datasets import load_dataset
from omegaconf import DictConfig

MODEL_NAME_TO_ID = {
    "drawbench": ("sayakpaul/drawbench", "train", ["Prompts"]),
}


def create_dataset(dataset_cfg: DictConfig):
    try:
        model_id, split, columns = MODEL_NAME_TO_ID[dataset_cfg.name]
        dataset = load_dataset(model_id, split=split, cache_dir=dataset_cfg.cache_dir)
        dataset = dataset.select_columns(columns).rename_column(columns[0], "prompt")
    except KeyError:
        raise ValueError(f"Dataset {dataset_cfg.name} not supported.")
    return dataset
