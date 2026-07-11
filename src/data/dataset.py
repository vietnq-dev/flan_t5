from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset
from huggingface_hub import hf_hub_download

logger = logging.getLogger(__name__)


def load_cot_collection(
    num_tasks: int | None = None,
    eval_ratio: float = 0.05,
    seed: int = 42,
) -> DatasetDict:
    logger.info("Loading CoT-Collection dataset...")
    
    json_path = hf_hub_download(
        repo_id="kaist-ai/CoT-Collection",
        filename="data/CoT_collection_en.json",
        repo_type="dataset",
    )
    
    with open(json_path, "r") as f:
        data = json.load(f)
    
    dataset = Dataset.from_list(list(data.values()))
    logger.info(f"Loaded {len(dataset)} samples from JSON")
    
    all_tasks = sorted(set(dataset["task"]))
    if num_tasks is not None and num_tasks < len(all_tasks):
        rng = __import__("random").Random(seed)
        selected_tasks = rng.sample(all_tasks, num_tasks)
        dataset = dataset.filter(lambda x: x["task"] in selected_tasks)
    
    split = dataset.train_test_split(test_size=eval_ratio, seed=seed)
    return DatasetDict({"train": split["train"], "eval": split["test"]})


def load_sat_math_datasets(
    dataset_configs: list[dict[str, str]],
    eval_ratio: float = 0.1,
    seed: int = 42,
) -> DatasetDict:
    datasets = []
    for cfg in dataset_configs:
        ds = load_dataset(cfg["name"], split=cfg.get("split", "train"))
        datasets.append(ds)

    combined = concatenate_datasets(datasets)
    split = combined.train_test_split(test_size=eval_ratio, seed=seed)
    return DatasetDict({"train": split["train"], "eval": split["test"]})


def load_datasets(config: dict[str, Any]) -> DatasetDict:
    data_config = config["data"]

    if data_config.get("datasets"):
        return load_sat_math_datasets(
            data_config["datasets"],
            eval_ratio=data_config.get("eval_ratio", 0.1),
            seed=data_config.get("seed", 42),
        )

    return load_cot_collection(
        num_tasks=data_config.get("num_tasks"),
        eval_ratio=data_config.get("eval_ratio", 0.05),
        seed=data_config.get("seed", 42),
    )
