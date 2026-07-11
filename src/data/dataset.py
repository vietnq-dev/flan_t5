from __future__ import annotations

import logging
from typing import Any

from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset

logger = logging.getLogger(__name__)


def load_cot_collection(
    num_tasks: int | None = None,
    eval_ratio: float = 0.05,
    seed: int = 42,
    max_samples: int | None = None,
) -> DatasetDict:
    logger.info("Loading CoT-Collection dataset...")

    ds = load_dataset("kaist-ai/CoT-Collection", "en", split="train", streaming=True)

    if num_tasks is not None:
        logger.info("Discovering tasks...")
        seen: set[str] = set()
        for ex in ds:
            seen.add(ex["task"])
        rng = __import__("random").Random(seed)
        selected = set(rng.sample(sorted(seen), min(num_tasks, len(seen))))
        ds = load_dataset("kaist-ai/CoT-Collection", "en", split="train", streaming=True)
        ds = ds.filter(lambda x: x["task"] in selected)
        logger.info(f"Filtered to {num_tasks} tasks")

    if max_samples is not None:
        ds = ds.take(max_samples)
        logger.info(f"Limited to {max_samples} samples")

    dataset = Dataset.from_list(list(ds))
    logger.info(f"Loaded {len(dataset)} samples")

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


def load_datasets(config: dict[str, Any], max_samples: int | None = None) -> DatasetDict:
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
        max_samples=max_samples,
    )
