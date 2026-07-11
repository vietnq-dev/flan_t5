from __future__ import annotations

import logging
from typing import Any, Iterator

import ijson
from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset
from huggingface_hub import hf_hub_download

logger = logging.getLogger(__name__)


def _iter_records(filepath: str) -> Iterator[dict[str, Any]]:
    """Yield each record from a JSON dict ``{id: record, ...}``
    without loading the whole file into memory."""
    with open(filepath, "rb") as f:
        for _, val in ijson.kvitems(f, ""):
            yield val


def _select_tasks(filepath: str, num_tasks: int, seed: int) -> set[str]:
    """First pass: stream through JSON to collect all unique task names."""
    seen: set[str] = set()
    for val in _iter_records(filepath):
        seen.add(val["task"])
    rng = __import__("random").Random(seed)
    return set(rng.sample(sorted(seen), min(num_tasks, len(seen))))


def load_cot_collection(
    num_tasks: int | None = None,
    eval_ratio: float = 0.05,
    seed: int = 42,
    max_samples: int | None = None,
) -> DatasetDict:
    logger.info("Loading CoT-Collection dataset...")

    json_path = hf_hub_download(
        repo_id="kaist-ai/CoT-Collection",
        filename="data/CoT_collection_en.json",
        repo_type="dataset",
    )

    selected_tasks: set[str] | None = None
    if num_tasks is not None:
        logger.info("Discovering tasks (streaming pass)...")
        selected_tasks = _select_tasks(json_path, num_tasks, seed)
        logger.info(f"Filtered to {num_tasks} tasks")

    logger.info("Streaming records...")
    records: list[dict[str, Any]] = []
    for val in _iter_records(json_path):
        if selected_tasks is not None and val["task"] not in selected_tasks:
            continue
        records.append(val)
        if max_samples is not None and len(records) >= max_samples:
            break

    logger.info(f"Loaded {len(records)} samples")
    dataset = Dataset.from_list(records)
    del records

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
