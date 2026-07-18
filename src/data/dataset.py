from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Iterator

import ijson
from datasets import DatasetDict, concatenate_datasets, load_dataset

from src.utils.download import download_file

logger = logging.getLogger(__name__)


def _iter_records(filepath: str | Path) -> Iterator[dict[str, Any]]:
    with open(filepath, "rb") as f:
        for _, val in ijson.kvitems(f, ""):
            yield val


def _select_tasks(filepath: str | Path, num_tasks: int, seed: int) -> set[str]:
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

    cache_dir = Path("data") / "kaist-ai_CoT-Collection"
    json_path = cache_dir / "CoT_collection_en.json"

    if not (json_path.exists() and json_path.stat().st_size > 0):
        url = (
            "https://huggingface.co/datasets/kaist-ai/CoT-Collection"
            "/resolve/main/data/CoT_collection_en.json"
        )
        logger.info(f"Downloading CoT-Collection ({url})...")
        download_file(url, json_path)

    selected_tasks: set[str] | None = None
    if num_tasks is not None:
        logger.info("Discovering tasks (streaming pass)...")
        selected_tasks = _select_tasks(json_path, num_tasks, seed)
        logger.info(f"Filtered to {num_tasks} tasks")

    logger.info("Streaming records to temp file...")
    fd, tmp_path = tempfile.mkstemp(suffix=".jsonl")
    count = 0
    try:
        with os.fdopen(fd, "w") as f:
            for val in _iter_records(json_path):
                if selected_tasks is not None and val["task"] not in selected_tasks:
                    continue
                f.write(json.dumps(val) + "\n")
                count += 1
                if max_samples is not None and count >= max_samples:
                    break

        logger.info(f"Loaded {count} samples, building dataset...")
        dataset = load_dataset("json", data_files=tmp_path, split="train")
        logger.info(f"Dataset built: {len(dataset)} samples")
    finally:
        os.unlink(tmp_path)

    split = dataset.train_test_split(test_size=eval_ratio, seed=seed)
    logger.info(
        f"Split: {len(split['train'])} train, {len(split['test'])} eval"
    )
    return DatasetDict({"train": split["train"], "eval": split["test"]})


def load_sat_math_datasets(
    dataset_configs: list[dict[str, str]],
    eval_ratio: float = 0.1,
    seed: int = 42,
    max_samples: int | None = None,
) -> DatasetDict:
    datasets = []
    for cfg in dataset_configs:
        ds = load_dataset(cfg["name"], split=cfg.get("split", "train"))
        datasets.append(ds)

    combined = concatenate_datasets(datasets)
    if max_samples is not None and max_samples < len(combined):
        combined = combined.select(range(max_samples))
    split = combined.train_test_split(test_size=eval_ratio, seed=seed)
    return DatasetDict({"train": split["train"], "eval": split["test"]})


def load_datasets(config: dict[str, Any], max_samples: int | None = None) -> DatasetDict:
    data_config = config["data"]
    if max_samples is None:
        max_samples = data_config.get("max_samples")

    if data_config.get("datasets"):
        return load_sat_math_datasets(
            data_config["datasets"],
            eval_ratio=data_config.get("eval_ratio", 0.1),
            seed=data_config.get("seed", 42),
            max_samples=max_samples,
        )

    return load_cot_collection(
        num_tasks=data_config.get("num_tasks"),
        eval_ratio=data_config.get("eval_ratio", 0.05),
        seed=data_config.get("seed", 42),
        max_samples=max_samples,
    )
