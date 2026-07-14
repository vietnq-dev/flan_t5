from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)

from src.utils.download import download_file
from src.utils.helpers import get_device

logger = logging.getLogger(__name__)

_MODEL_FILES = [
    "config.json",
    "tokenizer_config.json",
    "spiece.model",
    "tokenizer.json",
    "special_tokens_map.json",
    "model.safetensors",
]


def _download_model_repo(model_name: str, dest_dir: Path) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    base_url = f"https://huggingface.co/{model_name}/resolve/main"

    for filename in _MODEL_FILES:
        dest = dest_dir / filename
        if dest.exists() and dest.stat().st_size > 0:
            logger.info(f"  ✓ {filename} (cached, {dest.stat().st_size / 1e6:.1f} MB)")
            continue
        download_file(f"{base_url}/{filename}", dest)

    return dest_dir


def load_model_and_tokenizer(
    config: dict[str, Any],
) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
    model_config = config["model"]
    model_name = model_config["name_or_path"]

    cache_dir = Path("models") / model_name.replace("/", "_")
    logger.info(f"Downloading model {model_name} to {cache_dir}")
    local_path = _download_model_repo(model_name, cache_dir)

    tokenizer_name = model_config.get("tokenizer_name") or str(local_path)
    logger.info(f"Loading tokenizer from {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name if model_config.get("tokenizer_name") else local_path,
        use_fast=True,
    )

    device = get_device()
    load_kwargs: dict[str, Any] = {
        "trust_remote_code": model_config.get("trust_remote_code", False),
    }

    logger.info(f"Loading model on {device} (fp32 weights; AMP handles fp16 via Trainer)")
    model = AutoModelForSeq2SeqLM.from_pretrained(local_path, **load_kwargs)

    return model, tokenizer
