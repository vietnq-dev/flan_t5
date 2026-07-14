from __future__ import annotations

import logging
from typing import Any

from huggingface_hub import snapshot_download
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)

from src.utils.helpers import get_device

logger = logging.getLogger(__name__)


def load_model_and_tokenizer(
    config: dict[str, Any],
) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
    model_config = config["model"]
    model_name = model_config["name_or_path"]

    tokenizer_name = model_config.get("tokenizer_name") or model_name
    logger.info(f"Downloading tokenizer from {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        use_fast=True,
    )

    local_path = snapshot_download(
        repo_id=model_name,
        resume_download=True,
        max_workers=4,
    )
    logger.info(f"Model cached at {local_path}")

    device = get_device()
    load_kwargs: dict[str, Any] = {
        "trust_remote_code": model_config.get("trust_remote_code", False),
    }

    logger.info(f"Loading model on {device} (fp32 weights; AMP handles fp16 via Trainer)")
    model = AutoModelForSeq2SeqLM.from_pretrained(local_path, **load_kwargs)

    return model, tokenizer
