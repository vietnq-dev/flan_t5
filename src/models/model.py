from __future__ import annotations

import logging
from typing import Any

import torch
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

    tokenizer = AutoTokenizer.from_pretrained(
        model_config.get("tokenizer_name") or model_name,
        use_fast=True,
    )

    device = get_device()
    load_kwargs: dict[str, Any] = {
        "trust_remote_code": model_config.get("trust_remote_code", False),
    }

    if device == "cuda":
        load_kwargs["dtype"] = torch.float16
        load_kwargs["device_map"] = "auto"
    elif device == "mps":
        load_kwargs["dtype"] = torch.float32
    else:
        load_kwargs["dtype"] = torch.float32

    logger.info(f"Loading model on {device} with dtype={load_kwargs['dtype']}")
    
    try:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name, **load_kwargs)
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        logger.info("Retrying without device_map...")
        load_kwargs.pop("device_map", None)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name, **load_kwargs)

    if device == "mps":
        model = model.to(device)

    return model, tokenizer
