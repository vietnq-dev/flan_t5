from __future__ import annotations

import importlib
import random
from typing import Any


def set_seed(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def model_short_name(model_path: str) -> str:
    return model_path.split("/")[-1]


def format_number(n: int | float) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


def count_parameters(model: Any) -> dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


def is_package_available(package_name: str) -> bool:
    try:
        importlib.import_module(package_name)
        return True
    except ImportError:
        return False


def flatten_dict(d: dict[str, Any], parent_key: str = "", sep: str = ".") -> dict[str, Any]:
    items: list[tuple[str, Any]] = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def get_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def get_device_info() -> dict[str, Any]:
    import torch

    device = get_device()
    info = {"device": device}

    if device == "cuda":
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_memory_gb"] = torch.cuda.get_device_properties(0).total_memory / 1e9
        info["cuda_version"] = torch.version.cuda
    else:
        info["backend"] = "CPU"

    return info


def get_precision_settings(device: str) -> dict[str, bool]:
    if device == "cuda":
        import torch

        capability = torch.cuda.get_device_capability()[0]
        if capability >= 8:
            return {"fp16": True, "bf16": True, "tf32": True}
        return {"fp16": True, "bf16": False, "tf32": False}
    return {"fp16": False, "bf16": False, "tf32": False}
