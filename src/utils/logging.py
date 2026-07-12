from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table


def setup_logging(
    output_dir: str | Path | None = None,
    level: int = logging.INFO,
    name: str = "flan_t5",
) -> logging.Logger:
    console = Console(stderr=True)
    handlers: list[logging.Handler] = [RichHandler(console=console, rich_tracebacks=True)]

    if output_dir:
        log_dir = Path(output_dir) / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_dir / "training.log")
        file_handler.setLevel(level)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        handlers.append(file_handler)

    logging.basicConfig(level=level, handlers=handlers, force=True)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    return logger


def log_config_summary(config: dict[str, Any], logger: logging.Logger) -> None:
    logger.info("=" * 60)
    logger.info("EXPERIMENT CONFIGURATION")
    logger.info("=" * 60)

    exp = config.get("experiment", {})
    logger.info(f"  Experiment ID:   {exp.get('id', 'N/A')}")
    logger.info(f"  Experiment Name: {exp.get('name', 'N/A')}")
    logger.info(f"  Description:     {exp.get('description', 'N/A')}")

    model = config.get("model", {})
    logger.info(f"  Model:           {model.get('name_or_path', 'N/A')}")

    data = config.get("data", {})
    logger.info(f"  Dataset:         {data.get('dataset_name', 'N/A')}")
    logger.info(f"  Num Tasks:       {data.get('num_tasks', 'all')}")
    logger.info(f"  Use CoT:         {data.get('use_cot', True)}")

    training = config.get("training", {})
    logger.info(f"  Epochs:          {training.get('num_train_epochs', 3)}")
    logger.info(f"  Batch Size:      {training.get('per_device_train_batch_size', 4)}")
    logger.info(f"  Grad Accum:      {training.get('gradient_accumulation_steps', 4)}")
    logger.info(f"  Learning Rate:   {training.get('learning_rate', 5e-5)}")

    logger.info("=" * 60)


def save_config(config: dict[str, Any], output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "config.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def save_metrics(metrics: dict[str, Any], output_dir: str | Path, split: str = "eval") -> None:
    output_dir = Path(output_dir)
    results_dir = output_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / f"{split}_results.json", "w") as f:
        json.dump(metrics, f, indent=2)


def print_results_table(results: dict[str, float], title: str = "Results") -> None:
    console = Console()
    table = Table(title=title)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green", justify="right")

    for key, value in results.items():
        if isinstance(value, float):
            table.add_row(key, f"{value:.4f}")
        else:
            table.add_row(key, str(value))

    console.print(table)


def generate_run_name(config: dict[str, Any]) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    exp_name = config["experiment"]["name"]
    lr = config["training"]["learning_rate"]
    bs = config["training"]["per_device_train_batch_size"]
    ga = config["training"]["gradient_accumulation_steps"]

    return f"{exp_name}_lr{lr:.0e}_bs{bs}x{ga}_{timestamp}"


def get_output_dir(config: dict[str, Any], base_dir: str = "outputs") -> str:
    model_name = config["model"]["name_or_path"].split("/")[-1]
    run_name = generate_run_name(config)
    return str(Path(base_dir) / model_name / run_name)
