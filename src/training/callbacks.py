from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

from transformers import TrainerCallback, TrainerControl, TrainerState

logger = logging.getLogger(__name__)


class ExperimentCallback(TrainerCallback):
    def __init__(self, config: dict[str, Any], output_dir: str) -> None:
        self._config = config
        self._output_dir = Path(output_dir)
        self._start_time: float = 0.0

    def on_train_begin(
        self,
        args: Any,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        self._start_time = time.time()
        self._output_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Experiment run started")
        logger.info(f"Output directory: {self._output_dir}")

    def on_train_end(
        self,
        args: Any,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        elapsed = time.time() - self._start_time
        hours = int(elapsed // 3600)
        minutes = int((elapsed % 3600) // 60)
        seconds = int(elapsed % 60)
        logger.info(f"Training completed in {hours}h {minutes}m {seconds}s")

    def on_evaluate(
        self,
        args: Any,
        state: TrainerState,
        control: TrainerControl,
        metrics: dict[str, float] | None = None,
        **kwargs: Any,
    ) -> None:
        if metrics is None:
            return

        results_dir = self._output_dir / "results"
        results_dir.mkdir(parents=True, exist_ok=True)

        step = state.global_step
        results_file = results_dir / f"eval_results_step{step}.json"
        with open(results_file, "w") as f:
            json.dump(metrics, f, indent=2)

        logger.info(f"Step {step} eval results saved to {results_file}")

    def on_save(
        self,
        args: Any,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        logger.info(f"Checkpoint saved at step {state.global_step}")
