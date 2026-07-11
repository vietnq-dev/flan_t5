from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from src.utils.helpers import get_device
from src.utils.logging import print_results_table, save_metrics

logger = logging.getLogger(__name__)


def evaluate_checkpoint(
    checkpoint_path: str | Path,
    eval_dataset: Any,
    config: dict[str, Any],
    output_dir: str | Path,
    max_samples: int | None = None,
) -> dict[str, float]:
    checkpoint_path = Path(checkpoint_path)
    output_dir = Path(output_dir)

    logger.info(f"Loading model from {checkpoint_path}")
    model = AutoModelForSeq2SeqLM.from_pretrained(str(checkpoint_path))
    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint_path))

    device = torch.device(get_device())
    model = model.to(device)
    model.eval()

    if max_samples and max_samples < len(eval_dataset):
        eval_dataset = eval_dataset.select(range(max_samples))

    gen_config = config.get("generation", {})
    max_length = gen_config.get("max_length", 256)
    num_beams = gen_config.get("num_beams", 4)

    predictions = []
    references = []

    logger.info(f"Evaluating on {len(eval_dataset)} samples...")

    batch_size = config["training"].get("per_device_eval_batch_size", 4)
    for i in range(0, len(eval_dataset), batch_size):
        batch_end = min(i + batch_size, len(eval_dataset))
        batch = eval_dataset[i:batch_end]

        input_ids = torch.tensor(batch["input_ids"]).to(device)
        attention_mask = torch.tensor(batch["attention_mask"]).to(device)

        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=max_length,
                num_beams=num_beams,
                early_stopping=True,
            )

        for output_ids in outputs:
            predictions.append(tokenizer.decode(output_ids, skip_special_tokens=True).strip())

        for label_ids in batch["labels"]:
            ids = [int(x) for x in label_ids if x != -100]
            references.append(tokenizer.decode(ids, skip_special_tokens=True).strip())

    import numpy as np
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    rouge_scores = {"rouge1": [], "rouge2": [], "rougeL": []}
    exact_matches = 0

    for pred, ref in zip(predictions, references):
        if pred == ref:
            exact_matches += 1
        scores = scorer.score(ref, pred)
        rouge_scores["rouge1"].append(scores["rouge1"].fmeasure)
        rouge_scores["rouge2"].append(scores["rouge2"].fmeasure)
        rouge_scores["rougeL"].append(scores["rougeL"].fmeasure)

    total = len(predictions)
    results = {
        "exact_match": exact_matches / total if total > 0 else 0.0,
        "rouge1": float(np.mean(rouge_scores["rouge1"])),
        "rouge2": float(np.mean(rouge_scores["rouge2"])),
        "rougeL": float(np.mean(rouge_scores["rougeL"])),
        "num_samples": total,
    }

    print_results_table(results, title=f"Evaluation: {checkpoint_path.name}")
    save_metrics(results, output_dir, split="eval")

    return results
