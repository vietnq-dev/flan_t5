from __future__ import annotations

from typing import Any

import numpy as np
from rouge_score import rouge_scorer


def compute_metrics(eval_preds: Any) -> dict[str, float]:
    preds, labels = eval_preds

    if isinstance(preds, tuple):
        preds = preds[0]

    decoded_preds = _decode_tokens(preds)
    decoded_labels = _decode_tokens(labels)

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    rouge_scores = {"rouge1": [], "rouge2": [], "rougeL": []}

    exact_matches = 0
    total = len(decoded_preds)

    for pred, label in zip(decoded_preds, decoded_labels):
        pred = pred.strip()
        label = label.strip()

        if pred == label:
            exact_matches += 1

        scores = scorer.score(label, pred)
        rouge_scores["rouge1"].append(scores["rouge1"].fmeasure)
        rouge_scores["rouge2"].append(scores["rouge2"].fmeasure)
        rouge_scores["rougeL"].append(scores["rougeL"].fmeasure)

    results = {
        "exact_match": exact_matches / total if total > 0 else 0.0,
        "rouge1": np.mean(rouge_scores["rouge1"]),
        "rouge2": np.mean(rouge_scores["rouge2"]),
        "rougeL": np.mean(rouge_scores["rougeL"]),
    }

    return {k: float(v) for k, v in results.items()}


def _decode_tokens(token_ids: np.ndarray) -> list[str]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")

    decoded = []
    for ids in token_ids:
        ids = [int(i) for i in ids if i != -100]
        text = tokenizer.decode(ids, skip_special_tokens=True)
        decoded.append(text)
    return decoded


def compute_metrics_with_tokenizer(tokenizer: Any):
    def _compute(eval_preds: Any) -> dict[str, float]:
        preds, labels = eval_preds

        if isinstance(preds, tuple):
            preds = preds[0]

        decoded_preds = []
        decoded_labels = []

        for pred_ids in preds:
            ids = [int(i) for i in pred_ids if i != -100]
            decoded_preds.append(tokenizer.decode(ids, skip_special_tokens=True).strip())

        for label_ids in labels:
            ids = [int(i) for i in label_ids if i != -100]
            decoded_labels.append(tokenizer.decode(ids, skip_special_tokens=True).strip())

        scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
        rouge_scores = {"rouge1": [], "rouge2": [], "rougeL": []}

        exact_matches = 0
        total = len(decoded_preds)

        for pred, label in zip(decoded_preds, decoded_labels):
            if pred == label:
                exact_matches += 1

            scores = scorer.score(label, pred)
            rouge_scores["rouge1"].append(scores["rouge1"].fmeasure)
            rouge_scores["rouge2"].append(scores["rouge2"].fmeasure)
            rouge_scores["rougeL"].append(scores["rougeL"].fmeasure)

        results = {
            "exact_match": exact_matches / total if total > 0 else 0.0,
            "rouge1": float(np.mean(rouge_scores["rouge1"])),
            "rouge2": float(np.mean(rouge_scores["rouge2"])),
            "rougeL": float(np.mean(rouge_scores["rougeL"])),
        }

        return results

    return _compute
