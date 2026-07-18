from __future__ import annotations

from typing import Any

from datasets import Dataset
from transformers import PreTrainedTokenizer


def _first_value(example: dict[str, Any], keys: list[str]) -> str:
    for key in keys:
        val = example.get(key)
        if val is not None:
            return str(val)
    return ""


def preprocess_cot_example(example: dict[str, Any], use_cot: bool = True) -> dict[str, str]:
    source = _first_value(example, ["source", "input", "question"])
    target = _first_value(example, ["target", "output", "answer"])
    rationale = example.get("rationale", "")

    if use_cot and rationale:
        target = f"{rationale}\nThe answer is: {target}"

    return {"input_text": source, "target_text": target}


def preprocess_sat_math_example(example: dict[str, Any]) -> dict[str, str]:
    source = _first_value(example, ["source", "input", "question", "text"])
    target = _first_value(example, ["target", "output", "answer", "solution"])
    return {"input_text": source, "target_text": target}


def tokenize_dataset(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizer,
    max_source_length: int = 512,
    max_target_length: int = 256,
    use_cot: bool = True,
    num_proc: int = 4,
) -> Dataset:
    def preprocess_fn(examples: dict[str, list]) -> dict[str, list]:
        is_cot = "rationale" in examples
        already_preprocessed = "input_text" in examples and "target_text" in examples
        input_texts = []
        target_texts = []

        batch_size = len(next(iter(examples.values())))
        for i in range(batch_size):
            ex = {k: v[i] for k, v in examples.items()}
            if already_preprocessed:
                input_texts.append(ex["input_text"])
                target_texts.append(ex["target_text"])
            elif is_cot:
                processed = preprocess_cot_example(ex, use_cot=use_cot)
                input_texts.append(processed["input_text"])
                target_texts.append(processed["target_text"])
            else:
                processed = preprocess_sat_math_example(ex)
                input_texts.append(processed["input_text"])
                target_texts.append(processed["target_text"])

        model_inputs = tokenizer(
            input_texts,
            max_length=max_source_length,
            padding="max_length",
            truncation=True,
        )
        labels = tokenizer(
            target_texts,
            max_length=max_target_length,
            padding="max_length",
            truncation=True,
        )

        label_ids = labels["input_ids"]
        label_ids = [
            [(lid if lid != tokenizer.pad_token_id else -100) for lid in label]
            for label in label_ids
        ]

        model_inputs["labels"] = label_ids
        return model_inputs

    tokenized = dataset.map(
        preprocess_fn,
        batched=True,
        num_proc=num_proc,
        remove_columns=dataset.column_names,
    )

    return tokenized
