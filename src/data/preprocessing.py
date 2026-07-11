from __future__ import annotations

from typing import Any

from datasets import Dataset
from transformers import PreTrainedTokenizer


def preprocess_cot_example(example: dict[str, Any], use_cot: bool = True) -> dict[str, str]:
    source = example.get("source", example.get("input", example.get("question", "")))
    target = example.get("target", example.get("output", example.get("answer", "")))
    rationale = example.get("rationale", "")

    if use_cot and rationale:
        target = f"{rationale}\nThe answer is: {target}"

    return {"input_text": source, "target_text": target}


def preprocess_sat_math_example(example: dict[str, Any]) -> dict[str, str]:
    source = example.get("source", example.get("input", example.get("question", "")))
    target = example.get("target", example.get("output", example.get("answer", "")))
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
        input_texts = []
        target_texts = []

        batch_size = len(next(iter(examples.values())))
        for i in range(batch_size):
            ex = {k: v[i] for k, v in examples.items()}
            if is_cot:
                processed = preprocess_cot_example(ex, use_cot=use_cot)
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

    if "input_text" not in dataset.column_names:
        is_cot = "rationale" in dataset.column_names
        if is_cot:
            dataset = dataset.map(
                lambda ex: preprocess_cot_example(ex, use_cot=use_cot),
                num_proc=num_proc,
            )
        else:
            dataset = dataset.map(preprocess_sat_math_example, num_proc=num_proc)

    tokenized = dataset.map(
        preprocess_fn,
        batched=True,
        num_proc=num_proc,
        remove_columns=dataset.column_names,
    )

    return tokenized
