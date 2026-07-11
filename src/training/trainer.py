from __future__ import annotations

import logging
from typing import Any

from transformers import (
    DataCollatorForSeq2Seq,
    PreTrainedModel,
    PreTrainedTokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

from src.training.callbacks import ExperimentCallback
from src.utils.helpers import get_device, get_precision_settings

logger = logging.getLogger(__name__)


def build_training_args(config: dict[str, Any], output_dir: str) -> Seq2SeqTrainingArguments:
    tc = config["training"]
    pc = config.get("precision", {})
    lc = config.get("logging", {})
    sc = config.get("saving", {})
    ec = config.get("evaluation", {})
    ac = config.get("advanced", {})

    device = get_device()
    auto_precision = get_precision_settings(device)

    fp16 = auto_precision["fp16"] if pc.get("fp16") is None else pc["fp16"]
    bf16 = auto_precision["bf16"] if pc.get("bf16") is None else pc["bf16"]
    tf32 = auto_precision["tf32"] if pc.get("tf32") is None else pc["tf32"]

    if device == "mps":
        ac["dataloader_num_workers"] = 0

    logger.info(f"Device: {device}")
    logger.info(f"Precision: fp16={fp16}, bf16={bf16}, tf32={tf32}")

    return Seq2SeqTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=tc.get("num_train_epochs", 3),
        per_device_train_batch_size=tc.get("per_device_train_batch_size", 4),
        per_device_eval_batch_size=tc.get("per_device_eval_batch_size", 4),
        gradient_accumulation_steps=tc.get("gradient_accumulation_steps", 4),
        learning_rate=tc.get("learning_rate", 5e-5),
        weight_decay=tc.get("weight_decay", 0.01),
        adam_beta1=tc.get("adam_beta1", 0.9),
        adam_beta2=tc.get("adam_beta2", 0.999),
        adam_epsilon=tc.get("adam_epsilon", 1e-8),
        max_grad_norm=tc.get("max_grad_norm", 1.0),
        warmup_ratio=tc.get("warmup_ratio", 0.06),
        warmup_steps=tc.get("warmup_steps", 0),
        lr_scheduler_type=tc.get("lr_scheduler_type", "linear"),
        optim=tc.get("optim", "adamw_torch"),
        max_steps=tc.get("max_steps", -1),
        fp16=fp16,
        bf16=bf16,
        tf32=tf32,
        logging_steps=lc.get("logging_steps", 50),
        logging_first_step=lc.get("logging_first_step", True),
        logging_dir=f"{output_dir}/logs",
        report_to=lc.get("report_to", ["tensorboard"]),
        save_strategy=sc.get("save_strategy", "epoch"),
        save_steps=sc.get("save_steps", 500),
        save_total_limit=sc.get("save_total_limit", 2),
        save_only_model=sc.get("save_only_model", False),
        eval_strategy=ec.get("evaluation_strategy", "epoch"),
        eval_steps=ec.get("eval_steps", 500),
        metric_for_best_model=ec.get("metric_for_best_model", "eval_loss"),
        greater_is_better=ec.get("greater_is_better", False),
        load_best_model_at_end=ec.get("load_best_model_at_end", True),
        predict_with_generate=ec.get("predict_with_generate", True),
        seed=ac.get("seed", 42),
        dataloader_num_workers=ac.get("dataloader_num_workers", 0 if device == "mps" else 2),
        gradient_checkpointing=ac.get("gradient_checkpointing", False),
        push_to_hub=ac.get("push_to_hub", False),
        hub_model_id=ac.get("hub_model_id"),
        resume_from_checkpoint=ac.get("resume_from_checkpoint"),
        generation_max_length=config.get("generation", {}).get("max_length", 256),
        generation_num_beams=config.get("generation", {}).get("num_beams", 4),
    )


def create_trainer(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    train_dataset: Any,
    eval_dataset: Any,
    config: dict[str, Any],
    output_dir: str,
    compute_metrics_fn: Any = None,
) -> Seq2SeqTrainer:
    training_args = build_training_args(config, output_dir)

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        pad_to_multiple_of=8,
    )

    callbacks = [ExperimentCallback(config, output_dir)]

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics_fn,
        callbacks=callbacks,
    )

    return trainer
