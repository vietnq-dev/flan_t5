import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.dataset import load_datasets
from src.data.preprocessing import tokenize_dataset
from src.evaluation.metrics import compute_metrics_with_tokenizer
from src.models.model import load_model_and_tokenizer
from src.training.trainer import create_trainer
from src.utils.config import load_config
from src.utils.helpers import count_parameters, get_device_info, set_seed
from src.utils.logging import (
    get_output_dir,
    log_config_summary,
    print_results_table,
    save_config,
    save_metrics,
    setup_logging,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Flan-T5 model")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--base-config", type=str, default=None, help="Path to base config YAML")
    parser.add_argument("--output-dir", type=str, default=None, help="Override output directory")
    parser.add_argument(
        "--max-train-samples", type=int, default=None, help="Limit training samples"
    )
    parser.add_argument("--max-eval-samples", type=int, default=None, help="Limit eval samples")
    parser.add_argument(
        "--max-samples", type=int, default=None, help="Limit total samples before split (streaming)"
    )
    args = parser.parse_args()

    config = load_config(args.config, args.base_config)
    set_seed(config.get("advanced", {}).get("seed", 42))

    output_dir = args.output_dir or get_output_dir(config)
    logger = setup_logging(output_dir=output_dir)

    device_info = get_device_info()
    logger.info(f"Device: {device_info['device']}")
    if "gpu_name" in device_info:
        logger.info(f"GPU: {device_info['gpu_name']} ({device_info['gpu_memory_gb']:.1f} GB)")

    log_config_summary(config, logger)
    save_config(config, output_dir)

    logger.info("Loading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer(config)

    param_counts = count_parameters(model)
    total = param_counts["total"]
    trainable = param_counts["trainable"]
    logger.info(f"Model parameters: {total:,} total, {trainable:,} trainable")

    logger.info("Loading datasets...")
    datasets = load_datasets(config, max_samples=args.max_samples)
    train_dataset = datasets["train"]
    eval_dataset = datasets["eval"]

    logger.info(f"Train samples: {len(train_dataset):,}")
    logger.info(f"Eval samples: {len(eval_dataset):,}")

    if args.max_train_samples:
        train_dataset = train_dataset.select(range(min(args.max_train_samples, len(train_dataset))))
    if args.max_eval_samples:
        eval_dataset = eval_dataset.select(range(min(args.max_eval_samples, len(eval_dataset))))

    logger.info("Tokenizing datasets...")
    data_config = config["data"]
    num_proc = data_config.get("preprocessing_num_workers", 4)

    train_dataset = tokenize_dataset(
        train_dataset,
        tokenizer,
        max_source_length=data_config.get("max_source_length", 512),
        max_target_length=data_config.get("max_target_length", 256),
        use_cot=data_config.get("use_cot", True),
        num_proc=num_proc,
    )
    eval_dataset = tokenize_dataset(
        eval_dataset,
        tokenizer,
        max_source_length=data_config.get("max_source_length", 512),
        max_target_length=data_config.get("max_target_length", 256),
        use_cot=data_config.get("use_cot", True),
        num_proc=num_proc,
    )

    logger.info("Creating trainer...")
    trainer = create_trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        config=config,
        output_dir=output_dir,
        compute_metrics_fn=compute_metrics_with_tokenizer(tokenizer),
    )

    logger.info("Starting training...")
    resume = config.get("advanced", {}).get("resume_from_checkpoint")
    logger.info("Calling trainer.train()...")
    train_result = trainer.train(resume_from_checkpoint=resume)

    logger.info("Saving final model...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    save_metrics(metrics, output_dir, split="train")

    logger.info("Running final evaluation...")
    eval_result = trainer.evaluate()
    trainer.log_metrics("eval", eval_result)
    trainer.save_metrics("eval", eval_result)
    save_metrics(eval_result, output_dir, split="eval")

    print_results_table(eval_result, title="Final Evaluation Results")
    logger.info(f"Training complete. Outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
