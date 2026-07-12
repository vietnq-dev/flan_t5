import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.dataset import load_datasets
from src.data.preprocessing import tokenize_dataset
from src.evaluation.evaluate import evaluate_checkpoint
from src.utils.config import load_config
from src.utils.helpers import set_seed
from src.utils.logging import (
    get_output_dir,
    print_results_table,
    save_metrics,
    setup_logging,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Flan-T5 model")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--base-config", type=str, default=None, help="Path to base config YAML")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for results")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit total samples before split (streaming); caps dataset loading at the source.",
    )
    parser.add_argument(
        "--max-eval-samples",
        type=int,
        default=None,
        help="Limit eval samples after split (fast smoke testing).",
    )
    parser.add_argument(
        "--num-beams",
        type=int,
        default=None,
        help="Override generation num_beams (e.g. 1 for greedy ~4x faster ROUGE).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override per-device eval batch size.",
    )
    args = parser.parse_args()

    config = load_config(args.config, args.base_config)
    set_seed(config.get("advanced", {}).get("seed", 42))

    if args.num_beams is not None:
        config.setdefault("generation", {})["num_beams"] = args.num_beams
    if args.batch_size is not None:
        config.setdefault("training", {})["per_device_eval_batch_size"] = args.batch_size

    output_dir = args.output_dir or get_output_dir(config)
    logger = setup_logging(output_dir=output_dir)

    # Derive a source cap so we never stream the full 1.8M dataset just to eval
    # a small slice. If only --max-eval-samples is given, estimate the total
    # needed from the eval ratio (with a small buffer).
    import math

    eval_ratio = config.get("data", {}).get("eval_ratio", 0.05)
    if args.max_samples is not None:
        effective_max_samples = args.max_samples
    elif args.max_eval_samples is not None:
        effective_max_samples = math.ceil(args.max_eval_samples / max(eval_ratio, 1e-3)) * 2
    else:
        effective_max_samples = None

    logger.info("Loading datasets...")
    if effective_max_samples is not None:
        logger.info(f"Limiting dataset loading to {effective_max_samples} samples (source cap)")
    datasets = load_datasets(config, max_samples=effective_max_samples)
    eval_dataset = datasets["eval"]

    if args.max_eval_samples and args.max_eval_samples < len(eval_dataset):
        logger.info(
            f"Selecting first {args.max_eval_samples} eval samples from {len(eval_dataset)}"
        )
        eval_dataset = eval_dataset.select(range(args.max_eval_samples))

    data_config = config["data"]
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)

    eval_dataset = tokenize_dataset(
        eval_dataset,
        tokenizer,
        max_source_length=data_config.get("max_source_length", 512),
        max_target_length=data_config.get("max_target_length", 256),
        use_cot=data_config.get("use_cot", True),
        num_proc=data_config.get("preprocessing_num_workers", 4),
    )

    results = evaluate_checkpoint(
        checkpoint_path=args.checkpoint,
        eval_dataset=eval_dataset,
        config=config,
        output_dir=output_dir,
        max_samples=None,  # slicing already handled above to keep log counts accurate
    )

    print_results_table(results, title="Evaluation Results")
    save_metrics(results, output_dir, split="eval")

    logger.info(f"Evaluation complete. Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
