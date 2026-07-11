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
    parser.add_argument("--max-samples", type=int, default=None, help="Limit eval samples")
    args = parser.parse_args()

    config = load_config(args.config, args.base_config)
    set_seed(config.get("advanced", {}).get("seed", 42))

    output_dir = args.output_dir or get_output_dir(config)
    logger = setup_logging(output_dir=output_dir)

    logger.info("Loading datasets...")
    datasets = load_datasets(config)
    eval_dataset = datasets["eval"]

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
        max_samples=args.max_samples,
    )

    print_results_table(results, title="Evaluation Results")
    save_metrics(results, output_dir, split="eval")

    logger.info(f"Evaluation complete. Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
