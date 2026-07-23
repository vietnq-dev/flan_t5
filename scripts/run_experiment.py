import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logging import setup_logging

EXPERIMENTS = {
    "exp31": {
        "description": "Scaling number of tasks (CoT-Collection)",
        "configs": [
            "configs/exp31_scaling_tasks/flan_t5_small_100tasks.yaml",
            "configs/exp31_scaling_tasks/flan_t5_small_300tasks.yaml",
            "configs/exp31_scaling_tasks/flan_t5_small_600tasks.yaml",
            "configs/exp31_scaling_tasks/flan_t5_small_1060tasks.yaml",
        ],
    },
    "exp32": {
        "description": "Scaling model size (GSM8K / Elementary Math)",
        "configs": [
            "configs/exp32_scaling_model/flan_t5_small_gsm8k.yaml",
            "configs/exp32_scaling_model/flan_t5_base_gsm8k.yaml",
            "configs/exp32_scaling_model/flan_t5_small_sat_math.yaml",
            "configs/exp32_scaling_model/flan_t5_base_sat_math.yaml",
        ],
    },
    "exp33": {
        "description": "Chain-of-thought effect (CoT-Collection)",
        "configs": [
            "configs/exp33_cot/flan_t5_small_no_cot.yaml",
            "configs/exp33_cot/flan_t5_small_with_cot.yaml",
            "configs/exp33_cot/flan_t5_base_no_cot.yaml",
            "configs/exp33_cot/flan_t5_base_with_cot.yaml",
        ],
    },
}


def run_experiment(experiment_id: str, dry_run: bool = False) -> None:
    logger = setup_logging()

    if experiment_id not in EXPERIMENTS:
        logger.error(f"Unknown experiment: {experiment_id}")
        logger.info(f"Available experiments: {', '.join(EXPERIMENTS.keys())}")
        sys.exit(1)

    exp = EXPERIMENTS[experiment_id]
    logger.info(f"Running experiment: {experiment_id}")
    logger.info(f"Description: {exp['description']}")
    logger.info(f"Number of configs: {len(exp['configs'])}")

    for i, config_path in enumerate(exp["configs"], 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"Config {i}/{len(exp['configs'])}: {config_path}")
        logger.info(f"{'='*60}")

        cmd = [
            sys.executable,
            "scripts/train.py",
            "--config",
            config_path,
        ]

        if dry_run:
            logger.info(f"[DRY RUN] Would execute: {' '.join(cmd)}")
            continue

        result = subprocess.run(cmd, capture_output=False)

        if result.returncode != 0:
            logger.error(f"Config {config_path} failed with return code {result.returncode}")
            continue

        logger.info(f"Config {config_path} completed successfully")

    logger.info(f"\nExperiment {experiment_id} finished")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run batch experiments")
    parser.add_argument(
        "--experiment",
        type=str,
        required=False,
        choices=list(EXPERIMENTS.keys()) + ["all"],
        help="Experiment ID to run",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing")
    parser.add_argument("--list", action="store_true", help="List available experiments")
    args = parser.parse_args()

    if args.list:
        for exp_id, exp in EXPERIMENTS.items():
            print(f"  {exp_id}: {exp['description']} ({len(exp['configs'])} configs)")
        return

    if not args.experiment:
        parser.error("--experiment is required (unless using --list)")

    if args.experiment == "all":
        for exp_id in EXPERIMENTS:
            run_experiment(exp_id, dry_run=args.dry_run)
    else:
        run_experiment(args.experiment, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
