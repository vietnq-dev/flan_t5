# Flan-T5 Instruction Fine-Tuning Experiments

Replicate small-scale experiments from the paper [Scaling Instruction-Finetuned Language Models](https://arxiv.org/abs/2210.11416) (Chung et al., 2022) using Flan-T5-Small (77M) and Flan-T5-Base (248M).

## Setup

```bash
uv sync --all-groups
```

Verify installation:

```bash
uv run python -c "import torch; print(torch.__version__); print('MPS:', torch.backends.mps.is_available()); print('CUDA:', torch.cuda.is_available())"
```

## Device Support

Device detection is automatic. No config changes needed.

| Device | Precision | Notes |
|--------|-----------|-------|
| **NVIDIA T4** (Colab) | fp16 | 16GB VRAM. Use `batch_size=4`, `grad_accum=4` |
| **Apple Silicon** (M1/M2/M3/M4) | bf16 | MPS backend. `dataloader_num_workers` auto-set to 0 |
| **CPU** | fp32 | Slow but works. Reduce `max_source_length` to 256 for speed |

Override precision manually if needed:

```bash
uv run python scripts/train.py --config configs/exp33_cot/flan_t5_small_with_cot.yaml
```

The config `precision` fields default to `null` (auto-detect). Set them explicitly to override:

```yaml
precision:
  fp16: true   # force fp16 on T4
  bf16: false
  tf32: false
```

## Experiment Guide

Run experiments in this order, from quickest to most expensive:

### Step 1: Quick sanity check (~10 min on T4, ~1h on MPS)

Verify the pipeline works end-to-end with a tiny subset:

```bash
uv run python scripts/train.py \
  --config configs/exp33_cot/flan_t5_small_with_cot.yaml \
  --max-train-samples 100 \
  --max-eval-samples 20
```

This downloads the model, tokenizes 100 samples, trains for 3 epochs, and evaluates. If this works, everything is set up correctly.

### Step 2: Experiment 3.3 - Chain-of-Thought effect (~2-4h on T4, ~8-16h on MPS)

Compare training with vs without CoT rationales. This is the most impactful experiment and uses the smallest dataset.

```bash
uv run scripts/run_experiment.py --experiment exp33
```

Or run individually:

```bash
uv run python scripts/train.py --config configs/exp33_cot/flan_t5_small_with_cot.yaml
uv run python scripts/train.py --config configs/exp33_cot/flan_t5_small_no_cot.yaml
uv run python scripts/train.py --config configs/exp33_cot/flan_t5_base_with_cot.yaml
uv run python scripts/train.py --config configs/exp33_cot/flan_t5_base_no_cot.yaml
```

**What to look for:** Models trained with CoT should show better ROUGE-L scores. Models without CoT should degrade on reasoning tasks.

### Step 3: Experiment 3.2 - Scaling model size (~1-2h on T4, ~4-8h on MPS)

Train small and base on SAT Reading + Elementary Math (smaller datasets).

```bash
uv run scripts/run_experiment.py --experiment exp32
```

**What to look for:** Flan-T5-Base should outperform Flan-T5-Small on the same data, demonstrating that instruction tuning benefits scale with model size.

### Step 4: Experiment 3.1 - Scaling number of tasks (~4-8h on T4, ~16-32h on MPS)

The largest experiment. Trains on 100, 300, 600, and 1060 tasks from CoT-Collection.

```bash
uv run scripts/run_experiment.py --experiment exp31
```

**What to look for:** Performance should improve with more tasks, but with diminishing returns after ~300 tasks (matching the paper's finding of diminishing returns after 282 tasks).

### Run everything

```bash
uv run scripts/run_experiment.py --experiment all
```

## Project Structure

```
flan_t5/
├── configs/
│   ├── base.yaml                          # Shared defaults (auto-precision)
│   ├── exp31_scaling_tasks/               # 4 configs: 100/300/600/1060 tasks
│   ├── exp32_scaling_model/               # 2 configs: small + base on SAT/math
│   └── exp33_cot/                         # 4 configs: with/without CoT x small/base
├── src/
│   ├── data/
│   │   ├── dataset.py                     # Load CoT-Collection, sat-reading, elementary_math
│   │   └── preprocessing.py               # Tokenization, CoT formatting
│   ├── models/
│   │   └── model.py                       # Model/tokenizer loading
│   ├── training/
│   │   ├── trainer.py                     # Seq2SeqTrainer with auto device/precision
│   │   └── callbacks.py                   # Experiment logging callback
│   ├── evaluation/
│   │   ├── metrics.py                     # ROUGE + exact match
│   │   └── evaluate.py                    # Checkpoint evaluator
│   └── utils/
│       ├── config.py                      # YAML config loader with deep merge
│       ├── logging.py                     # Rich logging, run naming, result tables
│       └── helpers.py                     # Device detection, seed, param counting
├── scripts/
│   ├── train.py                           # Main training entry point
│   ├── evaluate.py                        # Standalone evaluation
│   └── run_experiment.py                  # Batch experiment runner
├── outputs/                               # Training outputs (gitignored)
├── tests/
├── docs/specs/                            # Design specs
├── pyproject.toml                         # uv project config
├── uv.lock                                # Locked dependencies
└── Makefile                               # Common commands
```

## Output Naming

```
outputs/{model_name}/{exp_name}_lr{lr}_bs{bs}x{ga}_{YYYYMMDD_HHMM}/
```

Example: `outputs/flan-t5-small/cot_small_with_cot_lr5e-05_bs4x4_20260711_1430/`

Each run saves:
- `checkpoints/` - Model checkpoints
- `logs/` - TensorBoard logs
- `results/` - JSON metrics (train + eval)
- `config.yaml` - Copy of the config used

## Evaluation

```bash
uv run python scripts/evaluate.py \
  --config configs/exp33_cot/flan_t5_small_with_cot.yaml \
  --checkpoint outputs/flan-t5-small/cot_small_with_cot_lr5e-05_bs4x4_20260711_1430
```

Metrics: ROUGE-1, ROUGE-2, ROUGE-L, Exact Match, Eval Loss.

## TensorBoard

```bash
uv run tensorboard --logdir outputs/
```

## Datasets

| Dataset | Source | Use |
|---------|--------|-----|
| CoT-Collection | `kaist-ai/CoT-Collection` | Exp 3.1, 3.3 (1.84M examples, 1060 tasks) |
| SAT Reading | `emozilla/sat-reading` | Exp 3.2 |
| Elementary Math | `emozilla/elementary_math-v1` | Exp 3.2 |

## Tips for Low-Resource

- Start with `--max-train-samples 500` to test configs before full runs
- Reduce `max_source_length` to 256 in config for faster training
- Use `gradient_accumulation_steps: 8` with `batch_size: 2` to save memory
- Set `save_total_limit: 1` to keep only the best checkpoint
- On MPS: avoid `num_proc > 0` for dataset preprocessing (auto-handled)
