# Flan-T5 Instruction Fine-Tuning Experiments

Replicate small-scale experiments from the paper [Scaling Instruction-Finetuned Language Models](https://arxiv.org/abs/2210.11416) (Chung et al., 2022) using Flan-T5-Small (77M) and Flan-T5-Base (248M).

## Setup

```bash
uv sync --all-groups
```

Verify installation:

```bash
uv run python -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available())"
```

```bash
uv run python scripts/train.py --config configs/exp33_cot/flan_t5_small_with_cot.yaml
```

## Precision (important for T5 on T4)

`base.yaml` pins `fp16: false`. T5's parallel attention numerically NaNs
(`loss=0.0`, `grad_norm=nan`, `lr=0.0`) under fp16 autocast — known issue
on T4 (Turing, no bf16 support). The Trainer runs fp32 on T4 (small
flan-t5 fits in 15.6 GB easily). On Ampere+ GPUs (A100, etc.), `bf16`
auto-enables for speed. Override per-config if needed:

```yaml
precision:
  fp16: false      # keep false for T5
  bf16: true       # set true on Ampere+ if you want explicit bf16
  tf32: true
```

## Dataloader workers

`base.yaml` pins `dataloader_num_workers: 0`. Higher values can deadlock
the PyTorch DataLoader in Jupyter/Colab (fork-based workers + HF
datasets), causing the trainer to hang after `on_train_begin` with no
tqdm bar. Bump per-config for CLI/local throughput, leave `0` on
notebooks.

## Running on Google Colab (T4)

Use a **T4 GPU runtime**. Clone the repo, install deps, then run `train.py`
directly. Pin `--max-samples` for quick tests; drop it for full runs.

```bash
!git clone https://github.com/vietnq-dev/flan_t5.git /content/flan_t5
%cd /content/flan_t5
!pip install -q torch transformers datasets accelerate sentencepiece \
  tensorboard rouge-score nltk sacrebleu tqdm rich pyyaml ijson

# quick sanity check (5k samples, ~10 min on T4)
!python scripts/train.py \
  --config configs/exp33_cot/flan_t5_small_with_cot.yaml \
  --max-samples 5000
```

Watch the log: after `Calling trainer.train()...` you should see `Step 1:
{'loss': ...}` within ~2s. If `loss=0.0` / `grad_norm=nan` appears, fp16 is
still on somewhere — ensure `precision.fp16: false` in config. For
TensorBoard in Colab, see the [TensorBoard](#tensorboard) section below.

## Experiment Guide

All experiments in recommended order, from quickest to most expensive:

---

### 0. Quick sanity check (~10 min on T4)

Verify the pipeline works end-to-end with a tiny subset:

```bash
uv run python scripts/train.py \
  --config configs/exp33_cot/flan_t5_small_with_cot.yaml \
  --max-samples 1000
```

---

### 1. Exp 3.3 — Chain-of-Thought effect (~2–4h on T4)

Compare training with vs without CoT rationales. Most impactful experiment, smallest dataset.

**Batch run:**

```bash
uv run scripts/run_experiment.py --experiment exp33
```

**Individual configs:**

```bash
# -- Flan-T5-Small -------------------------------------------------------
uv run python scripts/train.py --config configs/exp33_cot/flan_t5_small_no_cot.yaml
uv run python scripts/train.py --config configs/exp33_cot/flan_t5_small_with_cot.yaml

# -- Flan-T5-Base --------------------------------------------------------
uv run python scripts/train.py --config configs/exp33_cot/flan_t5_base_no_cot.yaml
uv run python scripts/train.py --config configs/exp33_cot/flan_t5_base_with_cot.yaml
```

**What to look for:** Models trained with CoT should show better ROUGE-L. Models without CoT should degrade on reasoning tasks.

**Evaluate:**

```bash
# After training, evaluate each checkpoint:
uv run python scripts/evaluate.py \
  --config configs/exp33_cot/flan_t5_small_no_cot.yaml \
  --checkpoint outputs/flan-t5-small/<cot_small_no_cot_run>

uv run python scripts/evaluate.py \
  --config configs/exp33_cot/flan_t5_small_with_cot.yaml \
  --checkpoint outputs/flan-t5-small/<cot_small_with_cot_run>

uv run python scripts/evaluate.py \
  --config configs/exp33_cot/flan_t5_base_no_cot.yaml \
  --checkpoint outputs/flan-t5-base/<cot_base_no_cot_run>

uv run python scripts/evaluate.py \
  --config configs/exp33_cot/flan_t5_base_with_cot.yaml \
  --checkpoint outputs/flan-t5-base/<cot_base_with_cot_run>
```

---

### 2. Exp 3.2 — Scaling model size (~1–2h on T4)

Train Small and Base on GSM8K and Elementary Math. Four configs in total — two dataset variants for each model size.

**Batch run:**

```bash
uv run scripts/run_experiment.py --experiment exp32
```

**Individual configs:**

```bash
# -- GSM8K ---------------------------------------------------------------
uv run python scripts/train.py --config configs/exp32_scaling_model/flan_t5_small_gsm8k.yaml
uv run python scripts/train.py --config configs/exp32_scaling_model/flan_t5_base_gsm8k.yaml

# -- Elementary Math -----------------------------------------------------
uv run python scripts/train.py --config configs/exp32_scaling_model/flan_t5_small_sat_math.yaml
uv run python scripts/train.py --config configs/exp32_scaling_model/flan_t5_base_sat_math.yaml
```

**What to look for:** Base > Small on the same data — instruction-tuning benefits scale with model size. GSM8K gives a larger and more stable reasoning benchmark than the tiny SAT Reading split.

**Evaluate:**

```bash
uv run python scripts/evaluate.py \
  --config configs/exp32_scaling_model/flan_t5_small_gsm8k.yaml \
  --checkpoint outputs/flan-t5-small/<gsm8k_small_run>

uv run python scripts/evaluate.py \
  --config configs/exp32_scaling_model/flan_t5_base_gsm8k.yaml \
  --checkpoint outputs/flan-t5-base/<gsm8k_base_run>

uv run python scripts/evaluate.py \
  --config configs/exp32_scaling_model/flan_t5_small_sat_math.yaml \
  --checkpoint outputs/flan-t5-small/<sat_math_small_run>

uv run python scripts/evaluate.py \
  --config configs/exp32_scaling_model/flan_t5_base_sat_math.yaml \
  --checkpoint outputs/flan-t5-base/<sat_math_base_run>
```

---

### 3. Exp 3.1 — Scaling number of tasks (~4–8h on T4)

Largest experiment. Trains Small on 100, 300, 600, and 1060 tasks from CoT-Collection.

**Batch run:**

```bash
uv run scripts/run_experiment.py --experiment exp31
```

**Individual configs:**

```bash
uv run python scripts/train.py --config configs/exp31_scaling_tasks/flan_t5_small_100tasks.yaml
uv run python scripts/train.py --config configs/exp31_scaling_tasks/flan_t5_small_300tasks.yaml
uv run python scripts/train.py --config configs/exp31_scaling_tasks/flan_t5_small_600tasks.yaml
uv run python scripts/train.py --config configs/exp31_scaling_tasks/flan_t5_small_1060tasks.yaml
```

**What to look for:** Performance improves with more tasks, diminishing returns after ~300 tasks (matches the paper's finding at 282 tasks).

**Evaluate:**

```bash
uv run python scripts/evaluate.py \
  --config configs/exp31_scaling_tasks/flan_t5_small_100tasks.yaml \
  --checkpoint outputs/flan-t5-small/<100tasks_run>

uv run python scripts/evaluate.py \
  --config configs/exp31_scaling_tasks/flan_t5_small_300tasks.yaml \
  --checkpoint outputs/flan-t5-small/<300tasks_run>

uv run python scripts/evaluate.py \
  --config configs/exp31_scaling_tasks/flan_t5_small_600tasks.yaml \
  --checkpoint outputs/flan-t5-small/<600tasks_run>

uv run python scripts/evaluate.py \
  --config configs/exp31_scaling_tasks/flan_t5_small_1060tasks.yaml \
  --checkpoint outputs/flan-t5-small/<1060tasks_run>
```

---

### All experiments

```bash
uv run scripts/run_experiment.py --experiment all
```

### Quick reference

```bash
# List all experiments
uv run scripts/run_experiment.py --list

# Dry run (print commands only)
uv run scripts/run_experiment.py --experiment exp33 --dry-run
```

## Project Structure

```
flan_t5/
├── configs/
│   ├── base.yaml                          # Shared defaults (fp32/bf16, workers=0)
│   ├── exp31_scaling_tasks/               # 4 configs: 100/300/600/1060 tasks
│   ├── exp32_scaling_model/               # 4 configs: SAT reading + SAT/math × small/base
│   └── exp33_cot/                         # 4 configs: with/without CoT x small/base
├── src/
│   ├── data/
│   │   ├── dataset.py                     # Load CoT-Collection, GSM8K, elementary_math
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

Metrics: ROUGE-1, ROUGE-2, ROUGE-L, Exact Match, Num samples. Generation
and scoring show tqdm progress bars with ETA.

### Fast smoke evaluation

`--max-samples` / `--max-eval-samples` cap dataset loading at the source so
you don't stream the full 1.8M CoT-Collection just to test a checkpoint.
`--num-beams 1` (greedy) is ~4× faster than the default `num_beams: 4` and
is fine for iterative ROUGE. `--batch-size` overrides the per-device eval
batch size.

```bash
uv run python scripts/evaluate.py \
  --config configs/exp33_cot/flan_t5_small_with_cot.yaml \
  --checkpoint outputs/flan-t5-small/cot_small_with_cot_lr5e-05_bs4x4_20260711_1430 \
  --max-eval-samples 500 \
  --num-beams 1 \
  --batch-size 8
```

Flag reference:

| Flag                    | Purpose                                                            |
| ----------------------- | ----------------------------------------------------------------- |
| `--max-samples N`       | Cap total samples before split (source cap during streaming)        |
| `--max-eval-samples N`  | Select first N eval samples after split (smoke test)               |
| `--num-beams N`         | Override `generation.num_beams` (1 = greedy, ~4× faster)           |
| `--batch-size N`        | Override `per_device_eval_batch_size`                              |

Metrics: ROUGE-1, ROUGE-2, ROUGE-L, Exact Match, Num samples.

## TensorBoard

```bash
uv run tensorboard --logdir outputs/
```

On Colab, run in a cell after training starts:

```python
%load_ext tensorboard
%tensorboard --logdir outputs
```

Colab streams events live; keep `--logdir outputs` to compare all runs
(grouped by subfolder name).

## Datasets

| Dataset | Source | Use |
|---------|--------|-----|
| CoT-Collection | `kaist-ai/CoT-Collection` | Exp 3.1, 3.3 (1.84M examples, 1060 tasks) |
| GSM8K | `openai/gsm8k` | Exp 3.2 |
| Elementary Math | `emozilla/elementary_math-v1` | Exp 3.2 |

## Tips for Low-Resource

- Start with `--max-samples 1000` to test configs before full runs (caps
  loading at the source; `--max-train-samples` only slices after loading)
- Reduce `max_source_length` to 256 in config for faster training
- Use `gradient_accumulation_steps: 8` with `batch_size: 2` to save memory
- Set `save_total_limit: 1` to keep only the best checkpoint
- Keep `precision.fp16: false` for T5 (fp16 NaNs the loss on T4). Use bf16
  on Ampere+ if available.
- Keep `dataloader_num_workers: 0` on Colab/Jupyter (workers can deadlock
  the DataLoader); bump to 2+ only on CLI/local runs.
