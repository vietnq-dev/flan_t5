# Flan-T5 Instruction Fine-Tuning Experiments Design

**Date:** 2026-07-11
**Paper:** Scaling Instruction-Finetuned Language Models (Chung et al., 2022)

## Overview

Replicate small-scale experiments from the Flan-T5 paper using:
- Models: Flan-T5-Small (77M) and Flan-T5-Base (248M)
- Datasets: kaist-ai/CoT-Collection, openai/gsm8k, emozilla/elementary_math-v1
- Hardware: CPU/low-resource

## Experiments

### 3.1 Scaling Number of Tasks
- **Goal:** Measure how increasing task diversity affects performance
- **Model:** flan-t5-small
- **Dataset:** CoT-Collection (1,060 tasks total)
- **Task counts:** 100, 300, 600, 1060
- **Expected:** Diminishing returns after ~282 tasks

### 3.2 Scaling Model Size
- **Goal:** Compare small vs base model on same domain data
- **Models:** flan-t5-small, flan-t5-base
- **Dataset:** GSM8K + Elementary Math
- **Expected:** Larger models benefit more from instruction tuning

### 3.3 Chain-of-Thought Effect
- **Goal:** Measure impact of CoT rationales on reasoning
- **Models:** flan-t5-small, flan-t5-base
- **Dataset:** CoT-Collection (with vs without rationale field)
- **Expected:** CoT improves reasoning benchmarks; no-CoT degrades reasoning

## Architecture

```
flan_t5/
├── configs/           # YAML experiment configs
├── src/
│   ├── data/          # Dataset loading & preprocessing
│   ├── models/        # Model/tokenizer loading
│   ├── training/      # Trainer & callbacks
│   ├── evaluation/    # Metrics & evaluation runner
│   └── utils/         # Config, logging, helpers
├── scripts/           # Entry points (train, evaluate, run_experiment)
├── outputs/           # Training outputs (gitignored)
├── notebooks/         # Exploration notebooks
├── tests/             # Unit tests
└── openwiki/          # Auto-generated documentation
```

## Output Naming

```
outputs/{model_name}/{exp_name}_lr{lr}_bs{bs}x{ga}_{YYYYMMDD_HHMM}/
```

Each output contains: checkpoints/, logs/ (TensorBoard), results/ (JSON), config.yaml copy.

## Evaluation Metrics

- ROUGE-1, ROUGE-2, ROUGE-L
- Exact Match
- Eval Loss
- Held-out tasks from CoT-Collection + standard benchmarks (MMLU, ARC, HellaSwag)
