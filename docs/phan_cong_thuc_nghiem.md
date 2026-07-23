# Phân Công Thực Nghiệm Flan-T5

## 3.1. Scaling Number of Tasks — Tiên

**Mô tả:** Huấn luyện Flan-T5-Small trên CoT-Collection với số lượng tasks khác nhau: 50, 100, 300, 600. Đánh giá ảnh hưởng của số lượng task đến hiệu năng.

**Model:** `google/flan-t5-small`

**Số configs:** 4

> ⚠️ Cần tạo thêm config `flan_t5_small_50tasks.yaml` (hiện đã có file untracked). Xóa hoặc bỏ qua config `1060tasks` — không nằm trong phân công.

```bash
# 50 tasks — chạy nhanh nhất
uv run python scripts/train.py \
  --config configs/exp31_scaling_tasks/flan_t5_small_50tasks.yaml

# 100 tasks
uv run python scripts/train.py \
  --config configs/exp31_scaling_tasks/flan_t5_small_100tasks.yaml

# 300 tasks
uv run python scripts/train.py \
  --config configs/exp31_scaling_tasks/flan_t5_small_300tasks.yaml

# 600 tasks — nặng nhất
uv run python scripts/train.py \
  --config configs/exp31_scaling_tasks/flan_t5_small_600tasks.yaml
```

**Hoặc chạy batch toàn bộ:**

```bash
uv run scripts/run_experiment.py --experiment exp31
```



---

## 3.2. Scaling Model Size — Khang & Trang

**Mô tả:** Huấn luyện Flan-T5-Small và Flan-T5-Base trên cùng dataset để so sánh hiệu năng theo kích thước mô hình.

**Model:** `google/flan-t5-small`, `google/flan-t5-base`

### Khang — Elementary Math

**Số configs:** 2

```bash
# Flan-T5-Small trên Elementary Math
uv run python scripts/train.py \
  --config configs/exp32_scaling_model/flan_t5_small_sat_math.yaml

# Flan-T5-Base trên Elementary Math
uv run python scripts/train.py \
  --config configs/exp32_scaling_model/flan_t5_base_sat_math.yaml
```

### Trang — GSM8K

**Số configs:** 2

```bash
# Flan-T5-Small trên GSM8K
uv run python scripts/train.py \
  --config configs/exp32_scaling_model/flan_t5_small_gsm8k.yaml

# Flan-T5-Base trên GSM8K
uv run python scripts/train.py \
  --config configs/exp32_scaling_model/flan_t5_base_gsm8k.yaml
```

---

## 3.3. Chain-of-Thought Finetuning — Việt

**Mô tả:** So sánh hiệu năng khi finetune có và không có Chain-of-Thought rationales trên CoT-Collection, với cả Flan-T5-Small và Flan-T5-Base.

**Model:** `google/flan-t5-small`, `google/flan-t5-base`

**Số configs:** 4

```bash
# -- Flan-T5-Small -------------------------------------------------------
# Không có CoT
uv run python scripts/train.py \
  --config configs/exp33_cot/flan_t5_small_no_cot.yaml

# Có CoT
uv run python scripts/train.py \
  --config configs/exp33_cot/flan_t5_small_with_cot.yaml

# -- Flan-T5-Base --------------------------------------------------------
# Không có CoT
uv run python scripts/train.py \
  --config configs/exp33_cot/flan_t5_base_no_cot.yaml

# Có CoT
uv run python scripts/train.py \
  --config configs/exp33_cot/flan_t5_base_with_cot.yaml
```

**Hoặc chạy batch toàn bộ:**

```bash
uv run scripts/run_experiment.py --experiment exp33
```

---

## Lưu ý chung

### Cài đặt môi trường

```bash
git clone https://github.com/vietnq-dev/flan_t5.git
cd flan_t5
uv sync --all-groups
```

### Kiểm tra GPU

```bash
uv run python -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available())"
```

### Giới hạn dữ liệu để test nhanh

Tất cả configs exp31 và exp32 đã được set `max_samples: 100000`. Có thể override bằng CLI:

```bash
uv run python scripts/train.py \
  --config configs/exp33_cot/flan_t5_small_with_cot.yaml \
  --max-samples 1000
```

### Theo dõi training

```bash
uv run tensorboard --logdir outputs/
```

### Cấu trúc output

```
outputs/{model_name}/{exp_name}_lr{lr}_bs{bs}x{ga}_{timestamp}/
├── checkpoints/
├── logs/          # TensorBoard
├── results/       # JSON metrics
└── config.yaml
```

### Tổng hợp kết quả

Sau khi tất cả mọi người train xong, tập hợp các file JSON trong `results/` để so sánh:
- **Exp 3.1:** So sánh ROUGE-L theo số lượng tasks (50 → 100 → 300 → 600)
- **Exp 3.2:** So sánh eval_loss giữa Small vs Base trên cùng dataset
- **Exp 3.3:** So sánh ROUGE-L giữa có CoT vs không CoT cho từng model size
