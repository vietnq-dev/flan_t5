.PHONY: install install-dev train evaluate lint format test clean

install:
	uv sync

install-dev:
	uv sync --all-groups

train:
	uv run python scripts/train.py --config $(CONFIG)

evaluate:
	uv run python scripts/evaluate.py --config $(CONFIG) --checkpoint $(CHECKPOINT)

run-exp:
	uv run python scripts/run_experiment.py --experiment $(EXP)

lint:
	uv run ruff check src/ scripts/
	uv run mypy src/ scripts/

format:
	uv run ruff format src/ scripts/

test:
	uv run pytest tests/ -v

clean:
	rm -rf outputs/ __pycache__ .pytest_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
