from pathlib import Path

from src.utils.config import deep_merge, load_config, load_yaml


def test_deep_merge() -> None:
    base = {"a": 1, "b": {"c": 2, "d": 3}}
    override = {"b": {"c": 10}, "e": 5}
    result = deep_merge(base, override)
    assert result == {"a": 1, "b": {"c": 10, "d": 3}, "e": 5}


def test_deep_merge_no_mutation() -> None:
    base = {"a": {"b": 1}}
    override = {"a": {"c": 2}}
    result = deep_merge(base, override)
    assert "c" not in base["a"]
    assert result["a"] == {"b": 1, "c": 2}


def test_load_config() -> None:
    config = load_config(
        "configs/exp31_scaling_tasks/flan_t5_small_100tasks.yaml",
        "configs/base.yaml",
    )
    assert config["model"]["name_or_path"] == "google/flan-t5-small"
    assert config["data"]["num_tasks"] == 100
    assert config["training"]["learning_rate"] == 5.0e-5


def test_base_config_loads() -> None:
    config = load_yaml("configs/base.yaml")
    assert "model" in config
    assert "data" in config
    assert "training" in config
