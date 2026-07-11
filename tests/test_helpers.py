from src.utils.helpers import flatten_dict, format_number, model_short_name


def test_model_short_name() -> None:
    assert model_short_name("google/flan-t5-small") == "flan-t5-small"
    assert model_short_name("google/flan-t5-base") == "flan-t5-base"


def test_format_number() -> None:
    assert format_number(500) == "500"
    assert format_number(1500) == "1.5K"
    assert format_number(1_500_000) == "1.5M"


def test_flatten_dict() -> None:
    d = {"a": {"b": 1, "c": {"d": 2}}, "e": 3}
    result = flatten_dict(d)
    assert result == {"a.b": 1, "a.c.d": 2, "e": 3}
