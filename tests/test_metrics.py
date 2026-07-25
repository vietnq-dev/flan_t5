from src.evaluation.metrics import exact_match_score, normalize_bool_answer


def test_normalize_bool_answer() -> None:
    assert normalize_bool_answer("Yes.") == "yes"
    assert normalize_bool_answer("true") == "yes"
    assert normalize_bool_answer("False.") == "no"


def test_exact_match_score_normalizes_bool_labels() -> None:
    assert exact_match_score("True.", "yes")
    assert exact_match_score("No, it is not.", "no")
    assert not exact_match_score("yes", "no")


def test_exact_match_score_keeps_non_bool_labels_strict() -> None:
    assert exact_match_score("Denver", "Denver")
    assert not exact_match_score("The Denver", "Denver")
