import pytest

from score_leads import priority_tier, validate_priority_thresholds


def test_custom_priority_thresholds_change_tier_boundaries():
    assert priority_tier(49.9, medium_threshold=50, high_threshold=80) == "Low"
    assert priority_tier(50, medium_threshold=50, high_threshold=80) == "Medium"
    assert priority_tier(79.9, medium_threshold=50, high_threshold=80) == "Medium"
    assert priority_tier(80, medium_threshold=50, high_threshold=80) == "High"


@pytest.mark.parametrize(
    ("medium", "high"),
    [(40, 40), (70, 40), (-1, 80), (20, 101), (float("nan"), 80)],
)
def test_invalid_priority_thresholds_are_rejected(medium, high):
    with pytest.raises(ValueError, match="0 <= medium < high <= 100"):
        validate_priority_thresholds(medium, high)
