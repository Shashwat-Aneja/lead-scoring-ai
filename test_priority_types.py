import pytest
from score_leads import priority_tier, validate_priority_thresholds


@pytest.mark.parametrize("value", [True, None, "50", complex(1, 2)])
def test_invalid_probability_type_is_a_value_error(value):
    with pytest.raises(ValueError, match="probability"):
        priority_tier(value)


@pytest.mark.parametrize("value", [True, None, "50", complex(1, 2)])
def test_invalid_threshold_types_are_validation_errors(value):
    with pytest.raises(ValueError, match="thresholds"):
        validate_priority_thresholds(value, 70)
    with pytest.raises(ValueError, match="thresholds"):
        validate_priority_thresholds(40, value)
