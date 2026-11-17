import numpy as np
import pandas as pd
from score_leads import score_leads


def test_changing_currency_units_does_not_change_lead_probabilities(tmp_path):
    frame = pd.DataFrame({"interaction_score": [1, 2, 8, 9], "visits": [1, 4, 8, 12], "total_spent": [10, 20, 100, 200], "converted": [0, 0, 1, 1]})
    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"
    frame.to_csv(first, index=False)
    frame["total_spent"] *= 100
    frame.to_csv(second, index=False)
    assert np.allclose(score_leads(first)["conversion_probability"], score_leads(second)["conversion_probability"])
