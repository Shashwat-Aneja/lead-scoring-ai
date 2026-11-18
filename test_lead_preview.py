import pandas as pd
from score_leads import score_leads


def test_scoring_returns_all_rows_with_bounded_console_output(tmp_path, capsys):
    source = tmp_path / "leads.csv"
    pd.DataFrame({"interaction_score": range(30), "visits": range(30), "total_spent": range(30), "converted": [0] * 15 + [1] * 15}).to_csv(source, index=False)
    result = score_leads(source, max_display_rows=3)
    assert len(result) == 30
    assert "Showing 3 of 30 leads" in capsys.readouterr().out
