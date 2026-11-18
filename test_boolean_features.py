import pandas as pd
from score_leads import score_leads


def test_boolean_features_cannot_be_interpreted_as_money_or_counts(tmp_path, capsys):
    source = tmp_path / "train.csv"
    pd.DataFrame({"interaction_score": [True, False], "visits": [1, 2], "total_spent": [10, 20], "converted": [0, 1]}).to_csv(source, index=False)
    assert score_leads(source) is None
    assert "not booleans" in capsys.readouterr().out


def test_boolean_candidate_features_are_rejected(tmp_path):
    source = tmp_path / "train.csv"
    candidate = tmp_path / "new.csv"
    source.write_text("interaction_score,visits,total_spent,converted\n1,1,10,0\n2,2,20,1\n")
    candidate.write_text("interaction_score,visits,total_spent\n1,True,10\n")
    assert score_leads(source, scoring_file=candidate) is None
