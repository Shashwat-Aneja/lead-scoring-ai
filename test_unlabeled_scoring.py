import pandas as pd
from score_leads import score_leads, main


def test_scores_unlabeled_candidates_without_training_on_them(tmp_path):
    train = tmp_path / "train.csv"
    candidates = tmp_path / "new.csv"
    train.write_text("interaction_score,visits,total_spent,converted\n1,1,10,0\n2,2,20,0\n8,8,80,1\n9,9,90,1\n")
    candidates.write_text("lead_id,interaction_score,visits,total_spent\nnew,5,5,50\n")
    result = score_leads(train, scoring_file=candidates)
    assert result["lead_id"].tolist() == ["new"]
    assert result["conversion_probability"].between(0, 100).all()
    original = candidates.read_bytes()
    assert main([str(train), "--score-csv", str(candidates), "--output", str(candidates)]) == 1
    assert candidates.read_bytes() == original
