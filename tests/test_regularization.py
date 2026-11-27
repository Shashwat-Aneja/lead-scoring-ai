import pytest
from score_leads import score_leads

def test_regularization_changes_score_spread(tmp_path):
    src=tmp_path/"a.csv"
    src.write_text("interaction_score,visits,total_spent,converted\n1,1,10,0\n2,2,20,0\n8,8,80,1\n9,9,90,1\n")
    strong=score_leads(src,regularization_c=.001).conversion_probability
    weak=score_leads(src,regularization_c=100).conversion_probability
    assert strong.max()-strong.min() < weak.max()-weak.min()

@pytest.mark.parametrize("c", [True, 0, -1, float('nan'), float('inf'), 1001])
def test_bad_regularization_precedes_io(c):
    assert score_leads("missing",regularization_c=c) is None
