import numpy as np
from score_leads import score_leads
def test_large_finite_lead_features_produce_probabilities(tmp_path):
    p=tmp_path/'leads.csv';p.write_text('interaction_score,visits,total_spent,converted\n1,1,1e308,0\n2,2,1.1e308,0\n8,8,1.5e308,1\n9,9,1.6e308,1\n')
    result=score_leads(p)
    assert result is not None
    assert np.isfinite(result['conversion_probability']).all()
    assert result['conversion_probability'].between(0,100).all()
