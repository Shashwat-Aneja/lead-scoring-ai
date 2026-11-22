import pandas as pd,numpy as np
import score_leads as app
def test_balanced_mode_configures_classifier_and_produces_finite_scores(tmp_path,monkeypatch):
    p=tmp_path/'leads.csv';pd.DataFrame({'interaction_score':[1,2,3,4,5,9],'visits':[1,2,3,4,5,9],'total_spent':[1,2,3,4,5,9],'converted':[0,0,0,0,0,1]}).to_csv(p,index=False)
    observed=[];original=app.make_pipeline
    def capture(*steps):observed.append(steps[-1].class_weight);return original(*steps)
    monkeypatch.setattr(app,'make_pipeline',capture)
    result=app.score_leads(p,balance_classes=True)
    assert observed==['balanced'] and np.isfinite(result['conversion_probability']).all()
