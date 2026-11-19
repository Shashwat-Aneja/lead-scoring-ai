import numpy as np
import score_leads as app
def test_ranked_leads_keep_equal_scores_stable(tmp_path,monkeypatch):
    p=tmp_path/'leads.csv';p.write_text('id,interaction_score,visits,total_spent,converted\na,1,1,1,0\nb,2,2,2,1\nc,3,3,3,0\n')
    class Model:
        classes_=[0,1]
        def fit(self,*args):pass
        def predict_proba(self,*args):return np.array([[.8,.2],[.2,.8],[.2,.8]])
    monkeypatch.setattr(app,'make_pipeline',lambda *args:Model())
    assert app.score_leads(p,sort_by_probability=True)['id'].tolist()==['b','c','a']
    assert app.score_leads(p)['id'].tolist()==['a','b','c']
