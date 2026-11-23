import numpy as np,json
import score_leads as app
def test_top_selection_keeps_ties_stable_and_reports_scored_coverage(tmp_path,monkeypatch):
    p=tmp_path/'leads.csv';p.write_text('id,interaction_score,visits,total_spent,converted\na,1,1,1,0\nb,2,2,2,1\nc,3,3,3,0\n')
    class Model:
        classes_=[0,1]
        def fit(self,*args):pass
        def predict_proba(self,*args):return np.array([[.2,.8],[.2,.8],[.8,.2]])
    monkeypatch.setattr(app,'make_pipeline',lambda *args:Model())
    result=app.score_leads(p,top_n=2)
    assert result['id'].tolist()==['a','b'] and result.attrs['scored_lead_count']==3
    out=tmp_path/'summary.json';app.export_lead_summary(result,out)
    data=json.loads(out.read_text());assert data['lead_count']==2 and data['scored_lead_count']==3
def test_invalid_top_selection_fails_before_csv_read():
    assert app.score_leads('missing.csv',top_n=0) is None
