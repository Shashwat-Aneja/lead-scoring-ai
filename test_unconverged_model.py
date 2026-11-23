import warnings
import score_leads as app
from sklearn.exceptions import ConvergenceWarning
def test_unconverged_fit_does_not_publish_scores_or_replace_export(tmp_path,monkeypatch):
    p=tmp_path/'leads.csv';p.write_text('interaction_score,visits,total_spent,converted\n1,1,1,0\n2,2,2,1\n')
    out=tmp_path/'out.csv';out.write_text('keep')
    class Model:
        def fit(self,*args):warnings.warn('iteration limit',ConvergenceWarning)
        def predict_proba(self,*args):raise AssertionError('unconverged scores used')
    monkeypatch.setattr(app,'make_pipeline',lambda *args:Model())
    assert app.main([str(p),'--output',str(out)])==1
    assert out.read_text()=='keep'
