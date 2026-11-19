import score_leads as app
def test_invalid_scoring_file_rejected_before_model_fit(tmp_path,monkeypatch):
    train=tmp_path/'train.csv';train.write_text('interaction_score,visits,total_spent,converted\n1,1,1,0\n2,2,2,1\n')
    bad=tmp_path/'bad.csv';bad.write_text('wrong\n1\n')
    class Model:
        def fit(self,*args):raise AssertionError('fit must not run before scoring validation')
    monkeypatch.setattr(app,'make_pipeline',lambda *args:Model())
    assert app.score_leads(train,scoring_file=bad) is None
