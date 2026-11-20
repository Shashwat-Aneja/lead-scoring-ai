import json,pandas as pd,pytest
from score_leads import export_lead_summary,main
def test_summary_counts_all_priority_tiers(tmp_path):
    p=tmp_path/'summary.json'
    export_lead_summary(pd.DataFrame({'priority':['High','Low','High'],'conversion_probability':[80.,20.,90.]}),p)
    data=json.loads(p.read_text());assert data['lead_count']==3 and data['priority_counts']=={'High':2,'Medium':0,'Low':1}
def test_summary_alias_cannot_overwrite_training_csv(tmp_path):
    p=tmp_path/'leads.csv';p.write_text('keep')
    assert main([str(p),'--summary-output',str(p)])==1
    assert p.read_text()=='keep'
def test_bad_summary_serialization_preserves_previous_file(tmp_path):
    p=tmp_path/'summary.json';p.write_text('keep')
    with pytest.raises(ValueError):export_lead_summary(pd.DataFrame({'priority':['High'],'conversion_probability':[float('nan')]}),p)
    assert p.read_text()=='keep'
