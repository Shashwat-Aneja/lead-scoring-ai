import json
from score_leads import score_leads,export_lead_summary

def test_probability_floor_can_select_no_leads_with_valid_json_summary(tmp_path):
    p=tmp_path/'in.csv';p.write_text('interaction_score,visits,total_spent,converted\n1,1,1,0\n2,2,2,0\n9,9,100,1\n10,10,110,1\n')
    result=score_leads(p,min_probability=100,top_n=1)
    assert result is not None and result.empty and result.attrs['scored_lead_count']==4
    out=tmp_path/'summary.json';export_lead_summary(result,out)
    data=json.loads(out.read_text())
    assert data['lead_count']==0 and data['scored_lead_count']==4 and data['average_conversion_probability'] is None
    assert score_leads(p,min_probability=float('nan')) is None
