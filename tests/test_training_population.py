import json
from score_leads import score_leads, export_lead_summary

def test_training_population_is_not_selected_scoring_population(tmp_path):
    train=tmp_path/"t.csv";score=tmp_path/"s.csv";out=tmp_path/"o.json"
    train.write_text("interaction_score,visits,total_spent,converted\n1,1,10,0\n2,2,20,0\n3,3,30,0\n8,8,80,1\n9,9,90,1\n")
    score.write_text("interaction_score,visits,total_spent\n1,1,10\n9,9,90\n")
    result=score_leads(train,scoring_file=score,top_n=1)
    export_lead_summary(result,out)
    data=json.loads(out.read_text())
    assert data["lead_count"]==1 and data["scored_lead_count"]==2
    assert data["training_population"]=={"samples":5,"converted":2,"not_converted":3,"conversion_rate_percent":40.0}
