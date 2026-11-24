from score_leads import score_leads

def training(tmp_path):
    p=tmp_path/'train.csv';p.write_text('interaction_score,visits,total_spent,converted\n1,1,1,0\n2,2,2,0\n9,9,100,1\n10,10,110,1\n');return p

def test_scoring_identifiers_are_literal_and_training_need_not_have_ids(tmp_path):
    p=tmp_path/'score.csv';p.write_text('id,interaction_score,visits,total_spent\nNA,1,1,1\n001,2,2,2\nNULL,3,3,3\n')
    result=score_leads(training(tmp_path),scoring_file=p,id_column='id')
    assert result['id'].tolist()==['NA','001','NULL']

def test_duplicate_candidate_identifiers_fail_before_scores(tmp_path):
    p=tmp_path/'score.csv';p.write_text('id,interaction_score,visits,total_spent\nA,1,1,1\n A ,2,2,2\n')
    assert score_leads(training(tmp_path),scoring_file=p,id_column='id') is None
