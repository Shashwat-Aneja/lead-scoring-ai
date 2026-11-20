from score_leads import _reject_duplicate_headers
import pytest
def test_duplicate_headers_are_rejected_instead_of_renamed(tmp_path):
    p=tmp_path/'data.csv';p.write_text('value,value\n1,2\n')
    with pytest.raises(ValueError,match='unique'):_reject_duplicate_headers(p)
def test_bom_header_and_distinct_names_are_supported(tmp_path):
    p=tmp_path/'data.csv';p.write_text('\ufeffvalue,other\n1,2\n',encoding='utf-8')
    _reject_duplicate_headers(p)

def test_duplicate_conversion_columns_do_not_train_on_first_label(tmp_path):
    from score_leads import score_leads
    p=tmp_path/'data.csv';p.write_text('interaction_score,visits,total_spent,converted,converted\n1,1,1,0,1\n2,2,2,1,0\n')
    assert score_leads(p) is None
