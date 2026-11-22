import pandas as pd,pytest
from score_leads import _read_csv_input
@pytest.mark.parametrize('delimiter',[';', '\t', '|'])
def test_alternate_delimiter_preserves_quoted_fields(tmp_path,delimiter):
    source=tmp_path/'input.csv';data=pd.DataFrame({'name':['first'+delimiter+'second'],'amount':[12]});data.to_csv(source,index=False,sep=delimiter)
    result=_read_csv_input(source,delimiter=delimiter)
    assert result['name'].tolist()==data['name'].tolist() and result['amount'].tolist()==[12]
def test_duplicate_headers_are_checked_with_selected_delimiter(tmp_path):
    p=tmp_path/'input.csv';p.write_text('amount;amount\n1;999\n')
    with pytest.raises(ValueError,match='unique'):_read_csv_input(p,delimiter=';')

def test_selected_delimiter_reaches_the_cli_workflow(tmp_path):
    from score_leads import main
    source=tmp_path/'input.csv';source.write_text('interaction_score;visits;total_spent;converted\\n1;1;1;0\\n2;2;2;0\\n8;8;8;1\\n9;9;9;1\\n'.replace('\\n','\n'))
    arguments=[str(source),'--delimiter',';']+[]
    assert main(arguments)==0
