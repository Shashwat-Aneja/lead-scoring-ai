import pandas as pd
from score_leads import export_scored_leads
def test_opt_in_export_escapes_text_and_preserves_numeric_values(tmp_path):
    data=pd.DataFrame({'id':['@SUM(1)',' -formula','plain'],'amount':[-2,3,4]})
    out=tmp_path/'out.csv';export_scored_leads(data,out,escape_formulas=True)
    result=pd.read_csv(out)
    assert result['id'].tolist()==["'@SUM(1)","' -formula",'plain']
    assert result['amount'].tolist()==[-2,3,4] and data['id'].iloc[0]=='@SUM(1)'
def test_default_export_preserves_original_text(tmp_path):
    data=pd.DataFrame({'id':['=2+2']});out=tmp_path/'out.csv';export_scored_leads(data,out)
    assert pd.read_csv(out)['id'].iloc[0]=='=2+2'
