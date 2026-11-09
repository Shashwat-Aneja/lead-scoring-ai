import os

import pytest

from score_leads import main


def test_scoring_export_rejects_hard_link_to_training_csv(tmp_path, capsys):
    source = tmp_path / "leads.csv"
    alias = tmp_path / "scored.csv"
    original = (
        "interaction_score,visits,total_spent,converted\n"
        "1,2,10,0\n2,4,20,0\n8,10,100,1\n9,12,150,1\n"
    )
    source.write_text(original, encoding="utf-8")
    try:
        os.link(source, alias)
    except (OSError, NotImplementedError):
        pytest.skip("Hard links are not supported by this filesystem")

    assert main([str(source), "--output", str(alias)]) == 1
    assert source.read_text(encoding="utf-8") == original
    assert alias.read_text(encoding="utf-8") == original
    assert "different from the source file" in capsys.readouterr().out
