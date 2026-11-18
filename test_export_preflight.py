import os
import score_leads


def test_input_alias_is_rejected_before_training(tmp_path, monkeypatch):
    source = tmp_path / "train.csv"
    source.write_text("data")
    alias = tmp_path / "alias.csv"
    os.link(source, alias)
    def fail(*args, **kwargs):
        raise AssertionError("must not fit model before rejecting input alias")
    monkeypatch.setattr(score_leads, "score_leads", fail)
    assert score_leads.main([str(source), "--output", str(alias)]) == 1
    assert source.read_text() == "data"
