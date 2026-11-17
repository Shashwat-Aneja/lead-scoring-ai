import pandas as pd
import pytest
from score_leads import export_scored_leads


def test_partial_export_does_not_destroy_previous_report(tmp_path, monkeypatch):
    output = tmp_path / "scored.csv"
    output.write_text("previous report")
    def fail(self, handle, index=False):
        handle.write("partial")
        raise OSError("disk full")
    monkeypatch.setattr(pd.DataFrame, "to_csv", fail)
    with pytest.raises(OSError, match="disk full"):
        export_scored_leads(pd.DataFrame({"score": [1]}), output)
    assert output.read_text() == "previous report"
    assert list(tmp_path.glob(".*.tmp")) == []


def test_successful_export_replaces_report(tmp_path):
    output = tmp_path / "scored.csv"
    export_scored_leads(pd.DataFrame({"score": [12]}), output)
    assert pd.read_csv(output)["score"].tolist() == [12]
