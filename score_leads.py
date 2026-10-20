import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

FEATURE_COLUMNS = ["interaction_score", "visits", "total_spent"]
REQUIRED_COLUMNS = set(FEATURE_COLUMNS + ["converted"])


def score_leads(csv_file):
    try:
        df = pd.read_csv(csv_file)
        missing = REQUIRED_COLUMNS.difference(df.columns)
        if missing:
            raise ValueError(f"CSV must include: {', '.join(sorted(missing))}")

        df = df.copy()
        for column in FEATURE_COLUMNS + ["converted"]:
            df[column] = pd.to_numeric(df[column], errors="coerce")
        if df.empty or df[FEATURE_COLUMNS + ["converted"]].isna().any().any():
            raise ValueError("Lead features and conversion labels must be numeric and non-empty.")
        if not np.isfinite(df[FEATURE_COLUMNS + ["converted"]].to_numpy(dtype=float)).all():
            raise ValueError("Lead features and conversion labels must be finite numbers.")
        if not set(df["converted"].unique()).issubset({0, 1}):
            raise ValueError("'converted' must contain only 0 or 1.")
        if df["converted"].nunique() != 2:
            raise ValueError("Training data must contain both converted and non-converted leads.")

        model = LogisticRegression()
        model.fit(df[FEATURE_COLUMNS], df["converted"].astype(int))
        positive_class_index = list(model.classes_).index(1)
        df["conversion_probability"] = model.predict_proba(df[FEATURE_COLUMNS])[:, positive_class_index] * 100

        print("\n===== Lead Conversion Scoring =====")
        print(df[FEATURE_COLUMNS + ["conversion_probability"]].to_string(index=False))
        print("===================================\n")
        return df
    except (OSError, pd.errors.ParserError, ValueError) as exc:
        print(f"❌ Error: {exc}")
        return None


def main():
    if len(sys.argv) != 2:
        print("Usage: python score_leads.py <csv_file>")
        return 2
    return 0 if score_leads(sys.argv[1]) is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
