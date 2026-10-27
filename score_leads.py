import argparse

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

FEATURE_COLUMNS = ["interaction_score", "visits", "total_spent"]
REQUIRED_COLUMNS = set(FEATURE_COLUMNS + ["converted"])


def priority_tier(probability):
    """Map a model probability percentage to an explainable follow-up tier."""
    if not np.isfinite(probability) or probability < 0 or probability > 100:
        raise ValueError("Conversion probability must be between 0 and 100.")
    if probability >= 70:
        return "High"
    if probability >= 40:
        return "Medium"
    return "Low"


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
        df["priority"] = df["conversion_probability"].apply(priority_tier)

        print("\n===== Lead Conversion Scoring =====")
        print(df[FEATURE_COLUMNS + ["conversion_probability", "priority"]].to_string(index=False))
        print("===================================\n")
        return df
    except (OSError, pd.errors.ParserError, ValueError) as exc:
        print(f"❌ Error: {exc}")
        return None


def main(argv=None):
    parser = argparse.ArgumentParser(description="Score leads from a CSV file.")
    parser.add_argument("csv_file", help="CSV containing lead features and conversion labels")
    parser.add_argument(
        "--output",
        "-o",
        help="Optional destination CSV for scored leads, including probabilities and priority tiers",
    )
    args = parser.parse_args(argv)

    result = score_leads(args.csv_file)
    if result is None:
        return 1
    if args.output:
        try:
            result.to_csv(args.output, index=False)
        except OSError as exc:
            print(f"❌ Could not write scored leads: {exc}")
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
