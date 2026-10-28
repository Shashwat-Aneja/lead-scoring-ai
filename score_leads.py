import argparse

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

FEATURE_COLUMNS = ["interaction_score", "visits", "total_spent"]
REQUIRED_COLUMNS = set(FEATURE_COLUMNS + ["converted"])
DEFAULT_MEDIUM_THRESHOLD = 40.0
DEFAULT_HIGH_THRESHOLD = 70.0


def validate_priority_thresholds(medium_threshold, high_threshold):
    """Validate the ordered probability boundaries used for lead follow-up tiers."""
    if (
        not np.isfinite(medium_threshold)
        or not np.isfinite(high_threshold)
        or not 0 <= medium_threshold < high_threshold <= 100
    ):
        raise ValueError(
            "Priority thresholds must satisfy 0 <= medium < high <= 100."
        )


def priority_tier(
    probability,
    medium_threshold=DEFAULT_MEDIUM_THRESHOLD,
    high_threshold=DEFAULT_HIGH_THRESHOLD,
):
    """Map a model probability percentage to an explainable follow-up tier."""
    validate_priority_thresholds(medium_threshold, high_threshold)
    if not np.isfinite(probability) or probability < 0 or probability > 100:
        raise ValueError("Conversion probability must be between 0 and 100.")
    if probability >= high_threshold:
        return "High"
    if probability >= medium_threshold:
        return "Medium"
    return "Low"


def score_leads(
    csv_file,
    medium_threshold=DEFAULT_MEDIUM_THRESHOLD,
    high_threshold=DEFAULT_HIGH_THRESHOLD,
):
    try:
        validate_priority_thresholds(medium_threshold, high_threshold)
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
        df["priority"] = df["conversion_probability"].apply(
            lambda probability: priority_tier(
                probability, medium_threshold=medium_threshold, high_threshold=high_threshold
            )
        )

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
    parser.add_argument(
        "--medium-threshold",
        type=float,
        default=DEFAULT_MEDIUM_THRESHOLD,
        help="Minimum conversion probability for Medium priority (default: 40)",
    )
    parser.add_argument(
        "--high-threshold",
        type=float,
        default=DEFAULT_HIGH_THRESHOLD,
        help="Minimum conversion probability for High priority (default: 70)",
    )
    args = parser.parse_args(argv)
    try:
        validate_priority_thresholds(args.medium_threshold, args.high_threshold)
    except ValueError as exc:
        parser.error(str(exc))

    result = score_leads(
        args.csv_file,
        medium_threshold=args.medium_threshold,
        high_threshold=args.high_threshold,
    )
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
