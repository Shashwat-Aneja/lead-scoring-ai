import argparse
import os
import tempfile
from pathlib import Path
from numbers import Real

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FEATURE_COLUMNS = ["interaction_score", "visits", "total_spent"]
REQUIRED_COLUMNS = set(FEATURE_COLUMNS + ["converted"])
DEFAULT_MEDIUM_THRESHOLD = 40.0
DEFAULT_HIGH_THRESHOLD = 70.0


def validate_priority_thresholds(medium_threshold, high_threshold):
    """Validate the ordered probability boundaries used for lead follow-up tiers."""
    if (
        isinstance(medium_threshold, (bool, np.bool_))
        or isinstance(high_threshold, (bool, np.bool_))
        or not isinstance(medium_threshold, Real)
        or not isinstance(high_threshold, Real)
        or not np.isfinite(medium_threshold)
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
    if (isinstance(probability, (bool, np.bool_)) or not isinstance(probability, Real)
            or not np.isfinite(probability) or probability < 0 or probability > 100):
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
    scoring_file=None,
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
        if (df["visits"] < 0).any() or not np.equal(
            df["visits"], np.floor(df["visits"])
        ).all():
            raise ValueError("Lead visit counts must be non-negative whole numbers.")
        if (df["interaction_score"] < 0).any():
            raise ValueError("Lead interaction_score must be non-negative.")
        if (df["total_spent"] < 0).any():
            raise ValueError("Lead total_spent must be non-negative.")
        if not set(df["converted"].unique()).issubset({0, 1}):
            raise ValueError("'converted' must contain only 0 or 1.")
        if df["converted"].nunique() != 2:
            raise ValueError("Training data must contain both converted and non-converted leads.")

        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
        model.fit(df[FEATURE_COLUMNS], df["converted"].astype(int))
        if scoring_file is not None:
            candidates = pd.read_csv(scoring_file)
            missing = set(FEATURE_COLUMNS).difference(candidates.columns)
            if missing:
                raise ValueError(f"Scoring CSV must include: {', '.join(sorted(missing))}")
            for column in FEATURE_COLUMNS:
                candidates[column] = pd.to_numeric(candidates[column], errors="coerce")
            if candidates.empty or not np.isfinite(candidates[FEATURE_COLUMNS].to_numpy(dtype=float)).all():
                raise ValueError("Scoring features must be non-empty finite numbers.")
            if (candidates[FEATURE_COLUMNS] < 0).any().any() or not np.equal(candidates["visits"], np.floor(candidates["visits"])).all():
                raise ValueError("Scoring features must be non-negative and visits must be whole numbers.")
            df = candidates.copy()
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


def export_scored_leads(result, output_path):
    """Replace an existing report only after a complete export is written."""
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="", delete=False,
            dir=output_path.parent, prefix=f".{output_path.name}.", suffix=".tmp",
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            result.to_csv(temporary_file, index=False)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        os.replace(temporary_path, output_path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


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
    parser.add_argument("--score-csv", help="Separate unlabeled lead CSV to score using the training CSV")
    args = parser.parse_args(argv)
    try:
        validate_priority_thresholds(args.medium_threshold, args.high_threshold)
    except ValueError as exc:
        parser.error(str(exc))

    scoring_options = {"scoring_file": args.score_csv} if args.score_csv else {}
    result = score_leads(
        args.csv_file,
        medium_threshold=args.medium_threshold,
        high_threshold=args.high_threshold,
        **scoring_options,
    )
    if result is None:
        return 1
    source_path = Path(args.csv_file).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve() if args.output else None
    if output_path is not None:
        same_file = output_path == source_path
        if args.score_csv:
            scoring_path = Path(args.score_csv).expanduser().resolve()
            same_file = same_file or output_path == scoring_path
            if output_path.exists() and scoring_path.exists():
                same_file = same_file or output_path.samefile(scoring_path)
        if not same_file and output_path.exists():
            try:
                same_file = output_path.samefile(source_path)
            except OSError:
                # Do not replace a file whose identity cannot be checked.
                same_file = True
        if same_file:
            print("❌ Output CSV must be different from the source file.")
            return 1
    if output_path is not None:
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            export_scored_leads(result, output_path)
        except OSError as exc:
            print(f"❌ Could not write scored leads: {exc}")
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
