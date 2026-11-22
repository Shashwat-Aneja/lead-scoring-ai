import csv
import argparse
import json
import os
import tempfile
from pathlib import Path
from numbers import Real

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, MaxAbsScaler

FEATURE_COLUMNS = ["interaction_score", "visits", "total_spent"]
REQUIRED_COLUMNS = set(FEATURE_COLUMNS + ["converted"])
DEFAULT_MEDIUM_THRESHOLD = 40.0
DEFAULT_HIGH_THRESHOLD = 70.0


def _read_csv_input(csv_file, delimiter=",", **options):
    if delimiter not in (",", ";", "\t", "|"):
        raise ValueError("Unsupported input delimiter.")
    frame = pd.read_csv(csv_file, **options, **({"sep": delimiter} if delimiter != "," else {}))
    _reject_duplicate_headers(csv_file, delimiter)
    return frame


def escape_csv_formulas(frame):
    """Escape formula-like text in a copy; leave numeric cells unchanged."""
    def escape(value):
        if isinstance(value, str) and value.lstrip(" \t\r\n").startswith(("=", "+", "-", "@")):
            return "'" + value
        return value
    return frame.apply(lambda column: column.map(escape))


def _reject_duplicate_headers(csv_file, delimiter=","):
    """Do not let pandas silently rename ambiguous source CSV columns."""
    if isinstance(csv_file, (str, os.PathLike)) and Path(csv_file).is_file():
        with open(csv_file, encoding="utf-8-sig", newline="") as source:
            headers = next(csv.reader(source, delimiter=delimiter), [])
        if len(headers) != len(set(headers)):
            raise ValueError("CSV column names must be unique; duplicate headers are ambiguous.")


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
    max_display_rows=20,
    sort_by_probability=False,
    delimiter=",",
):
    try:
        validate_priority_thresholds(medium_threshold, high_threshold)
        if not isinstance(max_display_rows, int) or isinstance(max_display_rows, bool) or max_display_rows < 1:
            raise ValueError("max_display_rows must be a positive integer.")
        df = _read_csv_input(csv_file, delimiter)
        missing = REQUIRED_COLUMNS.difference(df.columns)
        if missing:
            raise ValueError(f"CSV must include: {', '.join(sorted(missing))}")

        df = df.copy()
        if df[FEATURE_COLUMNS].apply(lambda column: column.map(lambda value: isinstance(value, (bool, np.bool_)))).any().any():
            raise ValueError("Lead features must be numeric, not booleans.")
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

        model = make_pipeline(MaxAbsScaler(), StandardScaler(), LogisticRegression(max_iter=1000))
        training_features = df[FEATURE_COLUMNS].copy()
        training_labels = df["converted"].astype(int).copy()
        if scoring_file is not None:
            candidates = _read_csv_input(scoring_file, delimiter)
            missing = set(FEATURE_COLUMNS).difference(candidates.columns)
            if missing:
                raise ValueError(f"Scoring CSV must include: {', '.join(sorted(missing))}")
            if candidates[FEATURE_COLUMNS].apply(lambda column: column.map(lambda value: isinstance(value, (bool, np.bool_)))).any().any():
                raise ValueError("Scoring features must be numeric, not booleans.")
            for column in FEATURE_COLUMNS:
                candidates[column] = pd.to_numeric(candidates[column], errors="coerce")
            if candidates.empty or not np.isfinite(candidates[FEATURE_COLUMNS].to_numpy(dtype=float)).all():
                raise ValueError("Scoring features must be non-empty finite numbers.")
            if (candidates[FEATURE_COLUMNS] < 0).any().any() or not np.equal(candidates["visits"], np.floor(candidates["visits"])).all():
                raise ValueError("Scoring features must be non-negative and visits must be whole numbers.")
            df = candidates.copy()
        model.fit(training_features, training_labels)
        positive_class_index = list(model.classes_).index(1)
        df["conversion_probability"] = model.predict_proba(df[FEATURE_COLUMNS])[:, positive_class_index] * 100
        df["priority"] = df["conversion_probability"].apply(
            lambda probability: priority_tier(
                probability, medium_threshold=medium_threshold, high_threshold=high_threshold
            )
        )

        if sort_by_probability:
            df = df.sort_values("conversion_probability", ascending=False, kind="stable")

        print("\n===== Lead Conversion Scoring =====")
        print(df[FEATURE_COLUMNS + ["conversion_probability", "priority"]].head(max_display_rows).to_string(index=False))
        if len(df) > max_display_rows:
            print(f"Showing {max_display_rows} of {len(df)} leads; exports include every scored lead.")
        print("===================================\n")
        return df
    except (OSError, pd.errors.ParserError, ValueError) as exc:
        print(f"❌ Error: {exc}")
        return None


def export_scored_leads(result, output_path, escape_formulas=False):
    """Replace an existing report only after a complete export is written."""
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="", delete=False,
            dir=output_path.parent, prefix=f".{output_path.name}.", suffix=".tmp",
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            export = escape_csv_formulas(result) if escape_formulas else result
            export.to_csv(temporary_file, index=False)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        os.replace(temporary_path, output_path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def export_lead_summary(result, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"lead_count": len(result), "priority_counts": {tier: int((result["priority"] == tier).sum()) for tier in ("High", "Medium", "Low")},
               "average_conversion_probability": float(result["conversion_probability"].mean())}
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=output_path.parent, delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, allow_nan=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output_path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


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
    parser.add_argument("--max-display-rows", type=int, default=20, help="Maximum lead rows printed; output CSV remains complete")
    parser.add_argument("--sort-by-probability", action="store_true", help="Rank leads from highest conversion probability, keeping ties stable")
    parser.add_argument("--summary-output", help="Atomically export JSON lead counts, priority totals and mean conversion probability")
    parser.add_argument("--escape-formulas", action="store_true", help="Prefix formula-like text with an apostrophe in CSV exports; calculations remain unchanged")
    parser.add_argument("--delimiter", choices=(",", ";", "tab", "|"), default=",", help="Input CSV delimiter; exports remain comma-separated")
    args = parser.parse_args(argv)
    delimiter = "\t" if args.delimiter == "tab" else args.delimiter
    try:
        validate_priority_thresholds(args.medium_threshold, args.high_threshold)
    except ValueError as exc:
        parser.error(str(exc))

    try:
        source_path = Path(args.csv_file).expanduser().resolve()
        output_path = Path(args.output).expanduser().resolve() if args.output else None
        summary_path = Path(args.summary_output).expanduser().resolve() if args.summary_output else None
        if summary_path is not None:
            protected = [source_path] + ([output_path] if output_path is not None else []) + ([Path(args.score_csv).expanduser().resolve()] if args.score_csv else [])
            if any(summary_path == other or (summary_path.exists() and other.exists() and summary_path.samefile(other)) for other in protected):
                print("❌ Summary destination must differ from every input and scored CSV export.")
                return 1
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
    except OSError as exc:
        print(f"❌ Could not validate output path: {exc}")
        return 1

    scoring_options = {"scoring_file": args.score_csv} if args.score_csv else {}
    scoring_options["max_display_rows"] = args.max_display_rows
    if delimiter != ",":
        scoring_options["delimiter"] = delimiter
    if args.sort_by_probability:
        scoring_options["sort_by_probability"] = True
    result = score_leads(
        args.csv_file,
        medium_threshold=args.medium_threshold,
        high_threshold=args.high_threshold,
        **scoring_options,
    )
    if result is None:
        return 1
    if output_path is not None:
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            export_scored_leads(result, output_path, **({"escape_formulas": True} if args.escape_formulas else {}))
        except OSError as exc:
            print(f"❌ Could not write scored leads: {exc}")
            return 1
    if summary_path is not None:
        try:
            export_lead_summary(result, summary_path)
        except (OSError, ValueError) as exc:
            print(f"❌ Could not export lead summary: {exc}")
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
