import zlib
import gzip
import bz2
import lzma
import io
import csv
import argparse
import warnings
from sklearn.exceptions import ConvergenceWarning
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


def _read_csv_input(csv_file, delimiter=",", max_rows=None, encoding="utf-8", **options):
    if encoding not in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
        raise ValueError("Unsupported input encoding.")
    if max_rows is not None and (isinstance(max_rows, bool) or not isinstance(max_rows, int) or not 1 <= max_rows <= 10_000_000):
        raise ValueError("max_rows must be an integer from 1 to 10000000.")
    if delimiter not in (",", ";", "\t", "|"):
        raise ValueError("Unsupported input delimiter.")
    if max_rows is not None:
        if "nrows" in options:
            raise ValueError("Do not combine nrows and max_rows.")
        options["nrows"] = max_rows + 1
    _reject_duplicate_headers(csv_file, delimiter, encoding=encoding)
    try:
        frame = pd.read_csv(csv_file, **({"encoding": encoding} if encoding != "utf-8" else {}), **options, **({"sep": delimiter} if delimiter != "," else {}))
    except (EOFError, lzma.LZMAError, zlib.error, csv.Error) as exc:
        raise ValueError("CSV input is corrupt or incomplete, or its header exceeds parser limits.") from exc
    if max_rows is not None and len(frame) > max_rows:
        raise ValueError("CSV input exceeds the configured row limit.")
    return frame


def escape_csv_formulas(frame):
    """Escape formula-like text in a copy; leave numeric cells unchanged."""
    def escape(value):
        if isinstance(value, str) and value.lstrip(" \t\r\n").startswith(("=", "+", "-", "@")):
            return "'" + value
        return value
    return frame.apply(lambda column: column.map(escape))


def _reject_duplicate_headers(csv_file, delimiter=",", encoding="utf-8"):
    """Inspect original UTF-8 CSV headers without taking ownership of streams."""
    def header_lines(source):
        # csv.reader needs physical lines even for StringIO objects whose default
        # iterator does not recognize CR-only records. Preserve quoted newlines.
        pending = ""
        first = True
        started = False
        while True:
            chars = []
            while True:
                char = pending or source.read(1)
                pending = ""
                if not char:
                    break
                if first:
                    first = False
                    if char == "\ufeff":
                        continue
                chars.append(char)
                if char == "\r":
                    following = source.read(1)
                    if following == "\n":
                        chars.append(following)
                    else:
                        pending = following
                    break
                if char == "\n":
                    break
            if not chars:
                return
            line = "".join(chars)
            if not started and not line.strip():
                continue
            started = True
            yield line

    if isinstance(csv_file, (str, os.PathLike)):
        if not Path(csv_file).is_file():
            return
        opener = {".gz": gzip.open, ".bz2": bz2.open, ".xz": lzma.open}.get(Path(csv_file).suffix, open)
        try:
            with opener(csv_file, mode="rt", encoding=encoding, newline="") as source:
                headers = next(csv.reader(header_lines(source), delimiter=delimiter), [])
        except (EOFError, lzma.LZMAError, zlib.error, csv.Error) as exc:
            raise ValueError("CSV input is corrupt or incomplete, or its header exceeds parser limits.") from exc
    else:
        if not all(hasattr(csv_file, name) for name in ("read", "seek", "tell")) or (hasattr(csv_file, "seekable") and not csv_file.seekable()):
            raise ValueError("CSV streams must be seekable for header validation.")
        position = csv_file.tell()
        binary = isinstance(csv_file.read(0), bytes)
        source = io.TextIOWrapper(csv_file, encoding=encoding, newline="") if binary else csv_file
        try:
            headers = next(csv.reader(header_lines(source), delimiter=delimiter), [])
        except (EOFError, lzma.LZMAError, zlib.error, csv.Error) as exc:
            raise ValueError("CSV input is corrupt or incomplete, or its header exceeds parser limits.") from exc
        finally:
            if binary:
                source.detach()
            csv_file.seek(position)
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
    balance_classes=False,
    top_n=None,
    id_column=None,
    min_probability=None,
    regularization_c=1.0,
    max_rows=None,
    encoding="utf-8",
):
    try:
        if isinstance(regularization_c, (bool, np.bool_)) or not isinstance(regularization_c, Real) or not np.isfinite(regularization_c) or not 1e-12 <= regularization_c <= 1000:
            raise ValueError("regularization_c must be finite from 1e-12 to 1000.")
        validate_priority_thresholds(medium_threshold, high_threshold)
        if min_probability is not None and (isinstance(min_probability, (bool, np.bool_)) or not isinstance(min_probability, Real) or not np.isfinite(min_probability) or not 0 <= min_probability <= 100):
            raise ValueError("min_probability must be a finite percentage from 0 to 100.")
        if top_n is not None and (not isinstance(top_n, int) or isinstance(top_n, bool) or top_n < 1):
            raise ValueError("top_n must be a positive integer.")
        if not isinstance(balance_classes, bool):
            raise ValueError("balance_classes must be a boolean.")
        if not isinstance(max_display_rows, int) or isinstance(max_display_rows, bool) or max_display_rows < 1:
            raise ValueError("max_display_rows must be a positive integer.")
        if id_column is not None and (not isinstance(id_column, str) or not id_column.strip() or id_column in set(FEATURE_COLUMNS) | {"converted", "priority", "conversion_probability"}):
            raise ValueError("Lead ID column must be a distinct source identifier.")
        id_options = {"converters": {id_column: lambda value: "__id__" + value}} if id_column is not None else {}
        df = _read_csv_input(csv_file, delimiter, **({"encoding": encoding} if encoding != "utf-8" else {}), **({"max_rows": max_rows} if max_rows is not None else {}), **(id_options if scoring_file is None else {}))
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

        model = make_pipeline(MaxAbsScaler(), StandardScaler(), LogisticRegression(max_iter=1000, **({"C": regularization_c} if regularization_c != 1.0 else {}), **({"class_weight": "balanced"} if balance_classes else {})))
        training_features = df[FEATURE_COLUMNS].copy()
        training_labels = df["converted"].astype(int).copy()
        if scoring_file is not None:
            candidates = _read_csv_input(scoring_file, delimiter, **({"encoding": encoding} if encoding != "utf-8" else {}), **({"max_rows": max_rows} if max_rows is not None else {}), **id_options)
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
        if id_column is not None:
            if id_column not in df.columns:
                raise ValueError("Lead ID column was not found in the scored input.")
            df[id_column] = df[id_column].astype("string").str.slice(6)
            identifiers = df[id_column].str.strip()
            if identifiers.isna().any() or identifiers.eq("").any() or identifiers.duplicated().any():
                raise ValueError("Scored lead IDs must be non-empty and unique.")
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", ConvergenceWarning)
                model.fit(training_features, training_labels)
        except ConvergenceWarning as exc:
            raise ValueError("Lead model did not converge; no scores were published.") from exc
        positive_class_index = list(model.classes_).index(1)
        df["conversion_probability"] = model.predict_proba(df[FEATURE_COLUMNS])[:, positive_class_index] * 100
        df["priority"] = df["conversion_probability"].apply(
            lambda probability: priority_tier(
                probability, medium_threshold=medium_threshold, high_threshold=high_threshold
            )
        )

        scored_lead_count = len(df)
        if sort_by_probability or top_n is not None:
            df = df.sort_values("conversion_probability", ascending=False, kind="stable")
        if min_probability is not None:
            df = df.loc[df["conversion_probability"] >= min_probability].copy()
        if top_n is not None and top_n < len(df):
            df = df.head(top_n).copy()
        df.attrs["training_population"] = {"samples": len(training_labels), "converted": int(training_labels.sum()), "not_converted": int((training_labels == 0).sum()), "conversion_rate_percent": float(training_labels.mean() * 100)}
        df.attrs["scored_lead_count"] = scored_lead_count

        print("\n===== Lead Conversion Scoring =====")
        if len(df) < scored_lead_count:
            print(f"Selected {len(df)} of {scored_lead_count} scored leads; exports follow this selection.")
        print(df[FEATURE_COLUMNS + ["conversion_probability", "priority"]].head(max_display_rows).to_string(index=False))
        if len(df) > max_display_rows:
            print(f"Showing {max_display_rows} of {len(df)} leads; exports include every selected lead.")
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
    payload = {"lead_count": len(result), "scored_lead_count": int(result.attrs.get("scored_lead_count", len(result))), "priority_counts": {tier: int((result["priority"] == tier).sum()) for tier in ("High", "Medium", "Low")},
               "average_conversion_probability": float(result["conversion_probability"].mean()) if len(result) else None}
    if "training_population" in result.attrs:
        payload["training_population"] = result.attrs["training_population"]
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
    parser.add_argument("--balance-classes", action="store_true", help="Use inverse-frequency training class weights for imbalanced labels")
    parser.add_argument("--top-n", type=int, help="Return and export only the N highest-scoring leads, with stable tie order")
    parser.add_argument("--id-column", help="Require unique non-empty scored lead identifiers, preserving literal text and leading zeros")
    parser.add_argument("--min-probability", type=float, help="Export only leads whose predicted conversion percentage meets this inclusive floor")
    parser.add_argument("--max-input-rows", type=int, help="Reject CSV files exceeding this raw row budget (1..10000000)")
    parser.add_argument("--regularization-c", type=float, default=1.0, help="Inverse regularization strength (1e-12..1000)")
    parser.add_argument("--encoding", choices=("utf-8", "utf-8-sig", "cp1252", "latin-1"), default="utf-8", help="Input CSV text encoding; output encoding remains UTF-8")
    args = parser.parse_args(argv)
    if args.max_input_rows is not None and not 1 <= args.max_input_rows <= 10_000_000:
        parser.error("--max-input-rows must be from 1 to 10000000")
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
    if args.id_column is not None:
        scoring_options["id_column"] = args.id_column
    if args.min_probability is not None:
        scoring_options["min_probability"] = args.min_probability
    if args.top_n is not None:
        scoring_options["top_n"] = args.top_n
    if args.balance_classes:
        scoring_options["balance_classes"] = True
    if args.regularization_c != 1.0:
        scoring_options["regularization_c"] = args.regularization_c
    if args.encoding != "utf-8":
        scoring_options["encoding"] = args.encoding
    if args.max_input_rows is not None:
        scoring_options["max_rows"] = args.max_input_rows
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
