# Lead Scoring AI

A machine learning model that predicts the probability of a lead converting into a customer using historical business data.

## Features
- Logistic Regression binary classification
- Conversion likelihood as a percentage from 0 to 100
- Numeric validation for feature columns and conversion labels
- Clear validation errors for missing, non-finite, or single-class training data

## Input CSV
The CSV must include these columns:

```csv
interaction_score,visits,total_spent,converted
80,10,12000,1
30,4,3000,0
60,7,7000,1
10,2,1000,0
90,12,20000,1
```

The `converted` column must contain both labels, `0` and `1`. Feature values and labels must be numeric and finite.

## Run
```bash
pip install pandas scikit-learn
python score_leads.py data.csv
```

The output includes each lead's input features and predicted conversion probability. These are model estimates, not guarantees of an individual lead's behavior.

## Project files
- `score_leads.py`: validation, model training, and probability output
- `test_score_leads.py`: regression tests for valid and invalid inputs

Developed by **Shashwat Aneja**.

Lead features are standardized inside the fitted model pipeline. Currency unit changes no longer dominate the model solely through their numeric scale.

Use `TRAIN.csv --score-csv NEW_LEADS.csv --output scored.csv` to fit on labeled history and score separate unlabeled leads. Outputs cannot overwrite either input. Without `--score-csv`, reported probabilities describe the training rows and are not out-of-sample accuracy measurements.

Use `--max-display-rows` to control the console preview (default 20). Every scored lead is retained in returned data and CSV exports.

Export paths are validated before model fitting. Input aliases and inaccessible file identities fail early rather than training a model that cannot be safely exported.

Boolean feature values are rejected in training and candidate CSVs instead of silently treating true/false as money, visits or interaction scores. Binary conversion labels remain supported.

Use `--sort-by-probability` to rank the preview and complete export by descending conversion probability. Equal scores retain their input order.

`--summary-output summary.json` atomically exports the scored lead count, all three priority counts and mean conversion probability. It cannot alias training/scoring inputs or the scored CSV output. The mean is a model estimate, not a calibrated accuracy metric.

Use `--escape-formulas` when CSV exports will be opened in spreadsheet software. Formula-like text cells beginning with `=`, `+`, `-` or `@` after leading whitespace receive an apostrophe; numeric cells and in-memory calculations remain unchanged. The default export preserves original text.

`--delimiter ";"`, `--delimiter tab` and `--delimiter "|"` support alternate input CSV separators, including quoted fields. Duplicate-header validation uses the selected separator. The default input and all CSV exports remain comma-separated.

`--balance-classes` enables inverse-frequency class weights in logistic-regression training. The default remains unweighted. Weighted-model scores can shift and are not guaranteed calibrated probabilities; compare outcomes with held-out data before using them for decisions.

Seekable caller-owned text and UTF-8 binary CSV streams now receive the same duplicate-header checks as file paths, before pandas can rename ambiguous columns. Validation restores the starting position and leaves caller streams open; nonseekable stream objects are rejected explicitly.

`--top-n N` selects the highest-scoring N leads for returned data and CSV output, with stable ties. All candidate leads are still scored; the preview and JSON summary disclose selected versus scored counts. Default output still includes every lead.
