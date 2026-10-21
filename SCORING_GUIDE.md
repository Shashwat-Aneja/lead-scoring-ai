Lead scoring guide

Required numeric features are `interaction_score`, `visits`, and `total_spent`; `converted` must contain both binary labels 0 and 1. Values must be finite and non-empty.

Run `python score_leads.py sample_leads.csv` after installing `requirements.txt`.

The output probability is an estimate from logistic regression, not a guaranteed outcome. Scores computed on training records are illustrative, not a valid performance evaluation. For deployment, evaluate on a separate time-aware holdout and monitor calibration and feature drift.