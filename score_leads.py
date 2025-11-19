import pandas as pd
from sklearn.linear_model import LogisticRegression
import sys

def score_leads(csv_file):
    try:
        df = pd.read_csv(csv_file)

        required_columns = {"interaction_score", "visits", "total_spent", "converted"}
        if not required_columns.issubset(df.columns):
            print(f"❌ CSV must include: {required_columns}")
            return

        X = df[["interaction_score", "visits", "total_spent"]]
        y = df["converted"]

        model = LogisticRegression()
        model.fit(X, y)

        df["conversion_probability"] = model.predict_proba(X)[:, 1] * 100

        print("\n===== Lead Conversion Scoring =====")
        print(df[["interaction_score", "visits", "total_spent", "conversion_probability"]])
        print("===================================\n")

    except Exception as e:
        print(f"❌ Error: {e}")

def main():
    if len(sys.argv) != 2:
        print("Usage: python score_leads.py <csv_file>")
        return

    score_leads(sys.argv[1])

if __name__ == "__main__":
    main()
