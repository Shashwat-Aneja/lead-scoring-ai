import tempfile
import unittest
from pathlib import Path

import pandas as pd

from score_leads import score_leads


class LeadScoringValidationTests(unittest.TestCase):
    def csv_path(self, content):
        handle = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False)
        handle.write(content)
        handle.close()
        self.addCleanup(Path(handle.name).unlink, missing_ok=True)
        return handle.name

    def test_scores_are_bounded_between_zero_and_one_hundred(self):
        path = self.csv_path(
            "interaction_score,visits,total_spent,converted\n"
            "1,2,10,0\n2,4,20,0\n8,10,100,1\n9,12,150,1\n"
        )
        result = score_leads(path)
        self.assertIsNotNone(result)
        self.assertTrue(result["conversion_probability"].between(0, 100).all())

    def test_scored_leads_include_valid_priority_tiers(self):
        path = self.csv_path(
            "interaction_score,visits,total_spent,converted\n"
            "1,2,10,0\n2,4,20,0\n8,10,100,1\n9,12,150,1\n"
        )
        result = score_leads(path)
        self.assertIsNotNone(result)
        self.assertIn("priority", result.columns)
        self.assertTrue(set(result["priority"]).issubset({"Low", "Medium", "High"}))

    def test_single_class_training_data_is_rejected(self):
        path = self.csv_path(
            "interaction_score,visits,total_spent,converted\n"
            "1,2,10,0\n2,4,20,0\n"
        )
        self.assertIsNone(score_leads(path))

    def test_invalid_conversion_labels_are_rejected(self):
        path = self.csv_path(
            "interaction_score,visits,total_spent,converted\n"
            "1,2,10,0\n2,4,20,2\n"
        )
        self.assertIsNone(score_leads(path))


    def test_priority_tier_boundaries(self):
        from score_leads import priority_tier

        self.assertEqual(priority_tier(39.99), "Low")
        self.assertEqual(priority_tier(40), "Medium")
        self.assertEqual(priority_tier(69.99), "Medium")
        self.assertEqual(priority_tier(70), "High")

    def test_priority_tier_rejects_out_of_range_probability(self):
        from score_leads import priority_tier

        with self.assertRaisesRegex(ValueError, "between 0 and 100"):
            priority_tier(101)

if __name__ == "__main__":
    unittest.main()
