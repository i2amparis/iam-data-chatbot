import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from feedback_review import (
    dedupe_feedback_candidates,
    load_feedback_candidates,
    to_eval_row,
    write_feedback_report,
)


class FeedbackReviewTests(unittest.TestCase):
    def test_load_dedupe_and_export_feedback_candidates(self):
        with TemporaryDirectory() as tmpdir:
            source = Path(tmpdir) / "feedback.jsonl"
            report = Path(tmpdir) / "review.md"
            csv_output = Path(tmpdir) / "feedback.csv"
            rows = [
                {
                    "timestamp": "2026-06-03T10:00:00+00:00",
                    "query": "missing CO2 data",
                    "route": "data_query",
                    "route_confidence": 0.9,
                    "entities": {"variable": "Emissions|CO2"},
                    "no_data_reason": "scenario combination unavailable",
                },
                {
                    "timestamp": "2026-06-03T11:00:00+00:00",
                    "query": "missing CO2 data",
                    "route": "data_query",
                    "route_confidence": 0.9,
                    "entities": {"variable": "Emissions|CO2", "region": "World"},
                    "no_data_reason": "scenario combination unavailable",
                },
            ]
            source.write_text("\n".join(json.dumps(row) for row in rows) + "\n")

            loaded = load_feedback_candidates(source)
            deduped = dedupe_feedback_candidates(loaded)
            _, eval_rows = write_feedback_report(source, report, csv_output)

            self.assertEqual(len(deduped), 1)
            self.assertEqual(deduped[0]["entities"]["region"], "World")
            self.assertEqual(eval_rows[0]["query"], "missing CO2 data")
            self.assertEqual(eval_rows[0]["expected_region"], "World")
            self.assertIn("missing CO2 data", report.read_text())
            self.assertIn("feedback-001", csv_output.read_text())

    def test_to_eval_row_infers_application_link_category(self):
        row = {
            "query": "where can I find Climate Watch",
            "route": "general_qa",
            "route_confidence": 0.8,
            "entities": {},
        }

        eval_row = to_eval_row(row, 1)

        self.assertEqual(eval_row["expected_route"], "general_qa")
        self.assertEqual(eval_row["useful_link"], "application_library")

    def test_to_eval_row_promotes_model_profile_questions(self):
        row = {
            "query": "tell me about WITCH model",
            "route": "data_query",
            "route_confidence": 0.45,
            "entities": {},
        }

        eval_row = to_eval_row(row, 1)

        self.assertEqual(eval_row["expected_route"], "model_explanation")
        self.assertEqual(eval_row["expected_model"], "WITCH")
        self.assertEqual(eval_row["useful_link"], "models")
        self.assertEqual(eval_row["useful_clarification"], "no")


if __name__ == "__main__":
    unittest.main()
