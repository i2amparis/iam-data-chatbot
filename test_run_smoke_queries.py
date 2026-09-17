import unittest

from run_smoke_queries import (
    _extract_queries_from_md,
    _extract_query_cases_from_md,
    _row,
    _smoke_status,
)


class SmokeQueryClassificationTests(unittest.TestCase):
    def test_markdown_headings_are_preserved_as_query_categories(self):
        markdown = """# Suite
## Model descriptions
- [ ] `what is GCAM?`
## Plots
1. [ ] `plot emissions`
"""

        cases = _extract_query_cases_from_md(markdown)

        self.assertEqual(
            cases,
            [
                {"query": "what is GCAM?", "category": "Model descriptions"},
                {"query": "plot emissions", "category": "Plots"},
            ],
        )
        self.assertEqual(_extract_queries_from_md(markdown), ["what is GCAM?", "plot emissions"])

    def test_level_three_headings_define_followup_conversation_groups(self):
        markdown = """## Follow-ups
### Block A — region switch
1. [ ] `CO2 for Europe`
2. [ ] `same for China`
### Block B — model pronoun
1. [ ] `describe IMAGE`
2. [ ] `what does it cover?`
"""

        cases = _extract_query_cases_from_md(markdown)

        self.assertEqual(
            [case.get("conversation_group") for case in cases],
            [
                "Block A — region switch",
                "Block A — region switch",
                "Block B — model pronoun",
                "Block B — model pronoun",
            ],
        )

    def test_requested_plot_with_grounded_no_data_is_not_reported_failed(self):
        row = _row(
            "plot carbon price",
            {"answer": "I can't combine these series on one axis.", "route": {}},
        )

        self.assertTrue(row["no_data"])
        self.assertEqual(row["status"], "PLOT_NO_DATA")

    def test_requested_plot_without_plot_or_no_data_explanation_is_failed(self):
        row = _row(
            "plot carbon price",
            {"answer": "Something went wrong while making the chart.", "route": {}},
        )

        self.assertFalse(row["no_data"])
        self.assertEqual(row["status"], "PLOT_FAILED")

    def test_clarification_is_distinct_from_success(self):
        row = _row(
            "electricity",
            {"answer": "Choose the variable: 1. Secondary Energy|Electricity", "route": {}},
        )

        self.assertTrue(row["clarification"])
        self.assertEqual(_smoke_status(row), "CLARIFICATION")

    def test_actual_plot_wins_over_plot_request(self):
        row = _row(
            "draw emissions",
            {"answer": "Showing emissions", "plot_base64": "abc", "route": {}},
        )

        self.assertEqual(row["status"], "PLOT")
        self.assertEqual(row["answer"], "Showing emissions")


if __name__ == "__main__":
    unittest.main()
