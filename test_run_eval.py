import unittest
from pathlib import Path
from unittest.mock import patch

from run_eval import (
    MIN_HOLDOUT_QUERIES,
    MIN_FEEDBACK_QUERIES,
    _has_useful_clarification,
    _has_useful_link,
    conversation_eval_is_green,
    load_conversations,
    load_eval_rows,
    render_conversation_results,
    run_live_conversation_eval,
)


class RunEvalTests(unittest.TestCase):
    def test_datastories_url_counts_as_data_stories_link(self):
        response = {
            "relevant_links": [
                {
                    "title": "Policy Catalogue Interactive Explorer",
                    "url": "https://iamparis.eu/datastories/policyCatalog",
                    "reason": "Matched: policy catalogue",
                    "search_hint": "",
                }
            ]
        }

        self.assertTrue(_has_useful_link(response, "data_stories"))

    def test_low_confidence_model_question_counts_as_useful_clarification(self):
        response = {
            "answer": "I matched REMIND as the model, but confidence is low. Which model should I use?"
        }

        self.assertTrue(_has_useful_clarification(response))

    def test_link_reason_does_not_count_as_clarification(self):
        response = {
            "answer": (
                "There are 74 models available.\n\n"
                "Relevant IAM PARIS links:\n"
                "- [SDG Model Coverage Matrix](https://iamparis.eu/models/sdg) - "
                "Matched: Models, Use when user asks which model covers which SDG/topic"
            )
        }

        self.assertFalse(_has_useful_clarification(response))

    def test_instructional_choose_tool_does_not_count_as_clarification(self):
        response = {
            "answer": "Aqueduct offers several tools. Choose the tool that suits your needs."
        }

        self.assertFalse(_has_useful_clarification(response))

    def test_load_conversation_fixture(self):
        conversations = load_conversations(Path("eval_conversations.json"))

        self.assertGreaterEqual(len(conversations), 10)
        self.assertGreaterEqual(len(conversations[0]["turns"]), 2)
        self.assertEqual(conversations[0]["turns"][0]["expected_route"], "data_query")

    def test_load_holdout_eval_fixture(self):
        rows = load_eval_rows(Path("eval_holdout_queries.csv"), min_queries=MIN_HOLDOUT_QUERIES)

        self.assertGreaterEqual(len(rows), 50)
        self.assertTrue(any(row["query"] == "electricity" for row in rows))
        self.assertTrue(any(row["query"] == "global impacts of NDCs" for row in rows))

    def test_load_feedback_eval_fixture_when_present(self):
        rows = load_eval_rows(Path("docs/eval_feedback_candidates.csv"), min_queries=MIN_FEEDBACK_QUERIES)

        self.assertGreaterEqual(len(rows), 1)
        self.assertIn("expected_route", rows[0])

    def test_live_conversation_eval_scores_session_continuity(self):
        conversations = [
            {
                "id": "conv-test",
                "title": "Session continuity",
                "turns": [
                    {
                        "id": "conv-test.1",
                        "query": "show emissions for EU",
                        "expected_route": "data_query",
                        "expected_variable": "Emissions|CO2",
                        "expected_region": "EU",
                        "expected_scenario": "",
                        "expected_model": "",
                        "useful_clarification": "no",
                        "useful_link": "results",
                        "no_hallucinated_data": "yes",
                    },
                    {
                        "id": "conv-test.2",
                        "query": "same for China",
                        "expected_route": "data_query",
                        "expected_variable": "Emissions|CO2",
                        "expected_region": "China",
                        "expected_scenario": "",
                        "expected_model": "",
                        "useful_clarification": "no",
                        "useful_link": "results",
                        "no_hallucinated_data": "yes",
                    },
                ],
            }
        ]
        responses = [
            {
                "session_id": "session-1",
                "answer": "### Emissions|CO2 in EU",
                "route": {"agent": "data_query"},
                "entities": {"variable": "Emissions|CO2", "region": "EU"},
                "relevant_links": [{"title": "Results", "url": "https://iamparis.eu/results"}],
            },
            {
                "session_id": "session-1",
                "answer": "### Emissions|CO2 in China",
                "route": {"agent": "data_query"},
                "entities": {"variable": "Emissions|CO2", "region": "CHN"},
                "relevant_links": [{"title": "Results", "url": "https://iamparis.eu/results"}],
            },
        ]

        with patch("run_eval.post_query", side_effect=responses) as mocked_post:
            results = run_live_conversation_eval(conversations, "http://test/query")

        self.assertEqual(results[0]["status"], "pass")
        self.assertTrue(results[0]["turns"][0]["scores"]["session_continuity"])
        self.assertTrue(results[0]["turns"][1]["scores"]["session_continuity"])
        self.assertEqual(mocked_post.call_args_list[1].kwargs["session_id"], "session-1")

    def test_render_conversation_results_includes_live_scores(self):
        conversations = [
            {
                "id": "conv-test",
                "title": "Session continuity",
                "turns": [
                    {
                        "id": "conv-test.1",
                        "query": "show emissions for EU",
                        "expected_route": "data_query",
                        "expected_variable": "Emissions|CO2",
                        "expected_region": "EU",
                        "expected_scenario": "",
                        "expected_model": "",
                        "useful_clarification": "no",
                        "useful_link": "results",
                        "no_hallucinated_data": "yes",
                    }
                ],
            }
        ]
        live_results = [
            {
                "id": "conv-test",
                "title": "Session continuity",
                "status": "pass",
                "turns": [
                    {
                        "id": "conv-test.1",
                        "status": "pass",
                        "query": "show emissions for EU",
                        "route": {"agent": "data_query"},
                        "scores": {"correct_route": True, "session_continuity": True},
                        "answer_preview": "### Emissions|CO2 in EU",
                    }
                ],
            }
        ]

        rendered = render_conversation_results(conversations, live_results)

        self.assertIn("Live `pass` conversations: 1/1", rendered)
        self.assertIn("`session_continuity`: 1/1", rendered)
        self.assertIn("conv-test: Session continuity", rendered)

    def test_conversation_eval_gate_requires_all_pass(self):
        self.assertTrue(conversation_eval_is_green([{"status": "pass"}, {"status": "pass"}]))
        self.assertFalse(conversation_eval_is_green([{"status": "pass"}, {"status": "review"}]))
        self.assertFalse(conversation_eval_is_green([]))


if __name__ == "__main__":
    unittest.main()
