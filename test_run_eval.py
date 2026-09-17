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
    score_response,
)


class RunEvalTests(unittest.TestCase):
    def test_single_query_eval_exit_status_reflects_results(self):
        import run_eval
        for results, expected in [([], 1), ([{"status": "review"}], 1),
                                  ([{"status": "fail"}], 1), ([{"status": "pass"}], 0)]:
            with self.subTest(results=results), patch("sys.argv", [
                "run_eval.py", "--live-url", "http://example.invalid/query"
            ]), patch.object(run_eval, "load_eval_rows", return_value=[{}]), patch.object(
                run_eval, "run_live_eval", return_value=results
            ), patch.object(run_eval, "render_results", return_value="report"), patch.object(
                Path, "write_text"
            ):
                self.assertEqual(run_eval.main(), expected)

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

        self.assertGreaterEqual(len(conversations), 20)
        self.assertGreaterEqual(
            sum(len(conversation["turns"]) for conversation in conversations),
            60,
        )
        self.assertGreaterEqual(len(conversations[0]["turns"]), 2)
        self.assertEqual(conversations[0]["turns"][0]["expected_route"], "data_query")
        self.assertEqual(
            [conversation["id"] for conversation in conversations[:10]],
            [f"conv-{index:03d}" for index in range(1, 11)],
        )
        covered = {tag for conversation in conversations for tag in conversation["tags"]}
        self.assertTrue({
            "model-description", "data", "plot", "links", "navigation",
            "clarification", "typo", "failed-scope-recovery", "follow-up",
        }.issubset(covered))

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
                "tags": ["data", "follow-up"],
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
        self.assertEqual(results[0]["tags"], ["data", "follow-up"])
        self.assertTrue(results[0]["turns"][0]["scores"]["session_continuity"])
        self.assertTrue(results[0]["turns"][1]["scores"]["session_continuity"])
        self.assertEqual(mocked_post.call_args_list[1].kwargs["session_id"], "session-1")

    def test_render_conversation_results_includes_live_scores(self):
        conversations = [
            {
                "id": "conv-test",
                "title": "Session continuity",
                "tags": ["data", "follow-up"],
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
        self.assertIn("`data`: 1 conversation", rendered)
        self.assertIn("`follow-up`: 1 conversation", rendered)
        self.assertIn("conv-test: Session continuity", rendered)

    def test_conversation_eval_gate_requires_all_pass(self):
        self.assertTrue(conversation_eval_is_green([{"status": "pass"}, {"status": "pass"}]))
        self.assertFalse(conversation_eval_is_green([{"status": "pass"}, {"status": "review"}]))
        self.assertFalse(conversation_eval_is_green([]))

    def test_structured_entity_mismatch_is_not_hidden_by_answer_text(self):
        row = {
            "expected_route": "data_query",
            "expected_variable": "Emissions|CO2",
            "expected_region": "EU",
            "expected_scenario": "",
            "expected_model": "",
            "useful_clarification": "no",
            "useful_link": "",
            "no_hallucinated_data": "no",
        }
        response = {
            "route": {"agent": "data_query"},
            "entities": {"variable": "Emissions|CH4", "region": "EU"},
            "answer": "You asked for Emissions|CO2, but here are methane results.",
        }

        scores = score_response(row, response)

        self.assertFalse(scores["correct_variable"])

    def test_optional_year_and_plot_expectations_catch_semantic_drift(self):
        row = {
            "expected_route": "data_plotting",
            "expected_variable": "Emissions|CO2",
            "expected_region": "World",
            "expected_scenario": "Baseline",
            "expected_model": "",
            "useful_clarification": "no",
            "useful_link": "",
            "no_hallucinated_data": "no",
            "expected_start_year": "2031",
            "expected_end_year": "none",
            "expected_action": "plot",
            "expected_chart_type": "line",
            "expected_plot": "yes",
        }
        response = {
            "route": {"agent": "data_plotting"},
            "entities": {
                "variable": "Emissions|CO2", "region": "World",
                "scenario": "Baseline", "start_year": 2030,
                "end_year": 2030, "action": "plot", "chart_type": "line",
            },
            "answer": "Plot for the requested scope.",
            "plot_base64": "encoded-image",
        }

        scores = score_response(row, response)

        self.assertFalse(scores["correct_start_year"])
        self.assertFalse(scores["correct_end_year"])
        self.assertTrue(scores["correct_action"])
        self.assertTrue(scores["correct_chart_type"])
        self.assertTrue(scores["correct_plot"])

    def test_optional_exact_link_url_is_scored(self):
        row = {
            "expected_route": "site_navigation",
            "expected_variable": "", "expected_region": "",
            "expected_scenario": "", "expected_model": "",
            "useful_clarification": "no", "useful_link": "",
            "no_hallucinated_data": "no",
            "expected_link_url": "https://iamparis.eu/models",
        }
        response = {
            "route": {"agent": "site_navigation"},
            "entities": {},
            "answer": "Open the model directory.",
            "relevant_links": [
                {"title": "Models", "url": "https://iamparis.eu/models"}
            ],
        }

        self.assertTrue(score_response(row, response)["correct_link_url"])

    def test_optional_no_plot_expectation_rejects_stale_plot_payload(self):
        row = {
            "expected_route": "data_query",
            "expected_variable": "Emissions|CO2",
            "expected_region": "World",
            "expected_scenario": "",
            "expected_model": "",
            "useful_clarification": "no",
            "useful_link": "",
            "no_hallucinated_data": "no",
            "expected_plot": "no",
        }
        response = {
            "route": {"agent": "data_query"},
            "entities": {"variable": "Emissions|CO2", "region": "World"},
            "answer": "That result includes data from several models.",
            "plot_base64": "stale-image-from-the-previous-turn",
        }

        self.assertFalse(score_response(row, response)["correct_plot"])

    def test_optional_region_count_catches_broadened_followup_scope(self):
        row = {
            "expected_route": "data_plotting",
            "expected_variable": "Primary Energy|Gas",
            "expected_region": "China",
            "expected_scenario": "",
            "expected_model": "",
            "useful_clarification": "no",
            "useful_link": "",
            "no_hallucinated_data": "no",
            "expected_region_count": "1",
        }
        response = {
            "route": {"agent": "data_plotting"},
            "entities": {
                "variable": "Primary Energy|Gas",
                "regions": ["CHN", "IND", "World"],
            },
            "answer": "Showing gas for several regions.",
        }

        scores = score_response(row, response)

        self.assertTrue(scores["correct_region"])
        self.assertFalse(scores["correct_region_count"])


if __name__ == "__main__":
    unittest.main()
