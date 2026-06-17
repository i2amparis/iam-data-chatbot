import unittest

from frontend_response_audit import audit_response, audit_results, render_markdown


class FrontendResponseAuditTests(unittest.TestCase):
    def test_valid_structured_response_has_no_issues(self):
        issues = audit_response(
            {
                "id": "row-1",
                "query": "where can I find Climate Watch",
                "answer": "Use these links.",
                "route": {"agent": "general_qa"},
                "links": [
                    {
                        "title": "Climate Watch",
                        "url": "https://iamparis.eu/application_library",
                        "action": "search",
                        "search_hint": "Climate Watch",
                    }
                ],
                "data_provenance": {"display_rows": []},
            }
        )

        self.assertEqual(issues, [])

    def test_search_link_requires_search_hint(self):
        issues = audit_response(
            {
                "id": "row-2",
                "query": "Climate Watch",
                "answer": "Use this link.",
                "links": [
                    {
                        "title": "Climate Watch",
                        "url": "https://iamparis.eu/application_library",
                        "action": "search",
                    }
                ],
            }
        )

        self.assertTrue(any(issue.field == "links[0].search_hint" for issue in issues))

    def test_plot_payload_requires_caption(self):
        issues = audit_response(
            {
                "id": "row-3",
                "query": "plot it",
                "answer": "Showing plot.",
                "plot_base64": "abc123",
            }
        )

        self.assertTrue(any(issue.field == "plot_caption" for issue in issues))

    def test_render_markdown_reports_pass(self):
        rendered = render_markdown([], source="docs/evaluation_live_results.json", total_rows=2)

        self.assertIn("Status: pass", rendered)
        self.assertIn("Responses checked: 2", rendered)

    def test_audit_results_checks_all_rows(self):
        issues = audit_results([
            {"id": "ok", "answer": "hello"},
            {"id": "bad", "answer": "", "links": [{"action": "jump"}]},
        ])

        self.assertGreaterEqual(len(issues), 2)


if __name__ == "__main__":
    unittest.main()
