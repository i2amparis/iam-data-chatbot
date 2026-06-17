import unittest
from pathlib import Path
from urllib import error

from validate_links import (
    check_url,
    render_markdown,
    unique_catalog_urls,
    validate_catalog_links,
)


class FakeResponse:
    def __init__(self, status=200, url="https://iamparis.eu/results"):
        self.status = status
        self._url = url

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def getcode(self):
        return self.status

    def geturl(self):
        return self._url


class FakeOpener:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def __call__(self, req, timeout=10):
        self.calls.append((req.full_url, req.get_method(), timeout))
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class ValidateLinksTests(unittest.TestCase):
    def test_unique_catalog_urls_deduplicates_and_filters_domain(self):
        catalog = [
            {"title": "Results", "url": "https://iamparis.eu/results", "category": "results"},
            {"title": "Results duplicate", "url": "https://iamparis.eu/results", "category": "results"},
            {"title": "External", "url": "https://example.org", "category": "main"},
        ]

        rows = unique_catalog_urls(catalog, domain="iamparis.eu")

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["url"], "https://iamparis.eu/results")
        self.assertIn("Results duplicate", rows[0]["titles"])

    def test_check_url_ok_from_head(self):
        opener = FakeOpener([FakeResponse(200, "https://iamparis.eu/results")])

        result = check_url("https://iamparis.eu/results", timeout=3, opener=opener)

        self.assertEqual(result.status, "ok")
        self.assertEqual(result.http_status, 200)
        self.assertEqual(result.method, "HEAD")
        self.assertEqual(opener.calls[0][1], "HEAD")

    def test_check_url_falls_back_to_get_when_head_not_allowed(self):
        opener = FakeOpener([
            error.HTTPError("https://iamparis.eu/results", 405, "Method Not Allowed", {}, None),
            FakeResponse(200, "https://iamparis.eu/results"),
        ])

        result = check_url("https://iamparis.eu/results", opener=opener)

        self.assertEqual(result.status, "ok")
        self.assertEqual(result.method, "GET")
        self.assertEqual([call[1] for call in opener.calls], ["HEAD", "GET"])

    def test_check_url_marks_404_broken(self):
        opener = FakeOpener([
            error.HTTPError("https://iamparis.eu/missing", 404, "Not Found", {}, None),
        ])

        result = check_url("https://iamparis.eu/missing", opener=opener)

        self.assertEqual(result.status, "broken")
        self.assertEqual(result.http_status, 404)

    def test_validate_catalog_links_attaches_titles_and_categories(self):
        catalog = [
            {"title": "Results", "url": "https://iamparis.eu/results", "category": "results"},
        ]
        opener = FakeOpener([FakeResponse(200, "https://iamparis.eu/results")])

        results = validate_catalog_links(catalog, opener=opener)

        self.assertEqual(results[0].titles, ["Results"])
        self.assertEqual(results[0].categories, ["results"])

    def test_render_markdown_summarizes_statuses(self):
        catalog = [
            {"title": "Results", "url": "https://iamparis.eu/results", "category": "results"},
            {"title": "Missing", "url": "https://iamparis.eu/missing", "category": "main"},
        ]
        opener = FakeOpener([
            error.HTTPError("https://iamparis.eu/missing", 404, "Not Found", {}, None),
            FakeResponse(200, "https://iamparis.eu/results"),
        ])

        results = validate_catalog_links(catalog, opener=opener)
        markdown = render_markdown(results, catalog_path=Path("catalog.json"))

        self.assertIn("Unique URLs checked: 2", markdown)
        self.assertIn("`ok`: 1", markdown)
        self.assertIn("`broken`: 1", markdown)
        self.assertIn("https://iamparis.eu/missing", markdown)


if __name__ == "__main__":
    unittest.main()
