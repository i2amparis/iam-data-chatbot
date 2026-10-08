import logging
import os
import tempfile
import unittest
from unittest.mock import patch

from main import IAMParisBot, docs_from_records, load_workspace_codes


class _Response:
    status_code = 200

    def __init__(self, rows):
        self._rows = rows

    def raise_for_status(self):
        return None

    def json(self):
        return self._rows


class FetchAllResultsTests(unittest.TestCase):
    def test_cache_isolated_by_endpoint(self):
        bot = IAMParisBot.__new__(IAMParisBot)
        bot.logger = logging.getLogger("FetchAllResultsTests")
        previous = os.getcwd()
        with tempfile.TemporaryDirectory() as directory:
            try:
                os.chdir(directory)
                with patch("main.requests.get", side_effect=[
                    _Response([{"value": 100}]), _Response([{"value": 999}])
                ]) as get:
                    first = bot.fetch_json("https://first.example/results")
                    second = bot.fetch_json("https://second.example/results")
                    again = bot.fetch_json("https://first.example/results")
                self.assertEqual(get.call_count, 2)
                self.assertEqual(first, again)
                self.assertEqual(second, [{"value": 999}])
            finally:
                os.chdir(previous)

    def _aged_cache_bot(self, directory):
        bot = IAMParisBot.__new__(IAMParisBot)
        bot.logger = logging.getLogger("FetchAllResultsTests")
        with patch("main.requests.get", return_value=_Response([{"value": 1}])):
            bot.fetch_json("https://aged.example/models")
        cache_file = bot.last_fetch_cache_file
        two_days_ago = os.path.getmtime(cache_file) - 48 * 3600
        os.utime(cache_file, (two_days_ago, two_days_ago))
        return bot

    def test_cache_older_than_max_age_is_refreshed(self):
        with tempfile.TemporaryDirectory() as directory:
            previous = os.getcwd()
            try:
                os.chdir(directory)
                bot = self._aged_cache_bot(directory)
                with patch.dict(os.environ, {"IAM_CACHE_MAX_AGE_HOURS": "24"}), patch(
                    "main.requests.get", return_value=_Response([{"value": 2}])
                ) as get:
                    rows = bot.fetch_json("https://aged.example/models")
                self.assertEqual(get.call_count, 1)
                self.assertEqual(rows, [{"value": 2}])
            finally:
                os.chdir(previous)

    def test_failed_refresh_of_aged_cache_serves_existing_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            previous = os.getcwd()
            try:
                os.chdir(directory)
                bot = self._aged_cache_bot(directory)
                with patch.dict(os.environ, {"IAM_CACHE_MAX_AGE_HOURS": "24"}), patch(
                    "main.requests.get", side_effect=RuntimeError("API down")
                ):
                    rows = bot.fetch_json("https://aged.example/models", max_retries=1)
                self.assertEqual(rows, [{"value": 1}])
            finally:
                os.chdir(previous)

    def test_cache_without_max_age_is_never_refreshed(self):
        with tempfile.TemporaryDirectory() as directory:
            previous = os.getcwd()
            try:
                os.chdir(directory)
                bot = self._aged_cache_bot(directory)
                with patch.dict(os.environ, {"IAM_CACHE_MAX_AGE_HOURS": "0"}), patch(
                    "main.requests.get"
                ) as get:
                    rows = bot.fetch_json("https://aged.example/models")
                get.assert_not_called()
                self.assertEqual(rows, [{"value": 1}])
            finally:
                os.chdir(previous)

    def test_force_refresh_replaces_the_normal_request_cache(self):
        bot = IAMParisBot.__new__(IAMParisBot)
        bot.logger = logging.getLogger("FetchAllResultsTests")
        payload = {"workspace_code": ["one"], "limit": -1}
        old = [{"resultId": "one", "years": {"2050": 100}}]
        new = [{"resultId": "one", "years": {"2050": 200}}]

        with tempfile.TemporaryDirectory() as directory:
            previous = os.getcwd()
            try:
                os.chdir(directory)
                with patch("main.requests.post", return_value=_Response(old)):
                    first = bot.fetch_json("https://api.example.test/results", payload=payload)
                with patch("main.requests.post", return_value=_Response(new)):
                    refreshed = bot.fetch_json(
                        "https://api.example.test/results",
                        payload={**payload, "_force_refresh": True},
                    )
                with patch("main.requests.post") as post:
                    restarted = bot.fetch_json(
                        "https://api.example.test/results", payload=payload
                    )
            finally:
                os.chdir(previous)

        self.assertEqual(first[0]["years"]["2050"], 100)
        self.assertEqual(refreshed[0]["years"]["2050"], 200)
        self.assertEqual(restarted[0]["years"]["2050"], 200)
        post.assert_not_called()

    def test_numeric_model_identifier_is_not_added_to_semantic_documents(self):
        documents = docs_from_records([
            {"modelName": "42", "description": "Source-only identifier"},
            {"modelName": "GCAM", "description": "Grounded model description"},
        ])

        self.assertEqual(len(documents), 1)
        self.assertEqual(documents[0].metadata["modelName"], "GCAM")

    def test_workspace_codes_are_loaded_from_configuration(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "workspaces.json")
            with open(path, "w") as handle:
                handle.write('{"workspaces": ["one", "two", "one"]}')

            self.assertEqual(load_workspace_codes(path), ["one", "two"])

    def test_fetch_all_pages_for_every_workspace(self):
        calls = []

        def fake_post(_url, json, timeout):
            del timeout
            workspace = json["workspace_code"][0]
            page = json["offset"]
            calls.append((workspace, page, json["limit"]))
            if page == 0:
                rows = [
                    {"resultId": f"{workspace}-{index}", "workspace_code": workspace}
                    for index in range(1000)
                ]
            elif page == 1:
                rows = [{"resultId": f"{workspace}-1000", "workspace_code": workspace}]
            else:
                rows = []
            return _Response(rows)

        bot = IAMParisBot.__new__(IAMParisBot)
        bot.logger = logging.getLogger("FetchAllResultsTests")
        with tempfile.TemporaryDirectory() as directory, patch(
            "main.requests.post", side_effect=fake_post
        ):
            previous = os.getcwd()
            try:
                os.chdir(directory)
                rows = bot.fetch_json(
                    "https://api.example.test/results",
                    payload={"workspace_code": ["one", "two"], "limit": -1},
                    cache=False,
                )
            finally:
                os.chdir(previous)

        self.assertEqual(len(rows), 2002)
        self.assertEqual(
            calls,
            [("one", 0, 1000), ("one", 1, 1000), ("two", 0, 1000), ("two", 1, 1000)],
        )


if __name__ == "__main__":
    unittest.main()
