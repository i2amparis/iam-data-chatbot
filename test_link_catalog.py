import json
import tempfile
import unittest
from pathlib import Path

from link_catalog import build_link_catalog, read_xlsx_tables, write_link_catalog


class LinkCatalogTests(unittest.TestCase):
    def test_reads_expected_workbook_sheets(self):
        tables = read_xlsx_tables(Path("iamparis_chatbot_links.xlsx"))

        self.assertIn("01_Main_Routes", tables)
        self.assertIn("03_Results", tables)
        self.assertIn("04_Models", tables)
        self.assertIn("05_App_Library", tables)
        self.assertEqual(len(tables["04_Models"]), 68)
        self.assertEqual(len(tables["05_App_Library"]), 144)

    def test_builds_catalog_with_direct_and_fallback_links(self):
        catalog = build_link_catalog(Path("iamparis_chatbot_links.xlsx"))
        by_title = {item["title"]: item for item in catalog}

        self.assertGreaterEqual(len(catalog), 250)
        self.assertEqual(by_title["Aqueduct"]["url"], "https://iamparis.eu/application_library/474")
        self.assertTrue(by_title["Aqueduct"]["verified_direct_url"])
        self.assertEqual(by_title["Climate Watch"]["url"], "https://iamparis.eu/application_library")
        self.assertFalse(by_title["Climate Watch"]["verified_direct_url"])
        self.assertEqual(by_title["Climate Watch"]["search_hint"], "Climate Watch")
        self.assertEqual(
            by_title["Climate Watch"]["fallback_instruction"],
            "Open the Application Library and search for: Climate Watch",
        )

    def test_generated_catalog_is_deterministic(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            first = Path(temp_dir) / "first.json"
            second = Path(temp_dir) / "second.json"

            write_link_catalog(Path("iamparis_chatbot_links.xlsx"), first)
            write_link_catalog(Path("iamparis_chatbot_links.xlsx"), second)

            self.assertEqual(first.read_text(), second.read_text())
            data = json.loads(first.read_text())
            self.assertTrue(all("id" in item and "url" in item for item in data))


if __name__ == "__main__":
    unittest.main()
