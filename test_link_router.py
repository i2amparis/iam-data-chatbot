import unittest

from runtime_context import load_link_catalog
from link_router import format_relevant_links, suggest_links


class LinkRouterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog = load_link_catalog()

    def test_model_query_prefers_models_link(self):
        links = suggest_links(
            "Tell me about the GCAM model assumptions",
            self.catalog,
            agent_name="model_explanation",
            entities={"model": "GCAM"},
        )

        self.assertTrue(links)
        self.assertEqual(links[0]["title"], "GCAM")
        self.assertEqual(links[0]["url"], "https://iamparis.eu/models")

    def test_model_query_without_direct_catalog_item_prefers_models_search_hint(self):
        links = suggest_links(
            "Tell me about the REMIND model",
            self.catalog,
            agent_name="model_explanation",
            entities={"model": "REMIND"},
        )

        self.assertTrue(links)
        self.assertEqual(links[0]["title"], "Models")
        self.assertEqual(links[0]["url"], "https://iamparis.eu/models")
        self.assertEqual(links[0]["search_hint"], "REMIND")

    def test_buildings_query_finds_buildings_transformation_result(self):
        links = suggest_links(
            "Show buildings transformation results for NDC pathways",
            self.catalog,
            agent_name="data_query",
            entities={"variable": "Final Energy|Residential and Commercial"},
        )

        titles = [link["title"] for link in links]
        self.assertIn("Buildings Transformation", titles)

    def test_ndc_sector_query_prefers_ndc_aspects_links(self):
        links = suggest_links(
            "NDC impacts for transport and buildings",
            self.catalog,
            agent_name="data_query",
        )

        urls = [link["url"] for link in links]
        self.assertTrue(any("ndc-aspects" in url for url in urls))
        self.assertFalse(any("fit-for-55" in url for url in urls))

    def test_transport_query_finds_transportation_transformation_result(self):
        links = suggest_links(
            "transportation transformation results for mobility and vehicles",
            self.catalog,
            agent_name="data_query",
        )

        titles = [link["title"] for link in links]
        self.assertIn("Transportation Transformation", titles)

    def test_afolu_query_finds_afolu_transformation_result(self):
        links = suggest_links(
            "AFOLU agriculture land forestry transformation results",
            self.catalog,
            agent_name="data_query",
        )

        titles = [link["title"] for link in links]
        self.assertTrue(any(title.lower() == "afolu transformation" for title in titles))

    def test_iam_compact_query_prefers_fit_for_55_link(self):
        links = suggest_links(
            "Fit-for-55 EU net zero results",
            self.catalog,
            agent_name="data_query",
        )

        self.assertIn("fit-for-55", links[0]["url"])

    def test_data_story_query_prefers_policy_catalogue(self):
        links = suggest_links(
            "policy catalogue climate policies",
            self.catalog,
            agent_name="general_qa",
        )

        self.assertEqual(links[0]["title"], "Policy Catalogue Interactive Explorer")

    def test_application_library_direct_detail_url(self):
        links = suggest_links(
            "Open the Aqueduct raw data application",
            self.catalog,
            agent_name="general_qa",
        )

        self.assertEqual(links[0]["title"], "Aqueduct")
        self.assertEqual(links[0]["url"], "https://iamparis.eu/application_library/474")
        self.assertEqual(links[0]["search_hint"], "")

    def test_application_library_fallback_search_hint(self):
        links = suggest_links(
            "Where can I find Climate Watch?",
            self.catalog,
            agent_name="general_qa",
        )

        self.assertEqual(links[0]["title"], "Climate Watch")
        self.assertEqual(links[0]["url"], "https://iamparis.eu/application_library")
        self.assertEqual(links[0]["search_hint"], "Climate Watch")

    def test_contact_query_prefers_contact_page(self):
        links = suggest_links(
            "contact IAM PARIS team",
            self.catalog,
            agent_name="general_qa",
        )

        self.assertTrue(links)
        self.assertEqual(links[0]["title"], "Contact")
        self.assertEqual(links[0]["url"], "https://iamparis.eu/contact")

    def test_format_relevant_links_includes_search_hint(self):
        formatted = format_relevant_links([
            {
                "title": "Climate Watch",
                "url": "https://iamparis.eu/application_library",
                "reason": "Matched: Climate Watch",
                "confidence": 1.0,
                "search_hint": "Climate Watch",
            }
        ])

        self.assertIn("[Climate Watch](https://iamparis.eu/application_library)", formatted)
        self.assertIn("Search for: Climate Watch.", formatted)


if __name__ == "__main__":
    unittest.main()
