import unittest

from manager import _looks_like_site_navigation_request
from runtime_context import load_link_catalog


class SiteNavigationRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog = load_link_catalog()

    def test_project_publications_is_navigation(self):
        self.assertTrue(_looks_like_site_navigation_request(
            "where can I find project publications?", self.catalog,
        ))

    def test_user_guide_existence_question_is_navigation(self):
        self.assertTrue(_looks_like_site_navigation_request(
            "is there a user guide for the platform?", self.catalog,
        ))

    def test_workspaces_page_is_catalogued_navigation(self):
        self.assertTrue(_looks_like_site_navigation_request(
            "take me to the workspaces page", self.catalog,
        ))


if __name__ == "__main__":
    unittest.main()
