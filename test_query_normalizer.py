import unittest

from query_normalizer import normalize_query_text, query_tokens


class QueryNormalizerTests(unittest.TestCase):
    def test_normalizes_punctuation_case_hyphen_and_underscore(self):
        normalized = normalize_query_text("Solar_PV--Power, BY 2050!")
        self.assertIn("solar pv power by 2050", normalized)
        tokens = query_tokens("Solar_PV--Power, BY 2050!")
        self.assertIn("solar", tokens)
        self.assertIn("electricity", tokens)

    def test_adds_english_synonyms(self):
        tokens = query_tokens(
            "Compare carbon dioxide and greenhouse gas pathways by country as a chart"
        )

        for token in ["comparison", "co2", "ghg", "emissions", "scenario", "region", "plot"]:
            self.assertIn(token, tokens)

    def test_data_value_terms_expand_to_data_query(self):
        tokens = query_tokens("show values from a time series")
        self.assertIn("data", tokens)
        self.assertIn("query", tokens)
        self.assertIn("timeseries", tokens)

    def test_plural_singular_variants_are_available(self):
        tokens = query_tokens("locations pathways turbines")
        self.assertIn("location", tokens)
        self.assertIn("pathway", tokens)
        self.assertIn("turbine", tokens)


if __name__ == "__main__":
    unittest.main()
