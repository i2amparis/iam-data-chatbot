import json
import tempfile
import unittest
from pathlib import Path

from runtime_context import build_runtime_context, build_workspace_lookup, load_link_catalog


class RuntimeContextTests(unittest.TestCase):
    def test_workspace_lookup_groups_records(self):
        lookup = build_workspace_lookup([
            {"workspace_code": "energy-systems", "variable": "A"},
            {"workspace_code": "energy-systems", "variable": "B"},
            {"workspace_code": "buildings-transf", "variable": "C"},
            {"variable": "D"},
        ])

        self.assertEqual(len(lookup["energy-systems"]), 2)
        self.assertEqual(len(lookup["buildings-transf"]), 1)
        self.assertEqual(len(lookup["unknown"]), 1)

    def test_load_link_catalog_returns_empty_for_missing_file(self):
        self.assertEqual(load_link_catalog(Path("missing-link-catalog.json")), [])

    def test_build_runtime_context_exposes_dict_like_resources(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            link_catalog_path = temp_path / "links.json"
            metadata_cache = temp_path / "metadata.pkl"
            link_catalog_path.write_text(json.dumps([{"title": "IAM PARIS Results"}]))

            context = build_runtime_context(
                models=[{"modelName": "GCAM"}],
                ts=[
                    {
                        "workspace_code": "energy-systems",
                        "variable": "Emissions|CO2",
                        "region": "World",
                        "scenario": "Baseline",
                        "modelName": "GCAM",
                        "unit": "Mt CO2/yr",
                    }
                ],
                vector_store="vector-store",
                env={"OPENAI_API_KEY": "test"},
                bot="bot",
                link_catalog_path=link_catalog_path,
                metadata_cache_file=str(metadata_cache),
            )

            self.assertEqual(context["models"][0]["modelName"], "GCAM")
            self.assertEqual(context.get("vector_store"), "vector-store")
            self.assertEqual(context.link_catalog[0]["title"], "IAM PARIS Results")
            self.assertIn("energy-systems", context.workspace_lookup)
            self.assertIn("GCAM", context.metadata.all_model_names)


if __name__ == "__main__":
    unittest.main()
