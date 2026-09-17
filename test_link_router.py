import unittest
import json

from runtime_context import load_link_catalog
from link_router import (
    catalog_category_root,
    catalog_link_matches_entry,
    format_relevant_links,
    has_catalog_navigation_target,
    infer_navigation_category,
    suggest_links,
)


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

    def test_workspaces_navigation_uses_verified_workspace_route(self):
        query = "take me to the workspaces page"
        links = suggest_links(
            query,
            self.catalog,
            agent_name="general_qa",
            navigation_category=infer_navigation_category(query, self.catalog),
        )

        self.assertTrue(links)
        self.assertEqual(links[0]["title"], "My Workspace")
        self.assertEqual(links[0]["url"], "https://iamparis.eu/scientific_module")

    def test_plural_model_entities_filter_unrelated_specific_model_links(self):
        links = suggest_links(
            "Compare limitations and typical uses",
            self.catalog,
            agent_name="model_explanation",
            entities={"models": ["REMIND", "WITCH"]},
        )

        self.assertTrue(links)
        self.assertEqual(links[0]["title"], "Models")
        self.assertEqual(links[0]["search_hint"], "REMIND, WITCH")
        self.assertFalse(any(link["title"] == "GEMINI-E3" for link in links))

    def test_model_link_matching_uses_identifier_tokens_not_substrings(self):
        links = suggest_links(
            "Compare the two model approaches",
            self.catalog,
            agent_name="model_explanation",
            entities={"models": ["MESSAGEix-GLOBIOM", "WITCH"]},
        )

        self.assertFalse(any(link["title"] == "GLOBIO" for link in links))
        self.assertEqual(links[0]["title"], "Models")

    def test_data_route_also_filters_unrelated_specific_model_links(self):
        links = suggest_links(
            "Compare model outputs",
            self.catalog,
            agent_name="data_plotting",
            entities={"model": "MESSAGEix-GLOBIOM"},
        )

        self.assertFalse(any(link["title"] == "GLOBIO" for link in links))

    def test_model_search_hint_deduplicates_singular_and_plural_state(self):
        links = suggest_links(
            "Give me documentation links for both models",
            self.catalog,
            agent_name="model_explanation",
            entities={"model": "REMIND", "models": ["REMIND", "WITCH"]},
        )

        self.assertEqual(links[0]["title"], "Models")
        self.assertEqual(links[0]["search_hint"], "REMIND, WITCH")

    def test_numeric_model_catalogue_entry_is_never_returned_as_a_link(self):
        links = suggest_links(
            "Tell me about model 42",
            self.catalog,
            agent_name="model_explanation",
            entities={"model": "42"},
        )

        serialized = json.dumps(links)
        self.assertNotRegex(serialized, r'(?<!\d)42(?!\d)')
        self.assertFalse(any(link.get("title") == "42" for link in links))

    def test_generic_model_catalogue_navigation_uses_catalogued_root(self):
        query = "Give me the IAM PARIS page where I can browse valid model names."
        category = infer_navigation_category(query, self.catalog)

        links = suggest_links(
            query,
            self.catalog,
            agent_name="general_qa",
            navigation_category=category,
        )

        self.assertEqual(category, "models")
        self.assertTrue(links)
        self.assertEqual(links[0]["title"], "Models")
        self.assertEqual(links[0]["url"], "https://iamparis.eu/models")

    def test_navigation_category_root_is_runtime_catalogue_driven(self):
        catalog = [
            {
                "title": "Widgets",
                "url": "https://example.test/widget-directory",
                "category": "widgets",
                "item_type": "route",
                "keywords": [],
                "verified_direct_url": True,
                "search_hint": "",
            }
        ]
        query = "Open the widget catalogue."
        category = infer_navigation_category(query, catalog)

        links = suggest_links(
            query,
            catalog,
            agent_name="general_qa",
            navigation_category=category,
        )

        self.assertEqual(category, "widgets")
        self.assertEqual(links[0]["url"], "https://example.test/widget-directory")

    def test_results_root_resolution_uses_runtime_category_metadata(self):
        catalog = [{
            "title": "Results",
            "url": "https://example.test/data-explorer",
            "category": "results",
            "item_type": "route",
            "keywords": [],
            "verified_direct_url": True,
            "search_hint": "",
        }]

        root = catalog_category_root(catalog, "results")

        self.assertIsNotNone(root)
        self.assertEqual(root["url"], "https://example.test/data-explorer")
        self.assertTrue(catalog_link_matches_entry(
            {"title": "Different label", "url": "https://example.test/data-explorer/"},
            root,
        ))

    def test_category_root_supports_legacy_records_and_prefers_exact_title(self):
        catalog = [
            {
                "title": "Portal Results",
                "url": "https://example.test/legacy-data",
                "category": "results",
                "keywords": [],
                "search_hint": "",
            },
            {
                "title": "Results",
                "url": "https://example.test/current-data",
                "category": "results",
                "item_type": "route",
                "keywords": [],
                "search_hint": "",
            },
        ]

        root = catalog_category_root(catalog, "results")

        self.assertEqual(root["url"], "https://example.test/current-data")
        self.assertEqual(
            catalog_category_root(catalog[:1], "results")["url"],
            "https://example.test/legacy-data",
        )

    def test_specific_model_navigation_still_outranks_category_root(self):
        query = "Open the Model Comparison page."
        category = infer_navigation_category(query, self.catalog)

        links = suggest_links(
            query,
            self.catalog,
            agent_name="general_qa",
            navigation_category=category,
        )

        self.assertEqual(category, "models")
        self.assertEqual(links[0]["title"], "Model Comparison")

    def test_scenario_data_download_uses_results_root_before_weak_matches(self):
        query = "where can I download the scenario data?"

        links = suggest_links(
            query,
            self.catalog,
            agent_name="general_qa",
            navigation_category=infer_navigation_category(query, self.catalog),
        )

        self.assertEqual([link["title"] for link in links], ["Results"])
        self.assertEqual(links[0]["url"], "https://iamparis.eu/results")

    def test_project_publications_use_verified_results_hub(self):
        links = suggest_links(
            "where can I find project publications?",
            self.catalog,
            agent_name="general_qa",
        )

        self.assertEqual([link["title"] for link in links], ["Results"])
        self.assertEqual(links[0]["url"], "https://iamparis.eu/results")

    def test_platform_guide_uses_contact_instead_of_unrelated_weak_link(self):
        links = suggest_links(
            "is there a user guide for the platform?",
            self.catalog,
            agent_name="general_qa",
        )

        self.assertEqual([link["title"] for link in links], ["Contact"])
        self.assertEqual(links[0]["url"], "https://iamparis.eu/contact")
        self.assertIn("No dedicated user-guide destination", links[0]["reason"])

    def test_transformation_navigation_selects_exact_child_pages(self):
        cases = [
            (
                "link me to the transport transformation results",
                "Transportation Transformation",
                "https://iamparis.eu/results/ndc-aspects/transportation-transformation/policy_questions",
            ),
            (
                "show me the AFOLU transformation results",
                "AFOLU transformation",
                "https://iamparis.eu/results/ndc-aspects/afolu-transformation/policy_questions",
            ),
            (
                "AFOLU land use results",
                "AFOLU transformation",
                "https://iamparis.eu/results/ndc-aspects/afolu-transformation/policy_questions",
            ),
            (
                "can I find the buildings results?",
                "Buildings Transformation",
                "https://iamparis.eu/results/ndc-aspects/buildings-transformation/policy_questions",
            ),
        ]

        for query, expected_title, expected_url in cases:
            with self.subTest(query=query):
                category = infer_navigation_category(query, self.catalog)
                links = suggest_links(
                    query,
                    self.catalog,
                    agent_name="general_qa",
                    navigation_category=category,
                )

                self.assertEqual(category, "results")
                self.assertEqual([link["title"] for link in links], [expected_title])
                self.assertEqual(links[0]["url"], expected_url)

    def test_model_catalogue_listing_prefers_models_root_for_data_route(self):
        links = suggest_links(
            "which models are available",
            self.catalog,
            agent_name="data_query",
        )

        self.assertEqual([link["title"] for link in links], ["Models"])
        self.assertEqual(links[0]["url"], "https://iamparis.eu/models")
        self.assertGreaterEqual(links[0]["confidence"], 0.4)

    def test_model_documentation_navigation_uses_hub_search_fallback(self):
        query = "link me to the model documentation for GCAM"
        category = infer_navigation_category(query, self.catalog)

        links = suggest_links(
            query,
            self.catalog,
            agent_name="general_qa",
            navigation_category=category,
        )

        self.assertEqual(category, "models")
        self.assertEqual([link["title"] for link in links], ["Models"])
        self.assertEqual(links[0]["url"], "https://iamparis.eu/models")
        self.assertEqual(links[0]["search_hint"], "GCAM")
        self.assertEqual(
            links[0]["fallback_instruction"],
            "Open the Models directory and search for: GCAM",
        )

    def test_navigation_paraphrase_prefers_complete_child_route_identity(self):
        catalog = [
            {
                "title": "Widgets",
                "url": "https://example.test/widgets",
                "category": "widgets",
                "item_type": "route",
                "keywords": ["Widget catalogue"],
                "verified_direct_url": True,
            },
            {
                "title": "Widget Comparison",
                "url": "https://example.test/widgets/side-by-side",
                "category": "widgets",
                "item_type": "route",
                "keywords": ["Compare widgets", "Side-by-side widgets"],
                "verified_direct_url": True,
            },
        ]
        query = "Where is the catalogue page for comparing widgets?"

        links = suggest_links(
            query,
            catalog,
            agent_name="general_qa",
            navigation_category=infer_navigation_category(query, catalog),
        )

        self.assertEqual([link["title"] for link in links], ["Widget Comparison"])

    def test_generic_explorer_navigation_prefers_runtime_category_root(self):
        catalog = [
            {
                "title": "Reports",
                "url": "https://example.test/reports",
                "category": "reports",
                "item_type": "route",
                "keywords": ["Report outputs hub", "Interactive explorer"],
                "verified_direct_url": True,
            },
            {
                "title": "Detailed Cost Study",
                "url": "https://example.test/reports/cost-study",
                "category": "reports",
                "item_type": "workspace",
                "keywords": ["Cost results", "Detailed report"],
                "verified_direct_url": True,
            },
        ]
        query = "Open the reports explorer."

        links = suggest_links(
            query,
            catalog,
            agent_name="general_qa",
            navigation_category=infer_navigation_category(query, catalog),
        )

        self.assertEqual([link["title"] for link in links], ["Reports"])

    def test_navigation_alias_can_select_specific_runtime_destination(self):
        catalog = [
            {
                "title": "Reports",
                "url": "https://example.test/reports",
                "category": "reports",
                "item_type": "route",
                "keywords": ["Report outputs hub"],
                "verified_direct_url": True,
            },
            {
                "title": "Long-form transition assessment",
                "url": "https://example.test/reports/transition",
                "category": "reports",
                "item_type": "workspace",
                "keywords": ["Clean heat transition"],
                "verified_direct_url": True,
            },
        ]
        query = "Where can I find the clean heat transition results?"

        links = suggest_links(
            query,
            catalog,
            agent_name="general_qa",
            navigation_category=infer_navigation_category(query, catalog),
        )

        self.assertEqual([link["title"] for link in links], ["Long-form transition assessment"])

    def test_named_detail_infers_parent_category_without_root_words(self):
        catalog = [
            {
                "title": "Archives",
                "url": "https://example.test/archives",
                "category": "archives",
                "item_type": "route",
                "verified_direct_url": True,
            },
            {
                "title": "Blue Lens Explorer",
                "url": "https://example.test/archives/blue-lens",
                "category": "archives",
                "item_type": "workspace",
                "keywords": ["Blue Lens"],
                "verified_direct_url": True,
            },
        ]
        query = "Take me directly to the Blue Lens Explorer."
        category = infer_navigation_category(query, catalog)

        links = suggest_links(
            query,
            catalog,
            agent_name="general_qa",
            navigation_category=category,
        )

        self.assertEqual(category, "archives")
        self.assertEqual([link["title"] for link in links], ["Blue Lens Explorer"])

    def test_named_child_beats_category_root_when_both_are_mentioned(self):
        catalog = [
            {
                "title": "Application Archive",
                "url": "https://example.test/apps",
                "category": "application_archive",
                "item_type": "route",
                "verified_direct_url": True,
            },
            {
                "title": "Beacon",
                "url": "https://example.test/apps/beacon",
                "category": "application_archive",
                "item_type": "workspace",
                "verified_direct_url": True,
            },
        ]
        query = "Open Beacon in the Application Archive."

        links = suggest_links(
            query,
            catalog,
            agent_name="general_qa",
            navigation_category=infer_navigation_category(query, catalog),
        )

        self.assertEqual([link["title"] for link in links], ["Beacon"])

    def test_shared_library_url_keeps_named_search_target_identity(self):
        query = "Send me the Climate Policy Radar application link."
        category = infer_navigation_category(query, self.catalog)
        links = suggest_links(
            query,
            self.catalog,
            agent_name="general_qa",
            navigation_category=category,
        )

        self.assertEqual(category, "application_library")
        self.assertEqual([link["title"] for link in links], ["Climate Policy Radar"])
        self.assertEqual(links[0]["url"], "https://iamparis.eu/application_library")
        self.assertEqual(links[0]["search_hint"], "Climate Policy Radar")
        self.assertFalse(links[0]["verified_direct_url"])

    def test_named_library_find_request_is_navigation_with_search_hint(self):
        query = "Find the AR6 Scenario Explorer in the application library."

        self.assertTrue(has_catalog_navigation_target(query, self.catalog))
        category = infer_navigation_category(query, self.catalog)
        links = suggest_links(
            query,
            self.catalog,
            agent_name="general_qa",
            navigation_category=category,
        )

        self.assertEqual(category, "application_library")
        self.assertEqual(links[0]["search_hint"], "AR6 Scenario Explorer and Database")

    def test_partial_long_title_beats_generic_analysis_root(self):
        query = "Where can I read the barriers and enablers analysis?"
        category = infer_navigation_category(query, self.catalog)
        links = suggest_links(
            query,
            self.catalog,
            agent_name="general_qa",
            navigation_category=category,
        )

        self.assertEqual(category, "data_stories")
        self.assertEqual([link["title"] for link in links], ["Barriers, Enablers & Policy Analysis"])
        self.assertEqual(links[0]["url"], "https://iamparis.eu/datastories/mitigation_barriers")

    def test_catalog_navigation_target_recognizes_direct_and_indirect_requests(self):
        self.assertTrue(has_catalog_navigation_target(
            "Open the model comparison page.",
            self.catalog,
        ))
        self.assertTrue(has_catalog_navigation_target(
            "Also give me the page used specifically to compare models.",
            self.catalog,
        ))
        self.assertFalse(has_catalog_navigation_target(
            "Compare GDP values for two models in 2050.",
            self.catalog,
        ))

    def test_navigation_target_detection_is_runtime_catalogue_driven(self):
        catalog = [
            {
                "title": "Widget Comparison",
                "url": "https://example.test/widgets/compare",
                "category": "widgets",
                "item_type": "route",
                "keywords": ["Side-by-side widget comparison", "Compare widgets"],
            }
        ]

        self.assertTrue(has_catalog_navigation_target(
            "Give me the page used to compare widgets.",
            catalog,
        ))
        self.assertTrue(has_catalog_navigation_target(
            "Open Widget Comparison.",
            catalog,
        ))
        self.assertFalse(has_catalog_navigation_target(
            "Compare widget output values in 2050.",
            catalog,
        ))

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

    def test_topic_availability_without_model_entity_drops_specific_model_links(self):
        links = suggest_links(
            "Which models provide agriculture or land-use variables?",
            self.catalog,
            agent_name="data_query",
            entities={},
        )

        self.assertFalse(any(link["category"] == "models" and link["search_hint"] for link in links))

    def test_iam_compact_query_prefers_fit_for_55_link(self):
        links = suggest_links(
            "Fit-for-55 EU net zero results",
            self.catalog,
            agent_name="data_query",
        )

        self.assertIn("fit-for-55", links[0]["url"])

    def test_fit_for_55_navigation_prefers_specific_workspace_over_project_hub(self):
        query = "Where can I find the Fit-for-55 policy results?"
        links = suggest_links(
            query,
            self.catalog,
            agent_name="general_qa",
            navigation_category=infer_navigation_category(query, self.catalog),
        )

        self.assertIn("fit-for-55", links[0]["url"])
        self.assertIn("policy_questions", links[0]["url"])

    def test_runtime_candidate_metadata_does_not_become_link_evidence(self):
        catalog = [
            {
                "title": "Results",
                "url": "https://iamparis.eu/results",
                "category": "results",
                "item_type": "route",
                "keywords": [],
                "verified_direct_url": True,
            },
            {
                "title": "Candidate Pathway",
                "url": "https://example.test/candidate-pathway",
                "category": "results",
                "item_type": "workspace",
                "keywords": ["Candidate Pathway"],
                "verified_direct_url": True,
            },
        ]

        links = suggest_links(
            "Population in Europe in 2050",
            catalog,
            agent_name="data_query",
            entities={
                "variable": "Population",
                "region": "Europe",
                "scenario_candidates": ["Candidate Pathway"],
                "scenarios": ["Candidate Pathway"],
                "result_models": ["Candidate Pathway"],
            },
        )

        self.assertEqual([link["title"] for link in links], ["Results"])

    def test_data_link_needs_more_than_one_incidental_topic_token(self):
        catalog = [
            {
                "title": "Results",
                "url": "https://iamparis.eu/results",
                "category": "results",
                "item_type": "route",
                "keywords": [],
                "verified_direct_url": True,
            },
            {
                "title": "Regional Transition",
                "url": "https://example.test/regional-transition",
                "category": "results",
                "item_type": "workspace",
                "keywords": ["EU pathways", "transition plan"],
                "verified_direct_url": True,
            },
        ]

        broad = suggest_links(
            "Population in EU in 2050",
            catalog,
            agent_name="data_query",
            entities={"variable": "Population", "region": "EU"},
        )
        specific = suggest_links(
            "Compare regional transition pathways for the EU",
            catalog,
            agent_name="data_query",
        )

        self.assertEqual([link["title"] for link in broad], ["Results"])
        self.assertEqual(specific[0]["title"], "Regional Transition")

    def test_broad_energy_query_does_not_promote_workspace_from_split_evidence(self):
        catalog = [
            {
                "title": "Results",
                "url": "https://example.test/results",
                "category": "results",
                "item_type": "route",
                "keywords": [],
                "verified_direct_url": True,
            },
            {
                "title": "Housing Transition",
                "url": "https://example.test/results/housing",
                "category": "results",
                "item_type": "workspace",
                "project": "Commitment Programme",
                "keywords": ["energy demand", "housing"],
                "verified_direct_url": True,
            },
        ]

        broad = suggest_links(
            "Show final energy electricity under the commitment scenario",
            catalog,
            agent_name="data_query",
            entities={"variable": "Final Energy|Electricity"},
        )
        topic_specific = suggest_links(
            "Show housing energy demand",
            catalog,
            agent_name="data_query",
        )

        self.assertEqual([link["title"] for link in broad], ["Results"])
        self.assertEqual(topic_specific[0]["title"], "Housing Transition")

    def test_reported_broad_final_energy_query_rejects_weak_buildings_link(self):
        links = suggest_links(
            "Show Final Energy Electricity for Brazil under PR_NDC_CP in 2050.",
            self.catalog,
            agent_name="data_query",
            entities={"variable": "Final Energy|Electricity", "region": "BRA"},
        )

        self.assertEqual(links[0]["title"], "Results")
        self.assertFalse(any(link["title"] == "Buildings Transformation" for link in links))

    def test_energy_demand_phrase_alone_does_not_identify_buildings_workspace(self):
        links = suggest_links(
            "final energy demand for India",
            self.catalog,
            agent_name="data_query",
            entities={"variable": "Final Energy", "region": "IND"},
        )

        self.assertEqual([link["title"] for link in links], ["Results"])

    def test_broad_primary_energy_scope_does_not_identify_a_project_link(self):
        links = suggest_links(
            "Show Primary Energy Coal for Europe under PR_WWH_CP from 2040 to 2050.",
            self.catalog,
            agent_name="data_query",
            entities={
                "variable": "Primary Energy|Coal",
                "region": "EU",
                "scenario": "PR_WWH_CP",
            },
        )

        self.assertEqual([link["title"] for link in links], ["Results"])
        self.assertFalse(any(link["title"] == "EU-CHINA BRIDGE" for link in links))

    def test_specific_project_identity_still_uses_runtime_metadata(self):
        catalog = [
            {
                "title": "Reports",
                "url": "https://example.test/reports",
                "category": "reports",
                "item_type": "route",
                "keywords": [],
                "verified_direct_url": True,
            },
            {
                "title": "Cross-border programme",
                "url": "https://example.test/reports/cross-border",
                "category": "reports",
                "item_type": "project",
                "keywords": ["North-South climate energy bridge programme"],
                "verified_direct_url": True,
            },
        ]

        links = suggest_links(
            "Show the North South climate energy bridge programme results.",
            catalog,
            agent_name="data_query",
        )

        self.assertEqual(links[0]["title"], "Cross-border programme")

    def test_industrial_topic_still_selects_concise_matching_workspace(self):
        links = suggest_links(
            "Show industrial energy efficiency data for India in 2050.",
            self.catalog,
            agent_name="data_query",
        )

        self.assertIn("Industrial transformation", [link["title"] for link in links])

    def test_common_catalogue_term_does_not_select_arbitrary_specific_result(self):
        links = suggest_links(
            "Which models report Primary Energy|Coal for EU?",
            self.catalog,
            agent_name="data_query",
            entities={"variable": "Primary Energy|Coal", "region": "EU"},
        )

        self.assertNotIn("Energy Progress Report", [link["title"] for link in links])
        self.assertIn("Results", [link["title"] for link in links])

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

    def test_near_title_phrase_outranks_generic_analysis_match(self):
        catalog = [
            {
                "title": "Implications of regional steel industry relocation",
                "url": "https://example.test/steel",
                "category": "results",
                "keywords": ["steel industry", "relocation"],
            },
            {
                "title": "Generic Policy Analysis",
                "url": "https://example.test/library",
                "category": "application_library",
                "keywords": ["analysis", "policy"],
            },
        ]

        links = suggest_links(
            "Where is the analysis about relocating regional steel production?",
            catalog,
            agent_name="general_qa",
        )

        self.assertEqual(links[0]["url"], "https://example.test/steel")

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
