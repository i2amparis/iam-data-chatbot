import json
import logging
import os
import re
from dataclasses import replace
from typing import Dict, Any, List, Tuple, Optional
from agents import BaseAgent, DataQueryAgent, ModelExplanationAgent, DataPlottingAgent, GeneralQAAgent, ModellingSuggestionsAgent
from llm_factory import get_chat_openai as ChatOpenAI
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from pathlib import Path
from data_utils import (
    _broad_electricity_candidates,
    _choice_prompt,
    _infer_variable_intent,
    _list_model_category,
    _model_scoped_category,
    _looks_like_comparison_request,
    _looks_like_category_list_request,
    _looks_like_capability_question,
    _looks_like_data_request,
    _looks_like_model_info_request,
    _looks_like_plot_request,
    _variable_matches_query_signal,
    _matched_workspace,
    _workspace_entries,
    sanitize_variable_for_query,
)
from canonical_aliases import (
    canonical_region_from_query,
    canonical_scenario_from_query,
    canonical_scenario_family_from_query,
    explicit_scenarios_from_query,
    preferred_variable_from_query,
    scenario_family_members,
    scenario_in_family,
)
from link_router import (
    catalog_category_root,
    catalog_link_matches_entry,
    format_relevant_links,
    has_catalog_navigation_target,
    infer_navigation_category,
    suggest_links,
)
from model_profiles import (
    find_model_profile,
    find_model_profiles,
    format_model_comparison_answer,
    format_model_profile_answer,
)
from query_plan import ScopePatch, build_query_plan, render_scope_query
from model_aliases import (
    display_model_label,
    model_family_key,
    normalize_model_name,
    resolve_model_candidates,
)
from utils_query import extract_region_from_query
from year_filters import YearFilter, extract_year_filter, extract_year_range
from llm_config import ROUTER_MODEL
from resolved_scope import (
    has_numeric_result_table,
    ConversationState,
    PendingClarification,
    consume_resolved_scope,
    record_resolved_scope,
)


# Navigation term lists live in config/site_navigation.json so new workspaces,
# data stories or site pages only need a config edit. The literals below are
# the fallback when the config file is missing or invalid.
_NAV_CONFIG_PATH = Path("config/site_navigation.json")
_NAV_DEFAULTS: Dict[str, tuple] = {
    "navigation_terms": (
        "where can i find", "where do i find", "where is", "open ", "find ",
        "link", "url", "page", "website", "application library",
        "raw data application", "data story", "data stories",
        "policy catalogue", "policy catalog", "database", "explorer",
    ),
    "named_site_items": (
        "aqueduct", "climate watch", "cdp open data portal", "data portal",
        "afolu transformation", "buildings transformation",
        "transportation transformation", "transport transformation",
        "industrial transformation",
    ),
    "data_story_items": (
        "policy catalogue", "policy catalog", "recovery policy", "circularity",
        "decarbonisation data story", "decarbonization data story",
        "technology inventories", "barriers and enablers", "scenario metadata",
    ),
    "project_workspace_items": (
        "iam compact", "fit for 55", "fit-for-55", "renewable energy metrics",
        "post glasgow", "post-glasgow", "steel relocation", "cost of capital",
        "behavioural change", "behavioral change", "technology constrained",
        "tech constrained", "ndc aspects", "global impacts of ndcs",
        "long term targets", "long-term targets",
    ),
    "analysis_contact_items": (
        "custom analysis", "analysis service", "analysis support",
        "request analysis", "contact iam paris",
    ),
    "transformation_workspace_items": (
        "buildings", "building", "transport", "transportation",
        "industrial", "industry", "afolu",
    ),
    "generic_site_targets": (
        "documentation", "docs", "user guide", "scenario explorer",
        "model documentation", "model catalogue", "model catalog",
        "models catalogue", "models catalog", "model directory", "model names",
        "application library", "data portal",
        "dashboard", "tutorial", "iam paris results", "paris results",
        "results page", "scenario database",
    ),
    "strong_nav_phrases": (
        "where can i find", "where do i find", "where is",
        "where can i read", "where should i read",
        "give me the link", "send me the link", "take me to", "navigate to",
        "how do i access", "how can i access", "how do i open", "link to",
    ),
    "extra_unambiguous_site_terms": (
        "application library", "raw data application", "data story", "data stories",
        "policy catalogue", "policy catalog", "recovery policy",
        "technology inventories", "barriers and enablers", "scenario metadata",
        "iam compact", "fit for 55", "fit-for-55", "post glasgow", "post-glasgow",
        "ndc aspects", "global impacts of ndcs", "cost of capital", "steel relocation",
    ),
}


def _load_nav_terms() -> Dict[str, tuple]:
    terms = dict(_NAV_DEFAULTS)
    try:
        data = json.loads(_NAV_CONFIG_PATH.read_text())
    except (OSError, ValueError):
        return terms
    for key, value in data.items():
        if key in terms and isinstance(value, list):
            terms[key] = tuple(str(item).lower() for item in value if str(item).strip())
    return terms


_NAV_TERMS = _load_nav_terms()


def _looks_like_site_navigation_request(
    query: str,
    catalog: Optional[List[Dict[str, Any]]] = None,
) -> bool:
    q = str(query or "").strip().lower()
    if not q:
        return False
    # A concrete variable path plus a year is a numeric-data request, even if
    # it also names a study.  For example, "results for Emissions|CO2 in 2050
    # for AFOLU transformation" must return values, not only a navigation link.
    if (
        _looks_like_data_request(q)
        and bool(re.search(r"\b(?:19|20|21)\d{2}\b", q))
        and (
            "|" in q
            or bool(re.search(
                r"\b(?:emissions?|co2|methane|ch4|n2o|energy|capacity|gdp|"
                r"population|demand|supply|price|generation|investment)\b",
                q,
            ))
        )
        and not re.search(r"\b(?:link|url|open|browse|navigate|take\s+me|go\s+to)\b", q)
    ):
        return False
    if has_catalog_navigation_target(query, catalog):
        return True
    # A navigation-shaped request that names a live catalogue *category*
    # ("the models page", "the data stories", "the results section") resolves
    # to that category's root. `infer_navigation_category` only fires when the
    # category tokens are actually named, so genuine data queries (which do not
    # name a category) are not hijacked; it stays data-driven rather than
    # enumerating page names here.
    # "list/show all models" is a category-listing request (enumerate the
    # items), not navigation, so those verbs are deliberately excluded here.
    if catalog and infer_navigation_category(query, catalog):
        if re.search(
            r"\b(?:page|section|link|url|website|site|portal|catalog(?:ue)?|"
            r"where|open|take\s+me|go\s+to|navigate|browse|visit)\b",
            q,
        ):
            return True
    navigation_terms = _NAV_TERMS["navigation_terms"]
    named_site_items = _NAV_TERMS["named_site_items"]
    data_story_items = _NAV_TERMS["data_story_items"]
    project_workspace_items = _NAV_TERMS["project_workspace_items"]
    analysis_contact_items = _NAV_TERMS["analysis_contact_items"]
    generic_site_targets = _NAV_TERMS["generic_site_targets"]
    strong_nav_phrases = _NAV_TERMS["strong_nav_phrases"]

    # Syntax-level navigation request. This is intentionally topic-agnostic:
    # the destination still has to be resolved from the runtime link catalog.
    if re.search(
        r"\bwhere\b[^.?!]{0,80}\b(?:read|browse|view|see|access|open|find)\b",
        q,
    ) and ("iam paris" in q or any(term in q for term in named_site_items + data_story_items + project_workspace_items)):
        return True

    if any(term in q for term in navigation_terms) and any(term in q for term in named_site_items):
        return True
    # Generic navigation intent: a navigation verb/term paired with a site or
    # documentation target. Targets are deliberately specific (no bare "results")
    # so genuine data queries are not hijacked to general_qa.
    nav_phrases = navigation_terms + (
        "how do i access", "how can i access", "how do i open",
        "give me the link", "send me the link", "take me to", "navigate to",
    )
    if any(p in q for p in nav_phrases) and any(t in q for t in generic_site_targets):
        return True

    # A transformation-workspace keyword ("buildings", "transport", ...) asked
    # for as a results/workspace page is navigation, even when a verb like
    # "show" makes it look data-shaped — as long as no data variable (emissions,
    # energy, gdp, ...) is named, which would make it a real data query.
    _data_variable_hint = (
        "emission", "co2", "energy", "capacity", "gdp", "population",
        "demand", "supply", "price", "generation", "investment",
    )
    _wants_workspace_page = any(
        term in q for term in _NAV_TERMS["transformation_workspace_items"]
    ) and any(
        term in q for term in ("result", "results", "workspace", "transformation", "policy questions")
    ) and not any(t in q for t in _data_variable_hint)
    if _wants_workspace_page:
        return True

    # Guard: a data-shaped question ("find CO2 emissions data in the database",
    # "show renewable energy metrics for EU") must stay a data query unless the
    # user clearly asks for a page/link or names an unambiguous site item.
    unambiguous_site_terms = named_site_items + _NAV_TERMS["extra_unambiguous_site_terms"]
    if (
        _looks_like_data_request(q)
        and not any(p in q for p in strong_nav_phrases)
        and not any(t in q for t in unambiguous_site_terms)
    ):
        return False

    # Multi-word data-story names are unambiguous; single-word ones (e.g.
    # "circularity") additionally need navigation/data-story intent so that
    # "what is circularity?" stays a general question.
    if any(term in q for term in data_story_items if " " in term):
        return True
    if any(term in q for term in data_story_items if " " not in term) and (
        "data story" in q or any(p in q for p in nav_phrases)
    ):
        return True
    if "global impacts of ndcs" in q:
        return True
    if any(term in q for term in project_workspace_items) and any(term in q for term in ("results", "workspace", "policy questions", "metrics", "pathways", "targets", "aspects", "policy")):
        return True
    if any(term in q for term in analysis_contact_items):
        return True
    if re.search(r"\bcontact\b", q):
        return True
    if any(term in q for term in ("application library", "raw data application", "online model", "dashboard", "interactive map")):
        return True
    if all(term in q for term in ("agriculture", "forestry", "land")) and any(term in q for term in ("result", "results", "workspace", "transformation")):
        return True
    if "afolu" in q and any(term in q for term in ("transformation", "results", "workspace")):
        return True
    if "ndc" in q and any(term in q for term in ("transport", "transportation", "buildings", "building", "afolu")) and any(term in q for term in ("result", "results", "workspace")):
        return True
    if any(term in q for term in named_site_items) and any(term in q for term in ("result", "results", "workspace", "transformation")):
        return True
    return False


def _is_integrated_assessment_model_concept_question(query: str) -> bool:
    """Match a definition request, not a request about one named IAM."""
    text = re.sub(r"\s+", " ", str(query or "").strip().casefold()).rstrip("?.! ")
    return bool(
        re.fullmatch(
            r"(?:what\s+(?:is|are)|define|explain|tell\s+me\s+about)\s+"
            r"(?:(?:an?|the)\s+)?integrated\s+assessment\s+models?",
            text,
        )
        or re.fullmatch(
            r"what\s+does\s+(?:an?\s+)?integrated\s+assessment\s+model\s+mean",
            text,
        )
    )


def _integrated_assessment_model_concept_answer() -> str:
    """A deterministic, source-linked IAM definition based on the IPCC glossary."""
    return (
        "### Integrated assessment model (IAM)\n\n"
        "An integrated assessment model combines knowledge from two or more domains in one "
        "consistent framework. In climate analysis, IAMs commonly connect parts of the economy "
        "and energy system with land use, greenhouse-gas emissions and a representation of the "
        "climate system. Researchers use them to explore how socioeconomic and technological "
        "pathways, policies and climate outcomes interact.\n\n"
        "IAM results are conditional scenarios, not a single prediction: they depend on the "
        "model structure and the assumptions supplied to it.\n\n"
        "Source: [IPCC glossary — Integrated assessment model (IAM)]"
        "(https://www.ipcc.ch/sr15/chapter/glossary/)"
    )


def _is_iam_vs_energy_system_question(query: str) -> bool:
    text = re.sub(r"\s+", " ", str(query or "").strip().casefold())
    return bool(
        re.search(r"\b(?:difference|differ|compare|comparison)\b", text)
        and re.search(r"\b(?:iam|integrated assessment model)\b", text)
        and re.search(r"\benergy[- ]system model\b", text)
    )


def _iam_vs_energy_system_answer() -> str:
    return (
        "### IAMs and energy-system models\n\n"
        "An integrated assessment model (IAM) connects multiple systems—commonly energy, "
        "the economy, land use, emissions and climate—to study interactions and policy pathways. "
        "An energy-system model concentrates more deeply on energy technologies, fuels, capacity, "
        "conversion and demand, often optimizing or simulating how the energy system evolves.\n\n"
        "The categories overlap: an IAM can contain a detailed energy-system component, and an "
        "energy-system model can include economic or emissions feedbacks. The practical distinction "
        "is breadth versus energy-detail, so check each model's documented boundary rather than "
        "assuming the label alone determines its capabilities."
    )


def _is_scenario_assumptions_concept_question(query: str) -> bool:
    text = re.sub(r"\s+", " ", str(query or "").strip().casefold())
    return bool(
        re.search(r"\bscenario assumptions?\b", text)
        and re.search(r"\b(?:affect|change|influence|shape|impact)\b", text)
        and re.search(r"\b(?:iam|model|output|result|pathway)s?\b", text)
    )


def _scenario_assumptions_concept_answer() -> str:
    return (
        "### How scenario assumptions affect IAM results\n\n"
        "Scenario assumptions define the conditions the model explores—for example population and "
        "GDP growth, policy strength, technology costs and availability, resource limits, land-use "
        "constraints and climate targets. Changing them alters the model's inputs or constraints, "
        "which can change energy mixes, emissions, prices, land use and technology deployment.\n\n"
        "IAM outputs are therefore conditional pathways, not unconditional forecasts. Compare "
        "scenarios with clearly documented assumptions, keep the model/version fixed when isolating "
        "one assumption, and inspect the experiment metadata before interpreting differences."
    )


def _load_skill_guidance(max_chars: int = 2000) -> str:
    skill_path = Path("skills/iam-timeseries-qa/SKILL.md")
    if not skill_path.exists():
        return ""
    text = skill_path.read_text()
    if text.lstrip().startswith("---"):
        parts = text.split("---", 2)
        if len(parts) == 3:
            text = parts[2]
    text = text.strip()
    if len(text) > max_chars:
        text = text[:max_chars].rstrip() + "\n\n[Skill guidance truncated]"
    return text
from query_extractor import QueryEntityExtractor


VALID_AGENT_NAMES = {
    "data_query",
    "data_plotting",
    "model_explanation",
    "general_qa",
    "modelling_suggestions",
}


class MultiAgentManager:
    def _conversation(self) -> ConversationState:
        """Return the typed state, lazily for lightweight test instances."""
        state = self.__dict__.get("conversation_state")
        if not isinstance(state, ConversationState):
            state = ConversationState()
            self.__dict__["conversation_state"] = state
        return state

    @property
    def last_entities(self) -> Dict[str, Any]:
        return self._conversation().active_scope

    @last_entities.setter
    def last_entities(self, value: Optional[Dict[str, Any]]) -> None:
        self._conversation().set_active(value)

    @property
    def previous_entities(self) -> Dict[str, Any]:
        return self._conversation().previous_scope

    @previous_entities.setter
    def previous_entities(self, value: Optional[Dict[str, Any]]) -> None:
        self._conversation().previous_scope = value

    @property
    def last_attempted_entities(self) -> Dict[str, Any]:
        return self._conversation().attempted_scope

    @last_attempted_entities.setter
    def last_attempted_entities(self, value: Optional[Dict[str, Any]]) -> None:
        self._conversation().attempted_scope = dict(value or {})

    @property
    def clarification_context(self) -> Optional[PendingClarification]:
        return self._conversation().pending_clarification

    @clarification_context.setter
    def clarification_context(self, value: Optional[Dict[str, Any]]) -> None:
        self._conversation().set_pending(value)

    def response_entities(self) -> Dict[str, Any]:
        """Scope to expose for this turn without replacing successful state."""
        return self._conversation().response_entities()

    def pending_clarification_payload(self) -> Dict[str, Any]:
        pending = self._conversation().pending_clarification
        return pending.api_payload() if pending else {}

    def __init__(self, shared_resources: Dict[str, Any], streaming: bool = True):
        self.shared_resources = shared_resources
        self.streaming = streaming
        self.logger = logging.getLogger(self.__class__.__name__)
        self.agents: Dict[str, BaseAgent] = {}
        self._initialize_agents()
        self.conversation_state = ConversationState()
        self.last_result_models: List[str] = []
        self.last_links: List[Dict[str, Any]] = []
        self.last_route_decision: Dict[str, Any] = {}
        self.turn_counter: int = 0
        self.current_turn: int = 0
        self.clarification_context: Optional[Dict[str, Any]] = None

        # The extractor's lookups (variables/regions/scenarios over all ts
        # records) are identical for every session; build once and share via
        # shared_resources so per-session manager creation stays cheap.
        shared_extractor = shared_resources.get("entity_extractor")
        if shared_extractor is not None:
            self.entity_extractor = shared_extractor
        else:
            extractor_api_key = str(
                (shared_resources.get("env") or {}).get("OPENAI_API_KEY")
                or os.getenv("OPENAI_API_KEY")
                or ""
            )
            self.entity_extractor = QueryEntityExtractor(
                models=shared_resources.get("models", []),
                ts_data=shared_resources.get("ts", []),
                api_key=extractor_api_key,
            )
            shared_resources["entity_extractor"] = self.entity_extractor

# LLM for intelligent query routing
        self.router_llm = ChatOpenAI(
            model_name=ROUTER_MODEL,
            temperature=0,
            streaming=False,
            timeout=30,
            max_retries=1
        )
        
        # Routing prompt
        skill_guidance = _load_skill_guidance()
        self.routing_prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(f"""You are a query classifier for an IAM PARIS climate data chatbot.

    CLASSIFY into ONE category:

    data_query - Questions about WHAT data exists:
    - "which models" "what models" "list models" "how many models"
    - "what scenarios" "list scenarios" "how many scenarios"  
    - "what variables" "list variables" "how many variables"
    - "show me all data" "what regions" "which variables"
    - Any question asking what models/scenarios/variables/regions are available

    data_plotting - Requests to CREATE CHARTS:
    - "plot" "graph" "chart" "visualize" data
    - Any request to show trends over time

    model_explanation - Questions EXPLAINING a MODEL:
    - "what is GCAM" "explain REMIND" "how does model work"
    - Specific model names with explain/what is

    modelling_suggestions - Study suggestions:
    - "suggest studies" "what to investigate" "research ideas"

    general_qa - General climate questions:
    - "climate change" "paris agreement" "policy"
    - General knowledge questions

    Respond with ONLY the category name, nothing else.

    Skill guidance (for routing context):
    {skill_guidance}"""),
                HumanMessagePromptTemplate.from_template("Query: {query}")
            ])

    def _is_site_navigation_request(self, query: str) -> bool:
        """Match navigation syntax to the currently loaded link catalogue."""
        return _looks_like_site_navigation_request(
            query,
            self.shared_resources.get("link_catalog", []),
        )

    def _looks_like_clarification_response(self, response: str) -> bool:
        text = str(response or "")
        markers = (
            "Choose the variable:",
            "Choose the region:",
            "Choose the scenario:",
            "Closest valid options:",
            "Closest variables:",
            "Closest regions:",
            "Closest scenarios:",
            "Which variable should I use?",
            "Which variable or region should I use instead?",
            "Please provide the",
            "Please clarify",
            "matched multiple",
            "I need one more detail",
            "I don't have an active numbered choice",
            "Reply with a number",
        )
        return any(marker.lower() in text.lower() for marker in markers) or bool(
            re.search(
                r"\bwhich\s+(?:variable|region|scenario|model)(?:\s+or\s+(?:variable|region|scenario|model))*\s+should\s+i\s+use\b",
                text,
                flags=re.IGNORECASE,
            )
        )

    def _append_relevant_links(
        self,
        response: str,
        query: str,
        entities: Optional[Dict[str, Any]],
        agent_name: str,
    ) -> str:
        if not response or "Relevant IAM PARIS links:" in response:
            return response
        is_clarification = self._looks_like_clarification_response(response)

        catalog = self.shared_resources.get("link_catalog", [])
        if not catalog:
            self.last_links = []
            return response

        try:
            variable_intent = _infer_variable_intent(query)
            links = suggest_links(
                query,
                catalog,
                agent_name=agent_name,
                entities=entities or {},
                variable_intent=variable_intent,
            )
            links = [
                link for link in links
                if float(link.get("confidence", 0) or 0) >= 0.4
            ]
            links = self._ensure_results_link_for_data_answer(links, catalog, agent_name)
        except Exception as err:
            self.logger.warning("Could not suggest IAM PARIS links: %s", err)
            self.last_links = []
            return response

        self.last_links = links
        formatted = format_relevant_links(links)
        if not formatted or is_clarification:
            return response
        return f"{response.rstrip()}\n\n{formatted}"

    def _grounded_site_navigation_answer(
        self,
        query: str,
        entities: Optional[Dict[str, Any]],
    ) -> str:
        catalog = self.shared_resources.get("link_catalog", [])
        if not catalog:
            self.last_links = []
            return (
                "I matched this as an IAM PARIS site/navigation request, but the link catalog is not loaded."
            )

        try:
            navigation_category = infer_navigation_category(query, catalog)
            links = suggest_links(
                query,
                catalog,
                agent_name="general_qa",
                entities=entities or {},
                variable_intent=_infer_variable_intent(query),
                navigation_category=navigation_category,
            )
            if links:
                top_confidence = max(float(link.get("confidence", 0) or 0) for link in links)
                cutoff = max(0.4, top_confidence - 0.15)
                links = [
                    link for link in links
                    if float(link.get("confidence", 0) or 0) >= cutoff
                ]
        except Exception as err:
            self.logger.warning("Could not build grounded IAM PARIS link answer: %s", err)
            self.last_links = []
            return "I could not match this request to a reliable IAM PARIS link."

        self.last_links = links
        if not links:
            return "I could not match this request to a reliable IAM PARIS link."

        grounded_destinations = []
        for link in links:
            title = str(link.get("title") or link.get("url") or "IAM PARIS page").strip()
            url = str(link.get("url") or "").strip()
            grounded_destinations.append(f"[{title}]({url})" if url else title)
        lines = [
            "Use these IAM PARIS links for this request: "
            + ", ".join(grounded_destinations)
            + "."
        ]
        formatted = format_relevant_links(links)
        if formatted:
            lines.extend(["", formatted])
        fallback_instructions = [
            str(link.get("fallback_instruction", "")).strip()
            for link in links
            if str(link.get("fallback_instruction", "")).strip()
        ]
        if fallback_instructions:
            lines.extend([
                "",
                "If a direct detail page is not available:",
                *[
                    f"- {instruction}"
                    for instruction in dict.fromkeys(fallback_instructions)
                ],
            ])
        return "\n".join(lines)

    def _ensure_results_link_for_data_answer(
        self,
        links: List[Dict[str, Any]],
        catalog: List[Dict[str, Any]],
        agent_name: str,
    ) -> List[Dict[str, Any]]:
        if agent_name not in {"data_query", "data_plotting"}:
            return links
        fallback = catalog_category_root(catalog, "results")
        if not fallback:
            return links
        if any(
            isinstance(link, dict) and catalog_link_matches_entry(link, fallback)
            for link in links
        ):
            return links
        result_link = {
            "title": str(fallback.get("title") or fallback.get("category") or "Results"),
            "url": str(fallback.get("url") or ""),
            "reason": "Catalogued results root for data follow-ups.",
            "confidence": 0.25,
            "search_hint": str(fallback.get("search_hint", "")),
            "category": str(fallback.get("category", "")),
        }
        deduped = [
            link for link in links
            if str(link.get("url", "")) != result_link["url"]
        ]
        # Never drop specific links to make room for the generic results page;
        # only pad when there is space left.
        if len(deduped) >= 3:
            return deduped
        return [*deduped, result_link]

    def _maybe_add_followup_guidance(self, response: str, query: str, agent_name: str) -> str:
        text = str(response or "").strip()
        if not text:
            return response
        if self._looks_like_clarification_response(text):
            return response
        if re.search(r"\breply with\b", text, re.IGNORECASE):
            return response
        # A workspace overview already contains its bounded study catalogue and
        # direct IAM PARIS destinations; adding generic data-filter guidance is
        # misleading and distracts from those links.
        if "Available in this study:" in text:
            return response
        answer_shape_is_real = bool(
            text.startswith("###")
            or text.startswith("Showing ")
            or text.startswith("No data found")
            or text.startswith("I could not find data")
            or text.startswith("Could not identify")
            or text.startswith("I found")
        )
        if not answer_shape_is_real:
            return response

        q = str(query or "").strip().lower()
        if not q or agent_name not in {"data_query", "data_plotting"}:
            return response

        # Discovery/list answers carry their own hints already.
        if any(
            marker in q
            for marker in (
                "list variables", "list models", "list regions", "list scenarios",
                "show all variables", "show all models", "show all regions",
                "which models are available",
            )
        ) and q != "show all scenarios":
            return response

        # Single rule: guide when the *answer* aggregates over an open scope
        # (multiple scenarios/models, or no explicit scope line at all) and the
        # *query* did not already pin scenario + year.
        scope_is_open = bool(
            re.search(r"(?:scenario|model)\s+`multiple`", text, re.IGNORECASE)
            or (
                "Scope:" not in text
                and not re.search(r"for\s+scenario\s+`[^`]+`", text, re.IGNORECASE)
            )
        )
        if not scope_is_open:
            return response

        has_year_filter = bool(extract_year_range(query)[0] or extract_year_range(query)[1])
        names_scenario = bool(self._match_scenario_from_text(query)) or any(
            term in q for term in ("baseline", "current policy", "current policies")
        )
        if has_year_filter or "latest" in q:
            return response

        return (
            f"{text}\n\n"
            "Reply with a scenario, model, region, or year to narrow the answer."
        )

    def _workspace_result_answer(self, query: str, response: str) -> str:
        text = str(response or "").strip()
        q = str(query or "").lower()
        # Only redirect to the IAM COMPACT workspace when the query really is
        # about it: an explicit project mention, or net-zero *in an EU context*.
        # A generic failed "net zero" question must not get this answer.
        explicit_project = any(term in q for term in ("fit-for-55", "fit for 55", "iam compact"))
        net_zero_eu = (
            any(term in q for term in ("net zero", "net-zero"))
            and bool(re.search(r"\beu\b|europe", q))
        )
        if not (explicit_project or net_zero_eu):
            return response
        if not (
            not text
            or "i need one more detail" in text.lower()
            or "please specify the variable, region, or scenario" in text.lower()
        ):
            return response
        return (
            "The best match is the IAM COMPACT results workspace for Fit-for-55 and EU net-zero pathways. "
            "Use the IAM PARIS links below to open the relevant policy-question workspace and related net-zero results."
        )

    def _model_metadata_fallback_answer(self, query: str, response: str, entities: Optional[Dict[str, Any]]) -> str:
        text = str(response or "").strip()
        model = str((entities or {}).get("model") or "").strip()
        if not model:
            return response
        if not (
            not text
            or "i need one more detail" in text.lower()
            or "please specify the variable, region, or scenario" in text.lower()
        ):
            return response
        profile = find_model_profile(model) or find_model_profile(query)
        if profile:
            return format_model_profile_answer(
                profile,
                requested_name=str(profile.get("name", "") or model),
                asks_assumptions=bool(re.search(r"\bassumption\b|\bassumptions\b", str(query or "").lower())),
                query=query,
            )
        return (
            f"I matched `{model}`, but IAM PARIS does not expose a dedicated assumptions/metadata page "
            "for that model in the local model catalog. Use the IAM PARIS model and results links below "
            "to inspect related documentation or available data."
        )

    def _runtime_model_mentions(self, query: str) -> List[str]:
        """Resolve model mentions against the live catalogue.

        Exact full labels are collected first.  Natural-language comparisons
        also commonly use a family label while the catalogue stores a versioned
        or institute-qualified name.  Resolve each grammatical clause through
        the catalogue alias resolver and keep its best grounded candidate.  No
        model family is encoded here; newly added catalogue values participate
        automatically.
        """
        identity_query = re.sub(
            r"\b(?:without|do\s+not|don't)\s+(?:confus\w*|mix\w*)\b[^.?!]*",
            " ",
            str(query or ""),
            flags=re.IGNORECASE,
        )
        runtime_names = list(getattr(self.entity_extractor, "available_models", []) or [])
        runtime_names.extend(
            str(record.get("modelName") or "").strip()
            for record in (self.shared_resources.get("models", []) or [])
            if isinstance(record, dict) and str(record.get("modelName") or "").strip()
        )
        runtime_names.extend(
            str(item.get("search_hint") or item.get("title") or "").strip()
            for item in (self.shared_resources.get("link_catalog", []) or [])
            if isinstance(item, dict)
            and str(item.get("category") or "").casefold() == "models"
            and str(item.get("item_type") or "").casefold() == "model"
            and str(item.get("search_hint") or item.get("title") or "").strip()
            and not re.fullmatch(
                r"\d+(?:\.\d+)?",
                str(item.get("search_hint") or item.get("title") or "").strip(),
            )
        )
        runtime_names = list(dict.fromkeys(runtime_names))
        mentioned = list(
            build_query_plan(identity_query, available_models=runtime_names).mentioned_models
        )

        clauses = re.split(
            r"[,;/]|\b(?:and|with|against|versus|vs\.?|from|to)\b",
            identity_query,
            flags=re.IGNORECASE,
        )
        for clause in clauses:
            candidates = resolve_model_candidates(clause, runtime_names)
            if not candidates:
                continue
            candidate = str(candidates[0]).strip()
            if candidate and all(candidate.casefold() != value.casefold() for value in mentioned):
                mentioned.append(candidate)
        deduplicated: List[str] = []
        seen_families: set[str] = set()
        for value in mentioned:
            family = model_family_key(value)
            if not family or family in seen_families:
                continue
            seen_families.add(family)
            deduplicated.append(value)
        return deduplicated

    def _runtime_model_profiles(self, query: str) -> List[Dict[str, Any]]:
        """Return grounded profiles for every runtime model named in *query*.

        Curated profiles remain preferred, while models without one are built
        from the loaded model catalogue.  This makes qualitative comparisons
        work for newly added models without adding their names to routing code.
        """
        runtime_names = list(getattr(self.entity_extractor, "available_models", []) or [])
        mentioned = self._runtime_model_mentions(query)

        # Runtime names commonly include a version suffix while users refer to
        # the model family (for example ``Family`` instead of ``Family 2.0``).
        # Merge profile-backed mentions with catalogue mentions so a broad
        # comparison can still identify every named model.  The values remain
        # grounded in the loaded catalogue/profile records; routing has no
        # model-name-specific branches.
        identity_query = re.sub(
            r"\b(?:without|do\s+not|don't)\s+(?:confus\w*|mix\w*)\b[^.?!]*",
            " ",
            str(query or ""),
            flags=re.IGNORECASE,
        )
        for profile in find_model_profiles(identity_query):
            name = str(profile.get("name") or "").strip()
            mentioned_families = {model_family_key(value) for value in mentioned}
            if name and model_family_key(name) not in mentioned_families:
                mentioned.append(name)
        profiles: List[Dict[str, Any]] = []

        def _catalog_text(*values: Any) -> str:
            for value in values:
                text = str(value or "").strip()
                if text and text.casefold() not in {"nan", "nat", "none", "null"}:
                    return text
            return ""

        for requested in mentioned:
            curated = find_model_profile(requested)
            candidates = resolve_model_candidates(requested, runtime_names) or [requested]
            candidate_keys = {str(value).casefold() for value in candidates}
            if curated:
                curated_name = str(curated.get("name") or "").strip()
                candidate_keys.update(
                    str(value).casefold()
                    for value in resolve_model_candidates(curated_name, runtime_names)
                )
            record = next(
                (
                    item for item in self.shared_resources.get("models", [])
                    if isinstance(item, dict)
                    and str(item.get("modelName") or "").casefold() in candidate_keys
                ),
                None,
            )
            if curated:
                profile = dict(curated)
                if any(str(requested).casefold() == str(value).casefold() for value in runtime_names):
                    profile["name"] = str(requested)
                    profile["search_hint"] = str(requested)
            else:
                if not record:
                    continue
                profile = {
                    "name": str(record.get("modelName") or requested).strip(),
                    "limitations": [],
                    "search_hint": str(record.get("modelName") or requested).strip(),
                }
            # Curated profiles provide stable summaries, while the runtime
            # catalogue may carry richer or newer structured method fields.
            # Fill only missing fields so neither source erases the other.
            if record:
                runtime_fields = {
                    "description": _catalog_text(record.get("description"), record.get("overview")),
                    "developer": _catalog_text(record.get("institute")),
                    "technology_note": _catalog_text(record.get("measures_technologies")),
                    "methodology_note": _catalog_text(
                        record.get("model_type"), record.get("economic_rationale")
                    ),
                    "assumptions_note": _catalog_text(record.get("key_parameters")),
                }
                for key, value in runtime_fields.items():
                    if not _catalog_text(profile.get(key)) and value:
                        profile[key] = value
                for key in (
                    "iamparis_model_url", "model_url", "modelUrl", "url", "URL",
                    "route", "path", "slug", "href",
                    "model_id", "modelId", "modelID", "id", "ID",
                    "pk", "modelPk", "model_pk",
                ):
                    value = _catalog_text(record.get(key))
                    if value and not _catalog_text(profile.get(key)):
                        profile[key] = value
            name = str(profile.get("name") or "").strip()
            if name and all(str(existing.get("name") or "") != name for existing in profiles):
                profiles.append(profile)
        return profiles

    def _models_covering_topic_answer(self, query: str) -> Optional[str]:
        """N4: when a model-list request carries a sector/topic qualifier, return a
        deterministic subset of models that report data for that topic instead of the
        full model list. Returns None when no topic is detected or metadata is missing."""
        metadata = self.shared_resources.get("metadata")
        if not metadata or not hasattr(metadata, "models_covering_topic"):
            return None
        if hasattr(metadata, "models_covering_topics"):
            matches = metadata.models_covering_topics(query)
        else:
            category, models = metadata.models_covering_topic(query)
            matches = [(category, models)] if category and models else []
        matches = [(category, models) for category, models in matches if category and models]
        if not matches:
            return None
        if hasattr(metadata, "distinct_model_labels"):
            total = len(metadata.distinct_model_labels())
        else:
            total = len(metadata.all_model_names) if hasattr(metadata, "all_model_names") else None
        if len(matches) > 1:
            combined = sorted({model for _category, models in matches for model in models})
            labels = " or ".join(category for category, _models in matches)
            lines = [f"### Models covering {labels}", ""]
            for category, models in matches:
                shown = models[:20]
                more = len(models) - len(shown)
                values = ", ".join(shown) + (f" … and {more} more" if more > 0 else "")
                lines.extend([f"**{category} ({len(models)}):** {values}", ""])
            lines.append(
                f"Combined coverage: {len(combined)} distinct model(s)"
                + (f" of {total}" if total else "")
                + "."
            )
            return "\n".join(lines)

        category, models = matches[0]
        shown = models[:20]
        more = len(models) - len(shown)
        lines = [
            f"### Models covering {category}",
            "",
            f"{len(models)} model(s)"
            + (f" of {total}" if total else "")
            + f" report at least one {category.lower()} variable in IAM PARIS:",
            "",
        ]
        lines.append(", ".join(shown) + (f" … and {more} more" if more > 0 else ""))
        lines.append("")
        lines.append(f"Ask for a specific model (e.g. `tell me about {shown[0]}`) or a data query "
                     f"(e.g. `{category.lower()} emissions for Europe`) to go deeper.")
        return "\n".join(lines)

    def _scoped_availability_answer(
        self,
        query: str,
        targets: tuple[str, ...],
        scope: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """Project requested dimensions from the runtime availability matrix.

        Requested dimensions are outputs, while every other grounded dimension
        is a filter.  Keeping those roles separate prevents a stale singular
        scenario/model from narrowing a plural comparison and makes contextual
        pronouns reuse the exact prior variable scope.
        """
        metadata = self.shared_resources.get("metadata")
        if metadata is None:
            return None

        entities = dict(scope or {})
        all_variables = sorted(getattr(metadata, "all_variables", set()) or [])
        carried_variable = str(entities.get("variable") or "").strip()
        explicit_variable = (
            self._match_catalog_value_from_text(query, all_variables)
            or preferred_variable_from_query(query, all_variables)
        )
        variable = explicit_variable or carried_variable
        if not variable and not carried_variable:
            extracted = self.entity_extractor.extract(query) or {}
            variable = sanitize_variable_for_query(extracted.get("variable"), query)
        if not variable:
            return None
        info = metadata.get_available_for_variable(variable)
        resolved = info.get("variable")
        if not resolved:
            return None

        requested = [
            target for target in targets
            if target in {"region", "scenario", "model"}
        ]
        if not requested:
            return None

        rows: List[Dict[str, Any]] = []
        matrix = getattr(metadata, "availability_matrix", {}) or {}
        region_map = matrix.get(resolved, {}) if isinstance(matrix, dict) else {}
        if isinstance(region_map, dict) and region_map:
            for region, scenario_map in region_map.items():
                for scenario, model_map in (scenario_map or {}).items():
                    for model, years in (model_map or {}).items():
                        rows.append({
                            "region": str(region or ""),
                            "scenario": str(scenario or ""),
                            "model": str(model or ""),
                            "years": {int(year) for year in (years or set()) if str(year).isdigit()},
                        })
        else:
            # Compatibility with lightweight metadata fixtures and older cache
            # objects that predate availability_matrix.
            for record in self.shared_resources.get("ts", []):
                if not isinstance(record, dict) or str(record.get("variable") or "") != resolved:
                    continue
                years = {int(key) for key in record if str(key).isdigit()}
                if isinstance(record.get("years"), dict):
                    years.update(int(key) for key in record["years"] if str(key).isdigit())
                rows.append({
                    "region": str(record.get("region") or ""),
                    "scenario": str(record.get("scenario") or ""),
                    "model": str(record.get("modelName") or record.get("model") or ""),
                    "years": years,
                })

        def _scope_values(singular: str, plural: str) -> List[str]:
            raw = entities.get(plural)
            if raw not in (None, "", []):
                values = raw if isinstance(raw, (list, tuple, set)) else [raw]
            else:
                raw = entities.get(singular)
                values = [] if raw in (None, "") else [raw]
            return [str(value).strip() for value in values if str(value or "").strip()]

        all_regions = sorted({row["region"] for row in rows if row["region"]})
        all_scenarios = sorted({row["scenario"] for row in rows if row["scenario"]})
        all_models = sorted({row["model"] for row in rows if row["model"]})

        explicit_region = (
            self._match_catalog_value_from_text(query, all_regions)
            or canonical_region_from_query(query, all_regions)
        )
        explicit_scenarios = explicit_scenarios_from_query(query, all_scenarios)
        if not explicit_scenarios:
            scenario = (
                self._match_catalog_value_from_text(query, all_scenarios)
                or canonical_scenario_from_query(query, all_scenarios)
            )
            explicit_scenarios = [scenario] if scenario else []
        mentioned_models = list(build_query_plan(
            query, available_models=all_models,
        ).mentioned_models)
        if not mentioned_models:
            mentioned_models = list(resolve_model_candidates(query, all_models))

        filters: Dict[str, List[str]] = {
            "region": ([explicit_region] if explicit_region else _scope_values("region", "regions")),
            "scenario": (explicit_scenarios or _scope_values("scenario", "scenarios")),
            "model": (mentioned_models or _scope_values("model", "models")),
        }
        # A requested dimension is projected, never constrained by stale state.
        for target in requested:
            filters[target] = []

        resolved_models: List[str] = []
        for value in filters["model"]:
            matches = resolve_model_candidates(value, all_models)
            for match in matches or [value]:
                if str(match).casefold() not in {item.casefold() for item in resolved_models}:
                    resolved_models.append(str(match))
        filters["model"] = resolved_models

        start_year = entities.get("start_year")
        end_year = entities.get("end_year")
        explicit_year_filter = extract_year_filter(query)
        if explicit_year_filter.explicit:
            start_year = explicit_year_filter.start_year
            end_year = explicit_year_filter.end_year

        def _matches(dimension: str, actual: str, expected: str) -> bool:
            if actual.casefold() == expected.casefold():
                return True
            return dimension == "scenario" and scenario_in_family(actual, expected)

        def _row_matches(row: Dict[str, Any]) -> bool:
            for dimension in ("region", "scenario", "model"):
                wanted = filters[dimension]
                if wanted and not any(_matches(dimension, row[dimension], value) for value in wanted):
                    return False
            if start_year is not None or end_year is not None:
                if not any(
                    (start_year is None or year >= int(start_year))
                    and (end_year is None or year <= int(end_year))
                    for year in row["years"]
                ):
                    return False
            return True

        eligible_rows = [row for row in rows if _row_matches(row)]
        projected: Dict[str, List[str]] = {}
        for target in requested:
            candidates = sorted({row[target] for row in eligible_rows if row[target]})
            # For a bounded plural filter, report only target values that cover
            # every selected member. This gives comparisons intersection
            # semantics instead of returning a misleading union.
            fully_covered: List[str] = []
            for candidate in candidates:
                candidate_rows = [
                    row for row in eligible_rows
                    if row[target].casefold() == candidate.casefold()
                ]
                covers_all = all(
                    all(
                        any(_matches(dimension, row[dimension], expected) for row in candidate_rows)
                        for expected in selected
                    )
                    for dimension, selected in filters.items()
                    if len(selected) > 1
                )
                if covers_all:
                    fully_covered.append(candidate)
            projected[target] = fully_covered

        labels = {"region": "Regions", "scenario": "Scenarios", "model": "Models"}
        lines = [f"### Availability for {resolved}", ""]
        for target in requested:
            title = labels[target]
            values = projected.get(target, [])
            sample = values[:20]
            suffix = f" … and {len(values) - len(sample)} more" if len(values) > len(sample) else ""
            lines.append(f"**{title} ({len(values)}):** " + (", ".join(sample) if sample else "none") + suffix)
            lines.append("")
        unit = info.get("unit")
        if unit:
            lines.append(f"Recorded unit: `{unit}`.")
        return "\n".join(lines).strip()

    def _scoped_variable_availability_answer(
        self,
        query: str,
        scope: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """List variables after applying explicit/carried catalogue filters."""
        records = [
            record for record in self.shared_resources.get("ts", [])
            if isinstance(record, dict)
        ]
        if not records:
            return None
        entities = dict(scope or {})

        runtime_models = sorted({
            str(record.get("modelName") or record.get("model") or "").strip()
            for record in records
            if str(record.get("modelName") or record.get("model") or "").strip()
        })
        explicit_models = list(build_query_plan(
            query, available_models=runtime_models,
        ).mentioned_models)
        alias_model_groups: List[set[str]] = []
        if not explicit_models:
            grouped_aliases: Dict[str, set[str]] = {}
            for candidate in resolve_model_candidates(query, runtime_models):
                grouped_aliases.setdefault(model_family_key(candidate), set()).add(str(candidate))
            alias_model_groups = [
                grouped_aliases[key] for key in sorted(grouped_aliases) if key
            ]
        carried_models = entities.get("models")
        if carried_models not in (None, "", []):
            carried_models = (
                list(carried_models)
                if isinstance(carried_models, (list, tuple, set))
                else [carried_models]
            )
        elif entities.get("model"):
            carried_models = [entities.get("model")]
        else:
            carried_models = []
        requested_models = [
            str(value).strip() for value in (explicit_models or carried_models)
            if str(value or "").strip()
        ]
        # A whole-query alias guess can pick up a region word that collides with
        # a model's first token (e.g. "for the EU" -> model "eu_times"). It must
        # not override a model already carried from context (a pronoun like
        # "it"); use it only when no model is carried, so "what variables does
        # it have for the EU?" stays scoped to the carried model.
        if alias_model_groups and not carried_models:
            model_groups = alias_model_groups
            requested_models = [sorted(group)[0] for group in alias_model_groups]
        else:
            model_groups = [
                set(resolve_model_candidates(value, runtime_models) or [value])
                for value in requested_models
            ]
        model_candidates = set().union(*model_groups) if model_groups else set()

        runtime_regions = sorted({
            str(record.get("region") or "").strip() for record in records
            if str(record.get("region") or "").strip()
        })
        explicit_region = (
            self._match_catalog_value_from_text(query, runtime_regions)
            or canonical_region_from_query(query, runtime_regions)
        )
        carried_regions = entities.get("regions")
        if carried_regions not in (None, "", []):
            carried_regions = (
                list(carried_regions)
                if isinstance(carried_regions, (list, tuple, set))
                else [carried_regions]
            )
        elif entities.get("region"):
            carried_regions = [entities.get("region")]
        else:
            carried_regions = []
        region_values = [explicit_region] if explicit_region else [
            str(value).strip() for value in carried_regions if str(value or "").strip()
        ]
        scenario_values = [
            str(value).strip() for value in (entities.get("scenarios") or [])
            if str(value or "").strip()
        ]
        runtime_scenarios = sorted({
            str(record.get("scenario") or "").strip() for record in records
            if str(record.get("scenario") or "").strip()
        })
        explicit_scenarios = explicit_scenarios_from_query(query, runtime_scenarios)
        if not explicit_scenarios:
            explicit_scenario = (
                self._match_catalog_value_from_text(query, runtime_scenarios)
                or canonical_scenario_from_query(query, runtime_scenarios)
            )
            explicit_scenarios = [explicit_scenario] if explicit_scenario else []
        if explicit_scenarios:
            scenario_values = [str(value) for value in explicit_scenarios]
        elif not scenario_values and entities.get("scenario"):
            scenario_values = [str(entities.get("scenario"))]
        parsed_year_filter = extract_year_filter(query)
        start_year, end_year = parsed_year_filter.start_year, parsed_year_filter.end_year
        if not parsed_year_filter.explicit:
            start_year = entities.get("start_year")
            end_year = entities.get("end_year")

        if not (
            model_candidates or region_values or scenario_values
            or start_year is not None or end_year is not None
        ):
            return None

        matching_records: List[Dict[str, Any]] = []
        for record in records:
            record_model = str(record.get("modelName") or record.get("model") or "").strip()
            if model_candidates and record_model not in model_candidates:
                continue
            record_region = str(record.get("region") or "")
            if region_values and not any(
                record_region.casefold() == region.casefold() for region in region_values
            ):
                continue
            record_scenario = str(record.get("scenario") or "")
            if scenario_values and not any(
                record_scenario.casefold() == scenario.casefold()
                or scenario_in_family(record_scenario, scenario)
                for scenario in scenario_values
            ):
                continue
            if start_year is not None or end_year is not None:
                years = {int(key) for key in record if str(key).isdigit()}
                if isinstance(record.get("years"), dict):
                    years.update(int(key) for key in record["years"] if str(key).isdigit())
                if not any(
                    (start_year is None or year >= int(start_year))
                    and (end_year is None or year <= int(end_year))
                    for year in years
                ):
                    continue
            matching_records.append(record)

        candidates = sorted({
            str(record.get("variable") or "").strip()
            for record in matching_records
            if str(record.get("variable") or "").strip()
        })
        variables: List[str] = []
        for variable in candidates:
            variable_records = [
                record for record in matching_records
                if str(record.get("variable") or "") == variable
            ]
            covers_models = all(
                any(
                    str(record.get("modelName") or record.get("model") or "") in group
                    for record in variable_records
                )
                for group in model_groups
            )
            covers_regions = all(
                any(str(record.get("region") or "").casefold() == region.casefold() for record in variable_records)
                for region in region_values
            )
            covers_scenarios = all(
                any(
                    str(record.get("scenario") or "").casefold() == scenario.casefold()
                    or scenario_in_family(str(record.get("scenario") or ""), scenario)
                    for record in variable_records
                )
                for scenario in scenario_values
            )
            if covers_models and covers_regions and covers_scenarios:
                variables.append(variable)

        values = variables
        shown = values[:20]
        suffix = f" … and {len(values) - len(shown)} more" if len(values) > len(shown) else ""
        scope_parts = []
        if requested_models:
            label = "models" if len(requested_models) > 1 else "model"
            scope_parts.append(label + " " + ", ".join(f"`{value}`" for value in requested_models))
        if region_values:
            label = "regions" if len(region_values) > 1 else "region"
            scope_parts.append(label + " " + ", ".join(f"`{value}`" for value in region_values))
        if scenario_values:
            scope_parts.append("scenario " + ", ".join(f"`{value}`" for value in scenario_values))
        lines = ["### Variables available for the requested scope", ""]
        if scope_parts:
            lines.extend(["Scope: " + ", ".join(scope_parts) + ".", ""])
        lines.append(f"**Variables ({len(values)}):** " + (", ".join(shown) if shown else "none") + suffix)
        return "\n".join(lines)

    def _referenced_models_availability_answer(
        self,
        query: str,
        referenced_models: List[str],
    ) -> Optional[str]:
        """Compare data availability only across models carried in state.

        Model names, variables, regions and scenarios are resolved from the
        loaded catalogues. This keeps a follow-up such as "which one reports
        this variable?" bounded to the previously compared set.
        """
        records = [record for record in self.shared_resources.get("ts", []) if isinstance(record, dict)]
        if not records or len(referenced_models) < 2:
            return None

        variables = getattr(self.entity_extractor, "available_variables", []) or sorted({
            str(record.get("variable", "")).strip()
            for record in records if str(record.get("variable", "")).strip()
        })
        variable = self._match_catalog_value_from_text(query, variables)
        if not variable:
            variable = preferred_variable_from_query(query, variables) or ""
        if not variable:
            return None

        regions = getattr(self.entity_extractor, "available_regions", []) or sorted({
            str(record.get("region", "")).strip()
            for record in records if str(record.get("region", "")).strip()
        })
        region = self._match_catalog_value_from_text(query, regions)
        if not region:
            try:
                region = extract_region_from_query(
                    query,
                    getattr(self.entity_extractor, "region_dict", {}) or {},
                    list(regions),
                )
            except Exception:
                region = ""
        scenarios = getattr(self.entity_extractor, "available_scenarios", []) or sorted({
            str(record.get("scenario", "")).strip()
            for record in records if str(record.get("scenario", "")).strip()
        })
        scenario = self._match_catalog_value_from_text(query, scenarios)
        if not scenario:
            scenario = canonical_scenario_from_query(query, scenarios) or ""
        start_year, end_year = extract_year_range(query)

        timeseries_models = sorted({
            str(record.get("modelName") or record.get("model") or "").strip()
            for record in records
            if str(record.get("modelName") or record.get("model") or "").strip()
        })

        def _record_has_requested_year(record: Dict[str, Any]) -> bool:
            if start_year is None and end_year is None:
                return True
            years = {
                int(key) for key in record
                if str(key).isdigit()
            }
            nested = record.get("years")
            if isinstance(nested, dict):
                years.update(int(key) for key in nested if str(key).isdigit())
            return any(
                (start_year is None or year >= int(start_year))
                and (end_year is None or year <= int(end_year))
                for year in years
            )

        statuses: List[Tuple[str, List[str]]] = []
        for reference in referenced_models:
            resolved_names = resolve_model_candidates(str(reference), timeseries_models)
            matching_names = set(resolved_names)
            matched = set()
            for record in records:
                record_model = str(record.get("modelName") or record.get("model") or "").strip()
                if record_model not in matching_names:
                    continue
                if str(record.get("variable", "")).casefold() != str(variable).casefold():
                    continue
                if region and str(record.get("region", "")).casefold() != str(region).casefold():
                    continue
                record_scenario = str(record.get("scenario", ""))
                if scenario and not (
                    record_scenario.casefold() == str(scenario).casefold()
                    or scenario_in_family(record_scenario, str(scenario))
                ):
                    continue
                if not _record_has_requested_year(record):
                    continue
                matched.add(record_model)
            statuses.append((str(reference), sorted(matched)))

        scope_parts = [f"variable `{variable}`"]
        if region:
            scope_parts.append(f"region `{region}`")
        if scenario:
            scope_parts.append(f"scenario `{scenario}`")
        if start_year is not None or end_year is not None:
            if start_year is None:
                scope_parts.append(f"years through `{end_year}`")
            elif end_year is None:
                scope_parts.append(f"years from `{start_year}`")
            else:
                scope_parts.append(
                    f"year `{start_year}`" if end_year == start_year
                    else f"years `{start_year}–{end_year}`"
                )
        lines = ["### Availability among the referenced models", "", "Scope: " + ", ".join(scope_parts) + ".", ""]
        for reference, matched in statuses:
            if matched:
                detail = ", ".join(f"`{name}`" for name in matched[:4])
                suffix = f" and {len(matched) - 4} more" if len(matched) > 4 else ""
                lines.append(f"- `{reference}`: available via {detail}{suffix}.")
            else:
                lines.append(f"- `{reference}`: no matching rows in the loaded time-series data.")
        return "\n".join(lines)

    def _closest_available_variable(
        self,
        canonical: str,
        tokens: Tuple[str, ...],
        available_variables: set,
    ) -> str:
        """Return `canonical` when availability is unknown or confirmed; else the
        shortest available variable containing all `tokens`; else ""."""
        if not available_variables or canonical in available_variables:
            return canonical
        candidates = [
            variable for variable in available_variables
            if all(token in variable.lower() for token in tokens)
        ]
        return min(candidates, key=len) if candidates else ""

    def _repair_comparison_entities(self, query: str, entities: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        repaired = dict(entities or {})
        q = str(query or "").lower()

        available_variables = {
            str(record.get("variable", "") or "")
            for record in self.shared_resources.get("ts", [])
            if isinstance(record, dict) and record.get("variable")
        }

        if re.search(r"\b(greenhouse gas|greenhouse gases|ghg)\b", q):
            ghg_variable = self._closest_available_variable(
                "Emissions|GHG", ("emissions", "ghg"), available_variables
            ) or self._closest_available_variable(
                "Emissions|Kyoto Gases", ("emissions", "kyoto"), available_variables
            )
            if ghg_variable:
                repaired["variable"] = ghg_variable
                confidence = dict(repaired.get("entity_confidence") or {})
                confidence["variable"] = max(float(confidence.get("variable", 0) or 0), 0.9)
                repaired["entity_confidence"] = confidence
        preferred_variable = preferred_variable_from_query(query, available_variables)
        existing_variable = str(repaired.get("variable", "") or "").strip()
        explicit_existing_variable = bool(existing_variable and existing_variable in str(query or ""))
        if preferred_variable and not explicit_existing_variable:
            repaired["variable"] = preferred_variable
            confidence = dict(repaired.get("entity_confidence") or {})
            confidence["variable"] = max(float(confidence.get("variable", 0) or 0), 0.9)
            repaired["entity_confidence"] = confidence

        available_scenarios = {
            str(record.get("scenario", "") or "")
            for record in self.shared_resources.get("ts", [])
            if isinstance(record, dict) and record.get("scenario")
        }
        scenario = canonical_scenario_from_query(query, available_scenarios)
        if scenario:
            repaired["scenario"] = scenario
            confidence = dict(repaired.get("entity_confidence") or {})
            confidence["scenario"] = max(float(confidence.get("scenario", 0) or 0), 0.9)
            repaired["entity_confidence"] = confidence

        if not (
            _looks_like_comparison_request(query)
            or self._is_textual_comparison_question(query)
        ):
            return repaired

        scenario_pair = re.search(
            r"\bunder\s+(.+?)\s+(?:versus|vs|against|compared\s+with|compared\s+to)\s+(.+?)(?:\s+for\s+model\b|$)",
            query,
            re.IGNORECASE,
        )
        if scenario_pair:
            scenarios = []
            for raw_scenario in scenario_pair.groups():
                matched = canonical_scenario_from_query(raw_scenario, available_scenarios)
                if matched and matched not in scenarios:
                    scenarios.append(matched)
            if len(scenarios) >= 2:
                repaired["scenarios"] = scenarios
                repaired["scenario"] = None
                repaired["comparison"] = "scenario"
                confidence = dict(repaired.get("entity_confidence") or {})
                confidence["scenario"] = max(float(confidence.get("scenario", 0) or 0), 0.9)
                confidence["comparison"] = max(float(confidence.get("comparison", 0) or 0), 0.9)
                repaired["entity_confidence"] = confidence

        has_wind = re.search(r"\bwind\b", q)
        has_solar = re.search(r"\b(solar|pv|photovoltaic|photovoltaics)\b", q)
        has_capacity_intent = re.search(r"\b(capacity|power|installed|pv)\b", q)
        if not (has_wind and has_solar and has_capacity_intent):
            return repaired

        # Validate the canonical wind/solar capacity variables against the
        # loaded data; fall back to the closest available variant instead of
        # forcing names that would yield a "no data" answer.
        wind_variable = self._closest_available_variable(
            "Capacity|Electricity|Wind", ("capacity", "wind"), available_variables
        )
        solar_variable = self._closest_available_variable(
            "Capacity|Electricity|Solar", ("capacity", "solar"), available_variables
        )
        if not wind_variable or not solar_variable:
            return repaired

        variables = [wind_variable, solar_variable]
        existing = repaired.get("variables")
        if isinstance(existing, list):
            for variable in existing:
                if variable and variable not in variables:
                    variables.append(str(variable))

        repaired["variable"] = wind_variable
        repaired["variables"] = variables
        repaired["comparison"] = repaired.get("comparison") or "variable"
        confidence = dict(repaired.get("entity_confidence") or {})
        confidence["variable"] = max(float(confidence.get("variable", 0) or 0), 0.9)
        confidence["comparison"] = max(float(confidence.get("comparison", 0) or 0), 0.85)
        repaired["entity_confidence"] = confidence
        return repaired

    def _is_textual_comparison_question(self, query: str) -> bool:
        """Interrogative comparisons ("which is higher, solar or wind?") expect a
        numeric/textual answer, not a forced chart."""
        q = str(query or "").lower()
        if _looks_like_plot_request(query):
            return False
        comparative = r"(?:higher|larger|bigger|greater|lower|smaller|more|less)"
        return bool(
            re.search(r"\bwhich\s+(?:one\s+)?(?:is|was|will\s+be|has|had)\b.*\b" + comparative + r"\b", q)
            or re.search(r"\bis\s+\S+.*\b" + comparative + r"\s+than\b", q)
        )

    def _textual_comparison_answer(self, query: str, entities: Optional[Dict[str, Any]]) -> str:
        """Answer "which is higher, X or Y?" with values from the loaded data.
        Returns "" when the question or data do not support a grounded answer."""
        if not self._is_textual_comparison_question(query):
            return ""
        entities = entities or {}
        variables = [str(v) for v in (entities.get("variables") or []) if v]
        if len(variables) < 2:
            return ""
        var_a, var_b = variables[0], variables[1]
        ts = self.shared_resources.get("ts") or []
        region = str(entities.get("region") or "").strip()
        scenario = str(entities.get("scenario") or "").strip()

        def _slices(var: str) -> Dict[Tuple[str, str, str], Dict[str, Any]]:
            out: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
            for rec in ts:
                if not isinstance(rec, dict) or str(rec.get("variable") or "") != var:
                    continue
                if region and str(rec.get("region") or "") != region:
                    continue
                if scenario and str(rec.get("scenario") or "") != scenario:
                    continue
                key = (
                    str(rec.get("region") or ""),
                    str(rec.get("scenario") or ""),
                    str(rec.get("modelName") or rec.get("model") or ""),
                )
                out.setdefault(key, rec)
            return out

        slices_a = _slices(var_a)
        slices_b = _slices(var_b)
        common = sorted(set(slices_a) & set(slices_b))
        if not common:
            return ""
        if not region:
            world_keys = [key for key in common if key[0].lower() == "world"]
            if world_keys:
                common = world_keys
        key = common[0]
        rec_a, rec_b = slices_a[key], slices_b[key]
        years_a = {str(y): v for y, v in (rec_a.get("years") or {}).items()}
        years_b = {str(y): v for y, v in (rec_b.get("years") or {}).items()}
        common_years = sorted(set(years_a) & set(years_b))
        if not common_years:
            return ""
        start_year, end_year = extract_year_range(query)
        target = str(end_year or start_year or "")
        year = target if target in common_years else common_years[-1]
        try:
            val_a = float(years_a[year])
            val_b = float(years_b[year])
        except (TypeError, ValueError):
            return ""

        unit_a = str(rec_a.get("unit") or "").strip()
        unit_b = str(rec_b.get("unit") or "").strip()
        region_key, scenario_key, model_key = key
        model_display = display_model_label(model_key)
        if val_a == val_b:
            verdict = f"Both are equal in {year}."
        else:
            higher = var_a if val_a > val_b else var_b
            verdict = f"`{higher}` is higher in {year}."
        record_resolved_scope(
            variable=var_a,
            region=region_key,
            scenario=scenario_key,
            model=model_display,
        )
        lines = [
            f"### Comparison — {var_a} vs {var_b} ({region_key})",
            "",
            f"In {year} under scenario `{scenario_key}` (model `{model_display}`):",
            f"- `{var_a}`: {val_a:,.2f} {unit_a}".rstrip(),
            f"- `{var_b}`: {val_b:,.2f} {unit_b}".rstrip(),
            "",
            verdict,
        ]
        if unit_a and unit_b and unit_a != unit_b:
            lines.append("Note: the two variables use different units, so compare with care.")
        lines.append(f"Ask `plot compare {var_a} versus {var_b}` to see the full trajectories.")
        return "\n".join(lines)

    def _ordered_variable_clarification_candidates(
        self,
        query: str,
        entities: Dict[str, Any],
        candidates: List[str],
    ) -> List[str]:
        """Return deterministic, scope-aware options for a broad variable ask.

        Catalogue similarity alone tends to rank whichever variable has the
        shortest matching path.  For a broad electricity request that can put
        installed capacity (or a sector-specific price) ahead of electricity
        output.  Likewise, natural-language ``oil demand`` maps to final liquid
        energy in IAM taxonomies, not to oil capacity, price, or primary-energy
        supply. Use runtime taxonomy families, restricted to the requested
        scope when possible. No region names are encoded here: the same logic
        applies to every runtime geography.
        """
        q_lower = str(query or "").casefold()
        electricity_request = _infer_variable_intent(query) == "electricity"
        oil_demand_request = bool(
            re.search(r"\boil\b", q_lower)
            and re.search(r"\b(?:demand|consumption|consume|consumed|use|usage)\b", q_lower)
        )
        if not electricity_request and not oil_demand_request:
            return list(candidates)

        available = {
            str(value).strip()
            for value in (
                getattr(self.entity_extractor, "available_variables", []) or []
            )
            if str(value or "").strip()
        }
        if not available:
            return list(candidates)

        requested_regions = list(entities.get("regions") or [])
        if not requested_regions and entities.get("region"):
            requested_regions = [entities.get("region")]
        requested_region_keys = {
            str(value).strip().casefold()
            for value in requested_regions
            if str(value or "").strip()
        }
        if requested_region_keys:
            scoped_available = {
                str(record.get("variable") or "").strip()
                for record in (
                    getattr(self.entity_extractor, "ts_data", []) or []
                )
                if isinstance(record, dict)
                and str(record.get("region") or "").strip().casefold()
                in requested_region_keys
                and str(record.get("variable") or "").strip()
            }
            if scoped_available:
                available = scoped_available

        if oil_demand_request:
            sector_order = {
                "transport": 0,
                "transportation": 0,
                "industry": 1,
                "industrial": 1,
                "buildings": 2,
                "residential": 3,
                "commercial": 4,
            }

            def liquid_demand_key(variable: str) -> tuple[int, int, int, str]:
                parts = [part.strip().casefold() for part in str(variable).split("|")]
                if parts == ["final energy", "liquids"]:
                    return (0, 0, len(parts), variable.casefold())
                sectors = [sector_order[part] for part in parts if part in sector_order]
                if "fossil" in parts and not sectors:
                    return (1, 0, len(parts), variable.casefold())
                if sectors:
                    return (2, min(sectors), len(parts), variable.casefold())
                # Other liquid-carrier splits (for example bioenergy) remain
                # useful clarification choices, but follow oil/fossil and
                # sector-specific demand paths.
                return (3, 0, len(parts), variable.casefold())

            liquid_demand = []
            for variable in available:
                parts = [part.strip().casefold() for part in str(variable).split("|")]
                if (
                    parts
                    and parts[0] == "final energy"
                    and any(part in {"liquid", "liquids", "oil"} for part in parts[1:])
                ):
                    liquid_demand.append(str(variable))
            if liquid_demand:
                return sorted(set(liquid_demand), key=liquid_demand_key)[:3]
            return list(candidates)

        broad = _broad_electricity_candidates(available)
        if not broad:
            return list(candidates)
        ordered: List[str] = []
        for value in [*broad, *candidates]:
            text = str(value or "").strip()
            if text and text not in ordered:
                ordered.append(text)
        return ordered[:3]

    @staticmethod
    def _is_catalogue_year_request(query: str) -> bool:
        """Whether the user asks for dataset-wide temporal coverage."""
        text = str(query or "").casefold()
        return bool(
            re.search(r"\b(?:latest|last|maximum|max|earliest|first)\s+(?:available\s+)?year\b", text)
            and re.search(
                r"\b(?:projection|projections|dataset|database|catalog(?:ue)?|data|available)\b",
                text,
            )
        )

    def _is_qualitative_named_model_request(
        self,
        query: str,
        entities: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Separate model description questions from bounded data checks."""
        text = str(query or "")
        if not _looks_like_model_info_request(text):
            return False
        if not (self._mentions_known_model(text) or (entities or {}).get("model")):
            return False
        if _looks_like_plot_request(text):
            return False
        if any(
            _looks_like_category_list_request(text, category)
            for category in ("models", "variables", "regions", "scenarios")
        ):
            return False
        # These verbs/nouns ask whether the named model has a concrete data
        # slice.  Keep them on data_query even though the sentence also says
        # "model" (e.g. "does WITCH report carbon price for EU?").
        if re.search(
            r"\b(?:report|reports|reported|reporting|data|dataset|values?|"
            r"timeseries|time\s+series|available|availability|provide|provides|"
            r"show|display|plot|graph|chart|retrieve|fetch)\b",
            text,
            re.IGNORECASE,
        ):
            return False
        return True

    def _low_confidence_entity_prompt(
        self,
        entities: Optional[Dict[str, Any]],
        query: str = "",
    ) -> str:
        entities = entities or {}
        # An explicitly unknown place is a region-resolution error, not
        # evidence that a separately resolved variable is ambiguous. Let the
        # data layer explain the invalid region and offer valid alternatives.
        if str(entities.get("unmatched_region") or "").strip():
            return ""
        if self._is_catalogue_year_request(query):
            return ""
        if self._is_qualitative_named_model_request(query, entities):
            return ""
        # A bare ``emissions`` query resolves safely to CO2, but still lacks a
        # geography.  Rendering the unscoped records would mix countries and
        # aggregates in a table whose row identity contains no region column.
        # Keep the resolved variable and ask for the missing dimension before
        # the data agent can produce that misleading table.
        bare_emissions = bool(re.fullmatch(
            r"\s*(?:emission|emissions|emisions|emisson)\s*[?.!]*\s*",
            str(query or ""),
            flags=re.IGNORECASE,
        ))
        if (
            bare_emissions
            and str(entities.get("variable") or "").strip() == "Emissions|CO2"
            and not entities.get("region")
            and not entities.get("regions")
        ):
            available_regions = [
                str(value).strip()
                for value in (
                    getattr(self.entity_extractor, "available_regions", []) or []
                )
                if str(value).strip()
            ]
            by_key = {value.casefold(): value for value in available_regions}
            preferred_regions: List[str] = []
            for preferred in ("World", "EU", "CHN", "IND", "USA"):
                match = by_key.get(preferred.casefold())
                if match and match not in preferred_regions:
                    preferred_regions.append(match)
            preferred_regions.extend(
                value for value in available_regions
                if value not in preferred_regions
            )
            return _choice_prompt(
                "Interpretation: I treated bare **emissions** as `Emissions|CO2` "
                "(carbon-dioxide emissions). I need a concrete region before I can "
                "return a grounded result.",
                "region",
                preferred_regions[:3],
            )
        confidence = entities.get("entity_confidence") or {}
        variable_candidates = [
            str(value).strip()
            for value in (entities.get("variable_candidates") or [])
            if str(value).strip()
        ]
        variable_candidates = self._ordered_variable_clarification_candidates(
            query,
            entities,
            variable_candidates,
        )
        if not entities.get("variable") and variable_candidates:
            unmatched = [
                str(value).strip()
                for value in (entities.get("unmatched_variable_terms") or [])
                if str(value).strip()
            ]
            if unmatched:
                rendered = ", ".join(f"`{value}`" for value in unmatched)
                prefix = (
                    "I could not confidently match all of the requested wording. "
                    f"The loaded variable catalogue does not represent these term(s) in the top match: {rendered}."
                )
            else:
                prefix = "I found several possible variables, but none has enough evidence for automatic selection."
            return _choice_prompt(prefix, "variable", variable_candidates[:3])
        labels = {
            "variable": "variable",
            "region": "region",
            "scenario": "scenario",
            "model": "model",
        }
        for field, label in labels.items():
            value = entities.get(field)
            score = confidence.get(field)
            if value and isinstance(score, (int, float)) and score < 0.5:
                return (
                    f"I matched `{value}` as the {label}, but confidence is low. "
                    f"Which {label} should I use?"
                )
        return ""

    def _record_route_decision(
        self,
        agent_name: str,
        confidence: float,
        source: str,
        reason: str,
    ) -> str:
        self.last_route_decision = {
            "agent": agent_name,
            "confidence": round(float(confidence), 3),
            "source": source,
            "reason": reason,
        }
        self.logger.info(
            "Route decision: agent=%s confidence=%.2f source=%s reason=%s",
            agent_name,
            confidence,
            source,
            reason,
        )
        return agent_name

    def _mentions_known_model(self, query: str) -> bool:
        q = (query or "").strip().lower()
        if find_model_profile(q):
            return True
        model_names = [
            str(m.get("modelName", "")).lower()
            for m in self.shared_resources.get("models", [])
            if m and m.get("modelName")
        ]
        if any(
            re.search(r"(?<!\w)" + re.escape(name) + r"(?!\w)", q)
            for name in model_names
            if name
        ):
            return True
        return any(
            re.search(
                r"(?<!\w)" + re.escape(str(item.get("search_hint") or item.get("title") or "").casefold()) + r"(?!\w)",
                q,
            )
            for item in (self.shared_resources.get("link_catalog", []) or [])
            if isinstance(item, dict)
            and str(item.get("category") or "").casefold() == "models"
            and str(item.get("item_type") or "").casefold() == "model"
            and str(item.get("search_hint") or item.get("title") or "").strip()
        )

    @staticmethod
    def _extract_model_like_subject(query: str) -> str:
        """Extract a code/version-shaped subject from a definition question.

        This is a syntax detector, not a list of model names. The returned
        subject still has to resolve against the runtime model catalogue.
        """
        text = str(query or "").strip()
        match = re.match(
            r"^(?:what\s+is|describe|explain|tell\s+me\s+about)\s+(?:the\s+)?"
            r"(.+?)(?=,|\s+and\s+(?:what|which|how|where|why)\b|[?!.]?\s*$)",
            text,
            flags=re.IGNORECASE,
        )
        if not match:
            return ""
        subject = match.group(1).strip().rstrip("?!. ")
        acronym = bool(re.search(r"\b[A-Z][A-Z0-9_-]{2,}\b", subject))
        version = bool(re.search(r"\b(?:v(?:ersion)?\s*)?\d+(?:\.\d+)+\b", subject, re.IGNORECASE))
        labelled_model = bool(re.search(r"\bmodel\b", subject, re.IGNORECASE))
        return subject if acronym and (version or labelled_model) else ""

    def _unqualified_model_family_candidates(self, query: str) -> Tuple[str, List[str]]:
        """Resolve a bare family definition request without silently picking a variant."""
        subject = self._extract_model_like_subject(query)
        if not subject:
            return "", []
        subject = re.sub(r"\s+models?$", "", subject, flags=re.IGNORECASE).strip()
        subject_norm = normalize_model_name(subject)
        runtime_names = [
            str(value).strip()
            for value in (getattr(self.entity_extractor, "available_models", []) or [])
            if str(value or "").strip()
        ]
        if not subject_norm or any(
            normalize_model_name(name) == subject_norm for name in runtime_names
        ):
            return "", []
        candidates = [
            str(value).strip()
            for value in resolve_model_candidates(subject, runtime_names)
            if normalize_model_name(value).startswith(subject_norm)
        ]
        candidates = list(dict.fromkeys(value for value in candidates if value))
        return (subject, candidates) if candidates else ("", [])

    def _deterministic_route_decision(
        self,
        query: str,
        entities: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Deterministic route order:
        plot, data query, model info, availability/discovery, study/link suggestion, general QA.
        Active clarification is handled before this helper in _route_single.
        """
        q = (query or "").strip().lower()
        entities = entities or {}

        explicit_plot_query = _looks_like_plot_request(query)
        explicit_data_query = _looks_like_data_request(query)

        if explicit_plot_query or entities.get("action") == "plot":
            return {
                "agent": "data_plotting",
                "confidence": 0.95 if explicit_plot_query else 0.85,
                "source": "deterministic",
                "reason": "plot request",
            }

        scenario_only_comparison_followup = bool(
            re.match(
                r"compare\s+(?:with|to|against)\s+(?:baseline|policy|current policies?|scenario|the scenario)",
                q,
            )
            or re.match(
                r"compare\s+.+\s+versus\s+(?:baseline|policy|current policies?|scenario|the scenario)\b",
                q,
            )
        )
        if not scenario_only_comparison_followup and _looks_like_comparison_request(query) and (
            entities.get("variable")
            or entities.get("variables")
            or entities.get("model")
            or entities.get("models")
            or any(term in q for term in ("solar", "wind", "co2", "emission", "emissions", "gcam", "message", "remind", "witch"))
        ):
            return {
                "agent": "data_plotting",
                "confidence": 0.9,
                "source": "deterministic",
                "reason": "comparison plot request",
            }

        if self._is_site_navigation_request(query):
            return {
                "agent": "general_qa",
                "confidence": 0.88,
                "source": "deterministic",
                "reason": "site/navigation link request",
            }

        if self._is_catalogue_year_request(query):
            return {
                "agent": "data_query",
                "confidence": 0.97,
                "source": "deterministic",
                "reason": "catalogue year coverage request",
            }

        asks_model_expl = _looks_like_model_info_request(query)
        explicit_what_is = bool(re.search(r"\bwhat\s+is\b", q) or re.search(r"\bwho\s+is\b", q))
        mentions_model = self._mentions_known_model(query) or bool(entities.get("model"))
        vague_model_info = bool(
            asks_model_expl
            and mentions_model
            and "model" not in q
            and not explicit_what_is
            and re.search(r"\b(info|information)\b", q)
        )
        if vague_model_info:
            return {
                "agent": "data_query",
                "confidence": 0.82,
                "source": "deterministic",
                "reason": "vague model information request",
            }
        if (asks_model_expl and ("model" in q or mentions_model)) or (explicit_what_is and mentions_model):
            return {
                "agent": "model_explanation",
                "confidence": 0.9,
                "source": "deterministic",
                "reason": "model information request",
            }

        if _looks_like_capability_question(query):
            return {
                "agent": "general_qa",
                "confidence": 0.9,
                "source": "deterministic",
                "reason": "assistant capability question",
                "clear_entities": True,
            }

        if explicit_data_query:
            return {
                "agent": "data_query",
                "confidence": 0.9,
                "source": "deterministic",
                "reason": "data request",
            }

        if any(
            _looks_like_category_list_request(query, category)
            for category in ("models", "variables", "regions", "scenarios")
        ) or re.search(r"\b(list|available|what)\b.*\bworkspaces?\b", q):
            return {
                "agent": "data_query",
                "confidence": 0.9,
                "source": "deterministic",
                "reason": "availability/discovery request",
            }

        if any(token in q for token in ("suggest", "research idea", "investigate", "study suggestion")):
            return {
                "agent": "modelling_suggestions",
                "confidence": 0.82,
                "source": "deterministic",
                "reason": "study suggestion request",
            }

        if any(entities.get(k) for k in ("variable", "region", "scenario", "model")):
            return {
                "agent": "data_query",
                "confidence": 0.75,
                "source": "deterministic",
                "reason": "extracted data entities",
            }

        if any(token in q for token in ("climate", "policy", "paris agreement", "decarbon", "mitigation")):
            return {
                "agent": "general_qa",
                "confidence": 0.7,
                "source": "deterministic",
                "reason": "general climate/policy question",
            }

        return None

    def _route_with_llm_fallback(self, query: str, entities: Optional[Dict[str, Any]]) -> str:
        try:
            result = self.routing_prompt | self.router_llm
            response_obj = result.invoke({"query": query})
            agent_name = str(response_obj.content or "").strip().lower()
            if agent_name not in VALID_AGENT_NAMES:
                fallback = self._classify_route_heuristic(query, entities)
                return self._record_route_decision(
                    fallback,
                    0.55,
                    "heuristic",
                    f"invalid LLM route `{agent_name}`",
                )
            return self._record_route_decision(agent_name, 0.6, "llm", "unclear deterministic route")
        except Exception as route_err:
            self.logger.warning(
                "Router LLM unavailable (%s). Falling back to heuristic routing.",
                route_err,
            )
            fallback = self._classify_route_heuristic(query, entities)
            return self._record_route_decision(fallback, 0.5, "heuristic", "router LLM unavailable")

    def _classify_route_heuristic(self, query: str, entities: Optional[Dict[str, Any]] = None) -> str:
        """
        Local, no-network route classifier used when router LLM is unavailable.
        """
        q = (query or "").strip().lower()
        entities = entities or {}

        if _looks_like_plot_request(query):
            return "data_plotting"
        if self._is_site_navigation_request(query):
            return "general_qa"
        if find_model_profile(q):
            mentions_profile_model = True
        else:
            mentions_profile_model = False
        model_names = [
            str(m.get("modelName", "")).lower()
            for m in self.shared_resources.get("models", [])
            if m and m.get("modelName")
        ]
        mentions_model = any(
            re.search(r"(?<!\w)" + re.escape(name) + r"(?!\w)", q)
            for name in model_names[:200]
            if name
        ) or mentions_profile_model
        asks_model_expl = _looks_like_model_info_request(query)
        explicit_what_is = bool(re.search(r"\bwhat\s+is\b", q) or re.search(r"\bwho\s+is\b", q))
        if (asks_model_expl and ("model" in q or mentions_model)) or (explicit_what_is and mentions_model):
            return "model_explanation"
        if any(
            _looks_like_category_list_request(query, category)
            for category in ("models", "variables", "regions", "scenarios")
        ) or re.search(r"\b(list|available|what)\b.*\bworkspaces?\b", q):
            return "data_query"
        if _looks_like_capability_question(query):
            return "general_qa"
        if any(entities.get(k) for k in ("variable", "region", "scenario", "model")):
            return "data_query"
        if _looks_like_data_request(query):
            return "data_query"
        return "general_qa"

    def _is_provider_error(self, err: Exception) -> bool:
        msg = str(err or "").lower()
        return any(
            token in msg
            for token in (
                "provider error",
                "api key",
                "insufficient_quota",
                "rate limit",
                "authentication",
                "connection error",
                "timeout",
                "openai",
                "401",
                "403",
                "429",
                "5xx",
            )
        )

    def _is_intentful_segment(self, segment: str) -> bool:
        """Heuristic check for whether a segment contains a recognizable intent."""
        s = segment.lower()
        intent_markers = [
            "list", "show", "plot", "graph", "chart", "visualize", "compare", "vs", "versus",
            "tell me about", "explain", "describe", "what models", "what variables",
            "what scenarios", "available models", "available variables", "available scenarios",
            "suggest", "research", "investigate"
        ]
        return any(m in s for m in intent_markers)

    def _split_multi_intent(self, query: str) -> List[str]:
        """
        Split multi-intent queries into sub-queries using conservative heuristics.
        """
        q = query.strip()
        lower = q.lower()
        # Split only where a new intent verb begins, so "show solar and wind
        # capacity and plot it" keeps "solar and wind" together.
        intent_verb = r"(?:list|show|display|plot|graph|chart|visualize|visualise|compare|tell\s+me\s+about|explain|describe|what|available)"
        if " and plot " in lower or lower.endswith(" and plot it") or " and plot it" in lower:
            parts = re.split(
                r"\s+and\s+(?=(?:plot|graph|chart|visualize|visualise)\b)",
                q,
                flags=re.IGNORECASE,
            )
            parts = [p.strip() for p in parts if p and p.strip()]
            return parts if len(parts) > 1 else [q]

        intent_markers = [
            "list", "show", "plot", "graph", "chart", "visualize", "compare",
            "tell me about", "explain", "describe", "what models", "what variables",
            "what scenarios", "available models", "available variables", "available scenarios"
        ]
        intent_hits = sum(1 for m in intent_markers if m in lower)
        if intent_hits < 2:
            return [q]

        parts = re.split(
            r";|\n|\s+(?:and|then|also)\s+(?=" + intent_verb + r"\b)",
            q,
            flags=re.IGNORECASE,
        )
        parts = [p.strip() for p in parts if p and p.strip()]

        # If split produced segments without intent, merge them back to previous
        merged: List[str] = []
        for part in parts:
            if not merged:
                merged.append(part)
                continue
            # A metadata clause such as "describe its limitations" is usually
            # a continuation of the named-model explanation immediately before
            # it, not a second output request. Resolve that grammatical
            # dependency generically while leaving independent data/plot/list
            # intents as separate segments.
            dependent_model_metadata = bool(
                self._runtime_model_profiles(merged[-1])
                and not find_model_profiles(part)
                and _looks_like_model_info_request(merged[-1])
                and _looks_like_model_info_request(part)
                and re.search(
                    r"\b(?:it|its|their|this\s+model|that\s+model|the\s+model)\b",
                    part,
                    flags=re.IGNORECASE,
                )
            )
            if dependent_model_metadata:
                merged[-1] = f"{merged[-1]} and {part}".strip()
                continue
            if self._is_intentful_segment(part):
                merged.append(part)
            else:
                merged[-1] = f"{merged[-1]} {part}".strip()

        return merged if len(merged) > 1 else [q]

    def _scenario_comparison_followup_values(
        self,
        query: str,
        carried: Optional[Dict[str, Any]],
    ) -> List[str]:
        """Resolve a scenario pair from a comparison follow-up.

        The current value comes from successful conversation state and the new
        value must be grounded in the live scenario catalogue.  Plural carried
        scopes are intentionally not extended: without a singular left-hand
        side, adding another scenario would be ambiguous and could silently
        widen an existing comparison.
        """
        if not carried or not _looks_like_comparison_request(query):
            return []
        if not re.search(
            r"\b(?:with|to|against|versus|vs\.?)\b",
            str(query or ""),
            flags=re.IGNORECASE,
        ):
            return []

        # Prefer a singular canonical scenario (e.g. "Current Policies") as the
        # left-hand side: the plural `scenarios` is usually just that family's
        # expansion into member codes, and treating it as a multi-scenario set
        # would wrongly block the comparison and let the new scenario be misread
        # as a region ("net zero" -> "RO").
        singular = str(carried.get("scenario") or "").strip()
        if singular:
            current_values = [singular]
        else:
            plural = carried.get("scenarios")
            if plural:
                current_values = (
                    list(plural)
                    if isinstance(plural, (list, tuple, set))
                    else [plural]
                )
            else:
                current_values = []
        current_values = [
            str(value).strip() for value in current_values if str(value or "").strip()
        ]
        if len(current_values) != 1:
            return []

        available = getattr(self.entity_extractor, "available_scenarios", []) or []
        # Prefer the canonical family label so both sides expand consistently
        # (family vs family), falling back to a specific matched code.
        target = (
            self._match_catalog_value_from_text(query, available)
            or canonical_scenario_family_from_query(query, available)
            or canonical_scenario_from_query(query, available)
            or self._match_scenario_from_text(query)
        )
        target = str(target or "").strip()
        current = current_values[0]
        if not target or target.casefold() == current.casefold():
            return []
        return [current, target]

    def _model_switch_names(self) -> List[str]:
        """Model names accepted for a "from <model>" scope switch.

        The live model catalogue only lists models with timeseries, so models
        that are referenced but carry no data (e.g. REMIND, WITCH) would be
        unrecognised. Augment it with the native-region model families declared
        under definitions/region/native_regions so such a switch is still
        understood and answered honestly as no-data rather than silently reusing
        the previous model. An energy source like "solar" matches nothing here
        and is left for normal source/variable refinement."""
        cached = getattr(self, "_model_switch_names_cache", None)
        if cached is not None:
            return cached
        names = set(getattr(self.entity_extractor, "available_models", []) or [])
        try:
            native_dir = Path("definitions/region/native_regions")
            for entry in native_dir.iterdir():
                family = re.sub(r"\.ya?ml$", "", entry.name, flags=re.IGNORECASE).strip()
                if family:
                    names.add(family)
        except Exception:
            pass
        result = sorted(name for name in names if name)
        self._model_switch_names_cache = result
        return result

    def _compose_contextual_query(self, query: str, carried: Optional[Dict[str, Any]]) -> str:
        """
        Enrich follow-up queries like "plot it" or "show me data" with the last
        resolved variable, region, scenario, or model when available.
        """
        if not carried:
            return query

        ql = query.lower()
        variable = str(carried.get("variable", "") or "").strip()
        region = str(carried.get("region", "") or "").strip()
        scenario = str(carried.get("scenario", "") or "").strip()
        model = str(carried.get("model", "") or "").strip()
        start_year = carried.get("start_year")
        end_year = carried.get("end_year")
        all_scenarios = bool(carried.get("all_scenarios"))
        carried_scenarios = [
            str(value).strip()
            for value in (carried.get("scenarios") or [])
            if str(value or "").strip()
        ]

        if self._is_contextual_dimension_followup(query):
            # Work on the filler-stripped, original-case query so scenario codes
            # keep their casing and leading "now/and/..." is ignored.
            _stripped = re.sub(self._FOLLOWUP_FILLER, "", query.strip(), flags=re.IGNORECASE).strip().rstrip("?.!").strip()
            same_for = re.search(
                r"(?i)\b(?:(?:show|display|give)(?:\s+me)?\s+)?"
                r"(?:do\s+)?(?:the\s+)?same(?:\s+[a-z0-9_-]+){0,2}\s+for\s+(.+)$",
                _stripped,
            )
            what_about = re.search(r"(?i)\b(?:what|how)\s+about\s+(.+)$", _stripped)
            compare_with = re.search(r"(?i)\bcompare\s+(?:with|to|against)\s+(.+)$", _stripped)
            compare_year = re.fullmatch(
                r"(?i)compare\s+(?:(?:that|it|this)\s+)?(?:with|to|against)\s+(\d{4})",
                _stripped,
            )
            scope_year = re.match(
                r"(?i)(?:show\s+|plot\s+)?"
                r"((?:after|before|by|until|up\s+to|through|in|from|since|only|just)"
                r"\s+\d{4}(?:\s*(?:to|until|-|and)\s*\d{4})?)\s*$",
                _stripped,
            )
            under_switch = re.fullmatch(r"(?i)(?:same\s+)?under\s+(.+)", _stripped)
            for_switch = re.fullmatch(r"(?i)(?:same\s+)?for\s+(.+)", _stripped)
            from_switch = re.fullmatch(r"(?i)(?:the\s+)?(?:same\s+)?from\s+(.+)", _stripped)
            possessive_switch = re.fullmatch(r"(?i)(?:its|their)\s+(.+)", _stripped)
            plot_switch = re.fullmatch(r"(?i)(?:plot|chart|graph|visuali[sz]e|draw)\s+(.+)", _stripped)

            # "same from REMIND" / "now from GCAM": override only the dimension
            # the trailing token names (the model, per the "from <model>" idiom),
            # keep the rest, and drop the stale carried model. Resolve the model
            # explicitly first: the generic dimension resolver tries region
            # before model and can misclassify a model token as a region.
            if from_switch:
                _target = from_switch.group(1).strip()
                # "from <X>" is a model switch only when X actually names a model.
                # It is resolved against the model catalogue plus known model
                # families (e.g. REMIND, WITCH), which may lack timeseries — those
                # still switch, yielding an honest no-data answer instead of
                # silently reusing the previous model. When X is not a model
                # (e.g. "from solar", an energy source), fall through so the rest
                # of the pipeline refines the source/variable as before. The
                # region resolver is deliberately bypassed here: model family
                # names also appear as native-region group labels.
                _new_model = ""
                try:
                    from model_aliases import match_model_name as _mmn
                    _new_model = _mmn(_target, self._model_switch_names())
                except Exception:
                    _new_model = ""
                if _new_model:
                    parts = ["show"]
                    if variable:
                        parts.append(variable)
                    if region:
                        parts.append(f"for {region}")
                    if scenario:
                        parts.append(f"under {scenario}")
                    parts.append(f"from {_new_model}")
                    if len(parts) > 1:
                        return " ".join(parts)

            # "and its GDP" / "its population": a new variable on the carried
            # region/scenario/model. If the trailing token actually names a
            # region/scenario/model, treat it as that dimension switch instead.
            if possessive_switch:
                target = possessive_switch.group(1).strip()
                _dim = self._resolve_carry_dimension(target)
                if _dim:
                    _kind, _val = _dim
                    parts = ["show"]
                    if variable:
                        parts.append(variable)
                    _r = _val if _kind == "region" else region
                    _s = _val if _kind == "scenario" else scenario
                    _m = _val if _kind == "model" else model
                    if _r:
                        parts.append(f"for {_r}")
                    if _s:
                        parts.append(f"under {_s}")
                    if _m:
                        parts.append(f"from {_m}")
                    if len(parts) > 1:
                        return " ".join(parts)
                else:
                    parts = ["show", target]
                    if region:
                        parts.append(f"for {region}")
                    if scenario:
                        parts.append(f"under {scenario}")
                    if model:
                        parts.append(f"from {model}")
                    if len(parts) > 1:
                        return " ".join(parts)

            if plot_switch:
                _target = plot_switch.group(1).strip()
                _dim = self._resolve_carry_dimension(_target)
                if _dim:
                    _kind, _val = _dim
                    parts = ["plot"]
                    if variable:
                        parts.append(variable)
                    _r = _val if _kind == "region" else region
                    _s = _val if _kind == "scenario" else scenario
                    _m = _val if _kind == "model" else model
                    if _r:
                        parts.append(f"for {_r}")
                    if _s:
                        parts.append(f"under {_s}")
                    if _m:
                        parts.append(f"for {_m}")
                    if len(parts) > 1:
                        return " ".join(parts)
                # "plot its CO2 emissions" names a new variable on the carried
                # scope. The possessive handler below only matches when the
                # query *starts* with "its", so a leading plot verb skipped it
                # and the carried model/region were dropped -- the chart then
                # spanned every model in the catalogue.
                _possessive_target = re.fullmatch(r"(?i)(?:its|their)\s+(.+)", _target)
                if _possessive_target:
                    parts = ["plot", _possessive_target.group(1).strip()]
                    if region:
                        parts.append(f"for {region}")
                    if scenario:
                        parts.append(f"under {scenario}")
                    if model:
                        parts.append(f"from {model}")
                    if len(parts) > 1:
                        return " ".join(parts)

            if "show all scenarios" in ql:
                parts = ["show all scenarios"]
                if variable:
                    parts.append(f"for {variable}")
                if region:
                    parts.append(f"in {region}")
                if model:
                    parts.append(f"for {model}")
                return " ".join(parts)

            if compare_year:
                target_year = int(compare_year.group(1))
                years = [int(y) for y in (start_year, end_year, target_year) if y is not None]
                parts = ["show"]
                if variable:
                    parts.append(variable)
                if region:
                    parts.append(f"for {region}")
                if scenario:
                    parts.append(f"under {scenario}")
                if years:
                    parts.append(f"from {min(years)} to {max(years)}")
                if model:
                    parts.append(f"for {model}")
                return " ".join(parts)

            if compare_with:
                target = compare_with.group(1).strip()
                parts = ["plot compare"]
                if variable:
                    parts.append(variable)
                if region:
                    parts.append(f"for {region}")
                if scenario:
                    parts.append(f"under {scenario}")
                if target:
                    parts.append(f"versus {target}")
                if model:
                    parts.append(f"for {model}")
                if len(parts) > 1:
                    return " ".join(parts)

            if scope_year:
                replacement = scope_year.group(1).strip()
                parts = ["show"]
                if variable:
                    parts.append(variable)
                if region:
                    parts.append(f"for {region}")
                if scenario:
                    parts.append(f"under {scenario}")
                parts.append(replacement)
                if model:
                    parts.append(f"for {model}")
                if len(parts) > 1:
                    return " ".join(parts)

            if under_switch:
                new_scenario = under_switch.group(1).strip()
                parts = ["show"]
                if variable:
                    parts.append(variable)
                if region:
                    parts.append(f"for {region}")
                parts.append(f"under {new_scenario}")
                if model:
                    parts.append(f"for {model}")
                if len(parts) > 1:
                    return " ".join(parts)

            replacement = ""
            if same_for:
                replacement = same_for.group(1).strip()
            elif what_about:
                replacement = what_about.group(1).strip()
            elif for_switch:
                replacement = for_switch.group(1).strip()

            if replacement:
                start_year, end_year = extract_year_range(replacement)
                scenario_replacement = self._match_scenario_from_text(replacement)
                # "same for wind" after solar electricity switches the carrier of
                # the carried variable, keeping region and scenario.
                carrier_variable = self._carrier_switch_variable(variable, replacement)
                if carrier_variable:
                    parts = ["show", carrier_variable]
                    if region:
                        parts.append(f"for {region}")
                    if scenario:
                        parts.append(f"under {scenario}")
                    if model:
                        parts.append(f"from {model}")
                    return " ".join(parts)
                parts = ["show"]
                if variable:
                    parts.append(variable)
                if start_year is not None or end_year is not None:
                    if region:
                        parts.append(f"for {region}")
                    if scenario:
                        parts.append(f"under {scenario}")
                    parts.append(replacement)
                elif scenario_replacement:
                    if region:
                        parts.append(f"for {region}")
                    parts.append(f"under {scenario_replacement}")
                else:
                    parts.append(f"for {replacement}")
                    if scenario:
                        parts.append(f"under {scenario}")
                if model:
                    parts.append(f"for {model}")
                if len(parts) > 1:
                    return " ".join(parts)

        if self._is_generic_followup(query):
            if any(token in ql for token in ("plot", "graph", "chart")):
                lead = "plot"
            else:
                lead = "show"

            parts: List[str] = [lead]
            if variable:
                parts.append(variable)
            if region:
                parts.append(f"for {region}")
            # "plot the comparison" must keep both sides of the comparison the
            # previous turn established. Only the singular `scenario` was read
            # here, so the follow-up silently collapsed to one scenario.
            comparison_followup = bool(
                re.search(r"\b(?:comparison|compare|both|them|two)\b", ql)
                and (carried.get("comparison") or len(carried_scenarios) > 1)
            )
            if comparison_followup and len(carried_scenarios) > 1:
                pair = carried_scenarios[:2]
                parts.append(f"under {pair[0]} and {pair[1]}")
            elif scenario:
                parts.append(f"under {scenario}")
            elif all_scenarios:
                parts.append("across available scenarios")
            if start_year is not None or end_year is not None:
                parts.append(YearFilter(
                    start_year,
                    end_year,
                    explicit=True,
                ).render())
            if model:
                parts.append(f"for {model}")
            if len(parts) > 1:
                return " ".join(parts)

        additions: List[str] = []

        def _append_if_missing(key: str, label: str) -> None:
            value = carried.get(key)
            if not value:
                return
            value_str = str(value).strip()
            if not value_str:
                return
            value_lower = value_str.lower()
            if re.search(r"\b" + re.escape(value_lower) + r"\b", ql):
                return
            additions.append(f"{label} {value_str}")

        _append_if_missing("variable", "variable")
        _append_if_missing("region", "region")
        _append_if_missing("scenario", "scenario")
        _append_if_missing("model", "model")

        if not additions:
            return query

        return f"{query} " + " ".join(additions)

    def _persist_last_entities(
        self,
        entities: Optional[Dict[str, Any]] = None,
        response: str = "",
    ) -> None:
        text = str(response or "")
        prior = dict(self.last_entities or {})
        unsuccessful = self._is_unsuccessful_response(text)

        # Keep attempted scope separately. A failed/no-data/clarification turn
        # must not become the source of pronouns such as "it" or mutations such
        # as "plot the same data". The clarification flow already retains the
        # attempted entities in clarification_context.
        if unsuccessful:
            attempted_entities = dict(entities or {})
            if MultiAgentManager._looks_like_clarification_response(self, text):
                attempted_entities = MultiAgentManager._finalize_clarification_entities(
                    attempted_entities,
                )
            unmatched_region = str(
                attempted_entities.get("unmatched_region") or ""
            ).strip()
            if unmatched_region:
                # An explicitly rejected place supersedes any successful
                # geography carried from an earlier turn, but remains an
                # attempted (not active) scope for conversational follow-ups.
                attempted_entities.pop("region", None)
                attempted_entities.pop("regions", None)
            state = getattr(self, "__dict__", {}).get("conversation_state")
            if isinstance(state, ConversationState):
                state.record_attempt(attempted_entities)
                if unmatched_region:
                    # Expose this turn's failed scope to the API while keeping
                    # ``active_scope`` intact for a later "plot it" follow-up.
                    state.response_scope_override = dict(attempted_entities)
            else:
                self.last_attempted_entities = attempted_entities
            consume_resolved_scope()
            return

        supplied_entities = dict(entities or {}) if entities is not None else {}
        merged = dict(supplied_entities) if entities is not None else prior
        requested_year_scope = bool(
            entities is not None
            and (
                "start_year" in supplied_entities
                or "end_year" in supplied_entities
            )
            and not YearFilter(
                supplied_entities.get("start_year"),
                supplied_entities.get("end_year"),
                explicit=True,
                operator="latest",
            ).is_latest
        )

        for key in (
            "variable", "variables", "region", "regions", "scenario", "scenarios",
            "model", "models", "unit", "start_year", "end_year", "action", "chart_type",
            "all_scenarios", "comparison", "unmatched_region", "workspace_code",
        ):
            value = (entities or {}).get(key)
            if value not in (None, "", [], {}):
                merged[key] = value

        # Preferred channel: the scope the answer formatter actually resolved,
        # reported structurally by data_utils/simple_plotter.
        structured_scope = consume_resolved_scope()
        if structured_scope:
            for key, value in structured_scope.items():
                if value in (None, "", [], {}):
                    continue
                if requested_year_scope and key in {"start_year", "end_year"}:
                    # Formatter scopes describe the data points that happened
                    # to be visible (for example 2035..2100).  They must not
                    # replace the user's open request (after 2030 =>
                    # 2031..unbounded), because the requested bounds are what
                    # subsequent follow-ups need to mutate.  Retain both facts
                    # under distinct names when they differ.
                    observed_key = f"observed_{key}"
                    if (
                        key not in supplied_entities
                        or supplied_entities.get(key) != value
                    ):
                        merged[observed_key] = value
                    else:
                        merged.pop(observed_key, None)
                    continue
                merged[key] = value
            # Aggregate comparison scope is authoritative. Do not retain a
            # singular extractor value when the answer rendered several
            # variables, regions, scenarios, or models.
            for singular, plural in (
                ("variable", "variables"), ("region", "regions"),
                ("scenario", "scenarios"), ("model", "models"),
            ):
                values = list(
                    structured_scope.get(plural)
                    or (structured_scope.get("result_models") if plural == "models" else [])
                    or []
                )
                if len(values) > 1:
                    merged.pop(singular, None)
                elif len(values) == 1:
                    merged[singular] = values[0]
        else:
            # Fallback: parse the rendered answer (legacy paths that do not
            # record their scope yet).
            first_line = text.splitlines()[0].strip() if text else ""

            header_match = re.match(r"^###\s+(.+?)\s+in\s+(.+?)\s*$", first_line)
            if header_match:
                merged["variable"] = header_match.group(1).strip()
                merged["region"] = header_match.group(2).strip()

            prompt_match = re.search(
                r"I found the variable\s+`([^`]+)`.*?\s+in\s+`([^`]+)`",
                text,
                re.IGNORECASE,
            )
            if prompt_match:
                merged["variable"] = prompt_match.group(1).strip()
                merged["region"] = prompt_match.group(2).strip()

            plot_match = re.search(
                r"Showing\s+.+?\s+in\s+(.+?)\s+for\s+scenario\s+`([^`]+)`",
                text,
                re.IGNORECASE,
            )
            if plot_match:
                merged["region"] = plot_match.group(1).strip()
                merged["scenario"] = plot_match.group(2).strip()

        # Keep singular and plural scope representations consistent. If this
        # turn explicitly changes one singular dimension, a plural selection
        # carried from the previous turn is stale unless the current turn also
        # supplies a replacement list.
        for singular, plural in (
            ("scenario", "scenarios"),
            ("model", "models"),
            ("variable", "variables"),
            ("region", "regions"),
        ):
            current_value = (entities or {}).get(singular)
            prior_value = prior.get(singular)
            current_plural = (entities or {}).get(plural) or structured_scope.get(plural)
            if (
                current_value not in (None, "")
                and str(current_value) != str(prior_value or "")
                and not current_plural
            ):
                merged.pop(plural, None)

        if merged:
            state = getattr(self, "__dict__", {}).get("conversation_state")
            if isinstance(state, ConversationState):
                state.record_success(merged)
            else:
                comparable_keys = ("variable", "region", "scenario", "model", "start_year", "end_year")
                if prior and any(prior.get(key) != merged.get(key) for key in comparable_keys):
                    self.previous_entities = prior
                self.last_entities = merged
                self.last_attempted_entities = {}

        # Remember which models produced the last data answer so a follow-up like
        # "which model is this from?" can be answered. Bold table headers read
        # "**<model> - <scenario>**"; a single-model answer states "model `X`".
        result_models: List[str] = []
        structured_models = (
            structured_scope.get("result_models")
            or structured_scope.get("models")
            or structured_scope.get("model")
        )
        if isinstance(structured_models, str):
            result_models = [display_model_label(structured_models)]
        elif structured_models:
            result_models = [
                display_model_label(name)
                for name in structured_models
                if str(name).strip()
            ]
        for name in re.findall(r"\*\*([^*]+?)\s+-\s+[^*]+\*\*", text):
            name = display_model_label(name.strip())
            if name and name not in result_models:
                result_models.append(name)
        if not result_models:
            single = re.search(r"model `([^`]+)`", text)
            if single and single.group(1) not in ("multiple", ""):
                result_models = [display_model_label(single.group(1))]
        if result_models:
            self.last_result_models = result_models

    _UNSUCCESSFUL_RESPONSE_MARKERS = (
        "i need one more detail",
        "please specify the variable",
        "i couldn't find",
        "i could not find",
        "couldn't match",
        "could not match",
        "i don't have an active",
        "which variable should i use",
        "which variable or region",
        "no data",
        "no time series data",
        "no timeseries data",
        "i can't combine",
        "i cannot combine",
        "incompatible units",
        "i can't plot",
        "i cannot plot",
        "could not identify enough",
        "could not identify variable",
        "could not identify a variable",
        "not found in loaded data",
        "sorry, the requested agent",
    )

    def _is_unsuccessful_response(self, text: str) -> bool:
        """Heuristic: did this turn fail to produce a real data/model answer?"""
        body = str(text or "").strip()
        if not body:
            return True
        lowered = body.lower()
        # A comparison can legitimately include a recovery notice such as
        # "no timeseries data for model X" while still returning a real plot
        # for the models that do have data.  The rendered plot is authoritative
        # evidence of success; failure-word heuristics only apply without one.
        if re.search(r"!\[plot\]\(", lowered):
            return False
        # A comparison may start with a missing-member notice while still
        # containing valid numeric tables for the available members.
        if has_numeric_result_table(body):
            return False
        return (
            any(marker in lowered for marker in self._UNSUCCESSFUL_RESPONSE_MARKERS)
            or MultiAgentManager._looks_like_clarification_response(self, body)
        )

    def _is_generic_followup(self, query: str) -> bool:
        ql = query.strip().lower()
        # Strip leading filler so "now plot it" / "ok show me that" are still
        # recognised as context-carrying follow-ups.
        ql = re.sub(r"^(?:(?:ok(?:ay)?|now|then|so|and|please|just|also|yeah|yes)\s+)+", "", ql).strip()
        ql = ql.rstrip("?.!").strip()
        if ql in {"continue", "keep going", "what about it"}:
            return True
        if re.fullmatch(
            r"(?:plot|graph|chart|show|display|draw|visuali[sz]e)\s+(?:me\s+)?"
            r"(?:the\s+)?(?:comparison|difference)"
            r"(?:\s+(?:between|of)\s+(?:the\s+)?(?:two|both|previous|last)\s+\w+)?",
            ql,
        ):
            return True
        return bool(
            re.fullmatch(
                r"(?:plot|graph|chart|show|display|draw|visuali[sz]e|give|use)\s+(?:me\s+)?"
                r"(?:it|this|that|those|them|the same)(?:\s+(?:one|result|results))?"
                r"(?:\s+(?:both|all|together|again|too|data))?"
                r"(?:\s+as\s+(?:a\s+)?(?:plot|graph|chart|visuali[sz]ation))?",
                ql,
            )
            or re.fullmatch(
                r"(?:make|draw|create|generate)\s+(?:me\s+)?an?\s+(?:plot|graph|chart)"
                r"(?:\s+(?:of|for|with)\s+(?:it|this|that|them|those))?",
                ql,
            )
            or re.fullmatch(
                r"(?:plot|graph|chart|show|display)\s+(?:me\s+)?the\s+(?:same\s+)?data(?:\s+again)?",
                ql,
            )
        )

    def _render_previous_scope_comparison(self, current: Dict[str, Any]) -> str:
        """Render a plot request when two successful scopes differ in one
        comparable dimension."""
        previous = dict(getattr(self, "previous_entities", {}) or {})
        if not previous or not current:
            return ""

        plural_keys = {"region": "regions", "scenario": "scenarios", "model": "models"}

        def _dimension_values(scope: Dict[str, Any], dimension: str) -> tuple:
            plural = scope.get(plural_keys[dimension])
            if plural:
                values = plural if isinstance(plural, (list, tuple, set)) else [plural]
            else:
                singular = scope.get(dimension)
                values = [] if singular in (None, "") else [singular]
            return tuple(sorted({str(value).strip().casefold() for value in values if str(value).strip()}))

        dimension_values = {
            dimension: (_dimension_values(previous, dimension), _dimension_values(current, dimension))
            for dimension in plural_keys
        }

        # When neither side pinned a scenario and both answers spanned all
        # available scenarios, the per-answer scenario lists are derived data
        # (whatever each slice happened to contain), not a user choice. Treat
        # that dimension as invariant so it cannot block a genuine one-
        # dimension comparison.
        def _scenario_is_noise() -> bool:
            return bool(
                previous.get("all_scenarios")
                and current.get("all_scenarios")
                and not str(previous.get("scenario") or "").strip()
                and not str(current.get("scenario") or "").strip()
            )

        changed = [
            dimension
            for dimension, (old_values, new_values) in dimension_values.items()
            if old_values and new_values and old_values != new_values
            and not (dimension == "scenario" and _scenario_is_noise())
        ]
        if len(changed) != 1:
            return ""
        dimension = changed[0]

        # A previous-v-current comparison is valid only when every other part
        # of the resolved scope is invariant. Missing scope on one side is not
        # evidence that the two answers are comparable.
        for other_dimension, (old_values, new_values) in dimension_values.items():
            if other_dimension == "scenario" and _scenario_is_noise():
                continue
            if other_dimension != dimension and old_values != new_values:
                return ""
        previous_variable = str(previous.get("variable") or "").strip()
        current_variable = str(current.get("variable") or "").strip()
        if (
            not previous_variable
            or not current_variable
            or previous_variable.casefold() != current_variable.casefold()
        ):
            return ""
        if any(previous.get(key) != current.get(key) for key in ("start_year", "end_year")):
            return ""

        old_values, new_values = dimension_values[dimension]
        if len(old_values) != 1 or len(new_values) != 1:
            return ""
        def _display_dimension(scope: Dict[str, Any], key: str) -> str:
            plural = scope.get(plural_keys[key])
            if plural:
                values = plural if isinstance(plural, (list, tuple, set)) else [plural]
                return str(next(iter(values), "")).strip()
            return str(scope.get(key) or "").strip()

        first = _display_dimension(previous, dimension)
        second = _display_dimension(current, dimension)
        variable = current_variable
        parts = ["plot", variable]
        if dimension == "region":
            parts.append(f"for {first} and {second}")
        else:
            region = str(current.get("region") or previous.get("region") or "").strip()
            if region:
                parts.append(f"for {region}")
            if dimension == "scenario":
                parts.append(f"under scenarios {first} and {second}")
            else:
                parts.append(f"for models {first} and {second}")
        if dimension != "scenario":
            scenario = str(current.get("scenario") or previous.get("scenario") or "").strip()
            if scenario:
                parts.append(f"under {scenario}")
        start, end = current.get("start_year"), current.get("end_year")
        if start is not None:
            parts.append(f"in {start}" if end in (None, start) else f"from {start} to {end}")
        return " ".join(parts)

    def _previous_scope_comparison_entities(self, current: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Build plot entities for "plot both ..." follow-ups from the last two
        successful scopes, when they differ in exactly one dimension (variable,
        region, or model). Passing structured entities straight to the plotting
        agent avoids re-extracting exact catalogue names (e.g. `GDP|MER`) from a
        rendered sentence, which is lossy for names with separators."""
        previous = dict(getattr(self, "previous_entities", {}) or {})
        current = dict(current or {})
        if not previous or not current:
            return None

        def _val(scope: Dict[str, Any], key: str) -> str:
            return str(scope.get(key) or "").strip()

        prev_var, cur_var = _val(previous, "variable"), _val(current, "variable")
        if not prev_var or not cur_var:
            return None
        if any(previous.get(key) != current.get(key) for key in ("start_year", "end_year")):
            return None
        prev_scenario, cur_scenario = _val(previous, "scenario"), _val(current, "scenario")
        if (prev_scenario or cur_scenario) and prev_scenario.casefold() != cur_scenario.casefold():
            # Scenario switches already have a dedicated comparison flow.
            return None
        prev_region, cur_region = _val(previous, "region"), _val(current, "region")
        prev_model, cur_model = _val(previous, "model"), _val(current, "model")

        diffs = [
            dimension
            for dimension, (old_value, new_value) in (
                ("variable", (prev_var, cur_var)),
                ("region", (prev_region, cur_region)),
                ("model", (prev_model, cur_model)),
            )
            if old_value.casefold() != new_value.casefold()
        ]
        if len(diffs) != 1:
            return None
        dimension = diffs[0]

        entities: Dict[str, Any] = {
            "action": "plot",
            "start_year": current.get("start_year"),
            "end_year": current.get("end_year"),
        }
        if cur_scenario:
            entities["scenario"] = cur_scenario
        else:
            entities["all_scenarios"] = True

        if dimension == "variable":
            entities["variables"] = [prev_var, cur_var]
            entities["variable"] = cur_var
            entities["comparison"] = "variable"
            if cur_region:
                entities["region"] = cur_region
            if cur_model:
                entities["model"] = cur_model
        elif dimension == "region":
            if not prev_region or not cur_region:
                return None
            entities["variable"] = cur_var
            entities["regions"] = [prev_region, cur_region]
            entities["comparison"] = "region"
            if cur_model:
                entities["model"] = cur_model
        else:
            if not prev_model or not cur_model:
                return None
            entities["variable"] = cur_var
            entities["models"] = [prev_model, cur_model]
            entities["comparison"] = "model"
            if cur_region:
                entities["region"] = cur_region
        return entities

    _FOLLOWUP_FILLER = r"^(?:(?:ok(?:ay)?|now|then|so|and|also|please)\s+)+"

    def _from_switch_model(self, stripped: str) -> str:
        """If `stripped` is a "from <X>" follow-up and X names a model, return the
        resolved model; otherwise "". Keeps "from solar" (an energy source) out
        of the model-switch path so it stays a source/variable refinement."""
        m = re.fullmatch(r"(?:the\s+)?(?:same\s+)?from\s+(.+)", stripped)
        if not m:
            return ""
        target = m.group(1).strip()
        if not (1 <= len(target.split()) <= 4):
            return ""
        try:
            from model_aliases import match_model_name as _mmn
            return _mmn(target, self._model_switch_names()) or ""
        except Exception:
            return ""

    def _is_contextual_dimension_followup(self, query: str) -> bool:
        ql = query.strip().lower()
        # Strip leading filler ("now", "and", ...) so phrasings like
        # "now show only 2050" are still recognised.
        stripped = re.sub(self._FOLLOWUP_FILLER, "", ql).strip().rstrip("?.!").strip()
        if (
            re.fullmatch(
                r"(?:(?:show|display|give)(?:\s+me)?\s+)?"
                r"(?:do\s+)?(?:the\s+)?same(?:\s+[a-z0-9_-]+){0,2}\s+for\s+.+",
                stripped,
            )
            or re.fullmatch(r"same\s+under\s+.+", stripped)
            # Possessive variable switch that keeps the carried region/model:
            # "and its GDP", "its population". A leading plot verb is allowed
            # ("plot its CO2 emissions") -- without it that phrasing matched no
            # follow-up rule at all, so the carried model was dropped and the
            # chart spanned every model in the catalogue.
            or re.fullmatch(r"(?:its|their)\s+.+", stripped)
            or re.fullmatch(
                r"(?:plot|chart|graph|visuali[sz]e|draw)\s+(?:its|their)\s+.+", stripped
            )
            or re.fullmatch(r"(?:what|how)\s+about\s+.+", stripped)
            or re.fullmatch(r"compare\s+(?:with|to|against)\s+.+", stripped)
            or re.fullmatch(r"compare\s+(?:(?:that|it|this)\s+)?(?:with|to|against)\s+\d{4}", stripped)
            or re.fullmatch(
                r"(?:after|before|by|until|up\s+to|through|in|from|since)\s+\d{4}"
                r"(?:\s*(?:to|until|-|and)\s*\d{4})?",
                stripped,
            )
            or re.fullmatch(r"(?:show\s+|plot\s+)?(?:only|just)\s+\d{4}(?:\s*(?:-|to|and)\s*\d{4})?", stripped)
            or stripped == "show all scenarios"
        ):
            return True
        # "same from REMIND" / "now from GCAM": a model switch, but only when the
        # trailing token actually names a model (not an energy source).
        if self._from_switch_model(stripped):
            return True
        # Dimension switches: "under PR_NDC_CP", "now for CHN". A short trailing
        # phrase is the new scenario ("under X") or region ("for X").
        m = re.fullmatch(r"(?:same\s+)?(?:under|for)\s+(.+)", stripped)
        if m and 1 <= len(m.group(1).split()) <= 4:
            return True
        # "plot India" / "chart China": a plot verb followed by a short token that
        # names a carried dimension (region/scenario/model), not a variable.
        pm = re.fullmatch(r"(?:plot|chart|graph|visuali[sz]e|draw)\s+(.+)", stripped)
        if pm and 1 <= len(pm.group(1).split()) <= 4:
            return self._resolve_carry_dimension(pm.group(1).strip()) is not None
        return False

    def _resolve_region_from_text(
        self,
        text: str,
        carried: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Resolve a region mention and prefer the catalogue value compatible
        with the active data scope when aliases and display names coexist."""
        ex = self.entity_extractor
        regions = list(getattr(ex, "available_regions", []) or [])
        candidates: List[str] = []

        def _add(value: Any) -> None:
            value = str(value or "").strip()
            if value and value not in candidates:
                candidates.append(value)

        try:
            _add(extract_region_from_query(
                text,
                getattr(ex, "region_dict", {}) or {},
                regions,
            ))
        except Exception:
            pass
        _add(self._match_catalog_value_from_text(text, regions))
        _add(canonical_region_from_query(text, regions))
        if not candidates:
            return ""

        scope = dict(carried or self.last_entities or {})
        records = [record for record in self.shared_resources.get("ts", []) if isinstance(record, dict)]
        if not records:
            return candidates[0]
        scoped_model = str(scope.get("model") or "").strip()
        resolved_scope_models = set()
        if scoped_model:
            runtime_model_names = sorted({
                str(record.get("modelName") or record.get("model") or "").strip()
                for record in records
                if str(record.get("modelName") or record.get("model") or "").strip()
            })
            resolved_scope_models = set(resolve_model_candidates(scoped_model, runtime_model_names))

        def _compatible_count(candidate: str) -> int:
            count = 0
            for record in records:
                if str(record.get("region", "")).casefold() != candidate.casefold():
                    continue
                variable = str(scope.get("variable") or "").strip()
                if variable and str(record.get("variable", "")).casefold() != variable.casefold():
                    continue
                scenario = str(scope.get("scenario") or "").strip()
                record_scenario = str(record.get("scenario", ""))
                if scenario and not (
                    record_scenario.casefold() == scenario.casefold()
                    or scenario_in_family(record_scenario, scenario)
                ):
                    continue
                record_model = str(record.get("modelName") or record.get("model") or "")
                if scoped_model and record_model not in resolved_scope_models:
                    continue
                count += 1
            return count

        return max(candidates, key=lambda candidate: (_compatible_count(candidate), -candidates.index(candidate)))

    # Carriers and sectors that appear as a segment of a hierarchical variable.
    # A follow-up naming one of these refines the variable rather than naming a
    # new region, which is how "same for wind" used to be read.
    _VARIABLE_SEGMENT_SWITCH_TOKENS = frozenset({
        "solar", "wind", "hydro", "nuclear", "coal", "gas", "oil", "biomass",
        "bioenergy", "hydrogen", "geothermal", "heat", "electricity",
        "transport", "transportation", "industry", "buildings", "residential",
        "commercial",
    })

    def _carrier_switch_variable(self, variable: str, replacement: str) -> str:
        """Swap the carrier/sector segment of the carried variable.

        After "solar electricity for India", the follow-up "same for wind" means
        `Secondary Energy|Electricity|Wind` -- not region "wind". Without this the
        token falls through to the region slot and the chart keeps showing solar,
        which is the clarification loop reported for that sequence. Returns "" when
        the swap does not name a real catalogue variable, so the caller can fall
        back to its existing behaviour.
        """
        base = str(variable or "").strip()
        token = str(replacement or "").strip().casefold()
        if not base or "|" not in base or not token:
            return ""
        if token not in self._VARIABLE_SEGMENT_SWITCH_TOKENS:
            return ""
        # No region check here. None of these tokens names a region, and asking
        # the region resolver about a bare word gives a false positive anyway --
        # it substring-matches codes inside unrelated words, so "wind" resolves
        # to IND (India) and "coal" to COL (Colombia). Consulting it aborted
        # every carrier switch. Multi-word queries are unaffected, so this is
        # scoped to the follow-up token only.
        segments = base.split("|")
        if segments[-1].casefold() == token:
            return ""
        available = {
            str(value).strip()
            for value in (getattr(self.entity_extractor, "available_variables", None) or [])
        }
        if not available:
            return ""
        for index in range(len(segments) - 1, 0, -1):
            if segments[index].casefold() not in self._VARIABLE_SEGMENT_SWITCH_TOKENS:
                continue
            for spelling in (replacement.strip(), token.title(), token.capitalize(), token.upper()):
                candidate = "|".join(segments[:index] + [spelling])
                if candidate in available:
                    return candidate
        return ""

    def _resolve_carry_dimension(self, token: str):
        """Classify a short follow-up token as a region/scenario/model so a
        phrasing like "plot India" can be understood as a dimension switch on the
        carried scope rather than a fresh plot of a variable. Returns a
        (kind, value) tuple or None."""
        t = str(token or "").strip()
        if not t:
            return None
        ex = self.entity_extractor
        resolved_region = self._resolve_region_from_text(t)
        if resolved_region:
            return ("region", resolved_region)
        scen = self._match_scenario_from_text(t)
        if scen:
            return ("scenario", scen)
        models = getattr(ex, "available_models", []) or []
        try:
            from model_aliases import match_model_name
            m = match_model_name(t, models)
            if m:
                return ("model", m)
        except Exception:
            pass
        return None

    def _is_result_provenance_question(self, query: str) -> bool:
        """A follow-up asking which model the previous result came from, e.g.
        "which model is this from?", "what model is this?"."""
        ql = re.sub(self._FOLLOWUP_FILLER, "", query.strip().lower()).strip().rstrip("?").strip()
        if ql in {
            "which model", "what model", "which models", "what models",
            "source model", "which model is this", "what model is this",
        }:
            return True
        result_reference = bool(re.search(
            r"\b(?:this|that|these|those|previous|last|above)\b[^.?!]*"
            r"\b(?:result|results|value|values|number|numbers|table|plot|data)\b",
            ql,
        ))
        source_language = bool(
            re.search(r"\b(?:which|what)\s+models?\b", ql)
            or re.search(r"\b(?:source|provenance|produced|reported|generated|come\s+from|comes\s+from)\b", ql)
        )
        if result_reference and source_language:
            return True
        return bool(
            re.fullmatch(r"(?:which|what)\s+models?\s+(?:is|are|was|were)\s+(?:this|that|it|these|those)(?:\s+from)?", ql)
            or re.fullmatch(r"(?:which|what)\s+models?\s+(?:did|does)\s+(?:this|that|it)\s+come\s+from", ql)
            or re.fullmatch(r"(?:where|which\s+model)\s+(?:is|are)\s+(?:this|that|these|those|it)\s+from", ql)
        )

    def _result_scope_dimension(self, query: str) -> str:
        """Identify a dimension requested from the latest result object.

        A result reference is mandatory, so ordinary discovery questions such
        as "which models are available?" continue to use the global catalogue.
        """
        ql = re.sub(self._FOLLOWUP_FILLER, "", str(query or "").strip().lower()).rstrip("?.! ")
        dimensions = {
            "model": r"models?",
            "scenario": r"scenarios?|pathways?",
            "region": r"regions?|countries|geograph(?:y|ies)",
            "variable": r"variables?|metrics?|indicators?",
        }
        result_reference = bool(re.search(
            r"\b(?:this|that|these|those|it|its|previous|last|above)\b|"
            r"\b(?:result|results|plot|chart|graph|table|data)\b",
            ql,
        ))
        result_operation = bool(re.search(
            r"\b(?:shown|showing|used|included|contributed|produced|reported|behind|in)\b",
            ql,
        ))
        if not (result_reference and result_operation):
            return ""
        for dimension, pattern in dimensions.items():
            if re.search(r"\b(?:which|what)\s+" + pattern + r"\b", ql):
                return dimension
        return ""

    def _result_scope_answer(self, dimension: str) -> Optional[str]:
        scope = dict(getattr(self, "last_entities", {}) or {})
        values: List[str] = []
        if dimension == "model":
            result_models = (
                getattr(self, "last_result_models", [])
                or scope.get("result_models")
                or scope.get("models")
                or ([scope.get("model")] if scope.get("model") else [])
            )
            values = [
                display_model_label(value)
                for value in result_models
                if value
            ]
        else:
            plural = {"scenario": "scenarios", "region": "regions", "variable": "variables"}[dimension]
            values = [str(value) for value in (scope.get(plural) or []) if value]
            singular = str(scope.get(dimension) or "").strip()
            if not values and singular:
                if dimension == "scenario":
                    available = getattr(self.entity_extractor, "available_scenarios", []) or []
                    values = scenario_family_members(singular, available)
                if not values:
                    values = [singular]
        values = list(dict.fromkeys(value for value in values if value))
        if not values:
            return None
        label = {"model": "models", "scenario": "scenarios", "region": "regions", "variable": "variables"}[dimension]
        listed = ", ".join(f"`{value}`" for value in values)
        return f"The latest result includes these {label}: {listed}."

    def _failed_scope_comparison_answer(self, query: str) -> Optional[str]:
        """Explain why a comparison cannot use a failed attempted scope."""
        if not (
            _looks_like_plot_request(query)
            and _looks_like_comparison_request(query)
            and getattr(self, "last_attempted_entities", None)
            and self.last_entities
        ):
            return None
        attempted = dict(self.last_attempted_entities or {})
        successful = dict(self.last_entities or {})
        changed = [
            key for key in ("region", "scenario", "model", "variable")
            if attempted.get(key) and successful.get(key)
            and str(attempted[key]) != str(successful[key])
        ]
        if len(changed) != 1:
            return None
        dimension = changed[0]
        return (
            f"I cannot build that comparison yet because the latest `{dimension}` scope "
            f"`{attempted[dimension]}` did not return data. The latest successful "
            f"`{dimension}` scope is `{successful[dimension]}`. Choose another "
            f"{dimension} or broaden the current filters first."
        )

    def _result_provenance_answer(self) -> Optional[str]:
        scope = dict(getattr(self, "last_entities", {}) or {})
        result_models = (
            getattr(self, "last_result_models", None)
            or scope.get("result_models")
            or scope.get("models")
            or ([scope.get("model")] if scope.get("model") else [])
        )
        models = [
            display_model_label(value)
            for value in result_models
            if value
        ]
        models = list(dict.fromkeys(models))
        if not models:
            return None
        if len(models) == 1:
            return f"That result comes from model `{models[0]}`."
        shown = models[:8]
        more = "" if len(models) <= len(shown) else f" and {len(models) - len(shown)} more"
        listed = ", ".join(f"`{m}`" for m in shown)
        return f"That result includes data from these models: {listed}{more}."

    def _is_model_scope_followup(self, query: str) -> bool:
        """A question about the scenarios/variables/regions of the model just
        discussed, referred to by pronoun (e.g. "what scenarios does it have").
        Recognising it lets the carried model flow into the data query so the
        answer is scoped to that model instead of an unscoped overview."""
        ql = query.strip().lower()
        if not re.search(r"\b(scenario|scenarios|variable|variables|region|regions)\b", ql):
            return False
        if not re.search(r"\b(does|do|has|have|run|runs|use|uses|cover|covers)\b", ql):
            return False
        return bool(re.search(r"\b(it|its|this model|that model|the model)\b", ql))

    _SMALL_TALK_GREETINGS = {
        "hi", "hello", "hey", "hiya", "good morning", "good afternoon", "good evening",
    }
    _SMALL_TALK_THANKS = {
        "thanks", "thank you", "thanks a lot", "many thanks", "thx", "ty", "cheers",
    }
    _SMALL_TALK_FAREWELLS = {"bye", "goodbye", "see you", "good night"}
    _SMALL_TALK_CAPABILITIES = {
        "help", "what can you do", "what can you do?", "who are you", "who are you?",
        "what is this", "what is this?", "how do you work", "how do you work?",
    }

    def _is_small_talk(self, query: str) -> bool:
        ql = re.sub(r"[!.?\s]+$", "", str(query or "").strip().lower())
        return ql in (
            self._SMALL_TALK_GREETINGS
            | self._SMALL_TALK_THANKS
            | self._SMALL_TALK_FAREWELLS
            | self._SMALL_TALK_CAPABILITIES
        )

    def _small_talk_answer(self, query: str) -> str:
        ql = re.sub(r"[!.?\s]+$", "", str(query or "").strip().lower())
        capabilities = (
            "I answer questions about IAM PARIS climate data (https://iamparis.eu/). "
            "You can ask me to:\n"
            "- Show data, e.g. `show CO2 emissions for Europe under Baseline`\n"
            "- Plot data, e.g. `plot solar capacity in Greece`\n"
            "- List what is available, e.g. `list models`, `show all scenarios`\n"
            "- Explain a model, e.g. `what is GCAM?`\n"
            "- Find IAM PARIS pages, e.g. `where can I find the policy catalogue?`"
        )
        if ql in self._SMALL_TALK_THANKS:
            return "You're welcome! Ask me anything else about IAM PARIS data whenever you like."
        if ql in self._SMALL_TALK_FAREWELLS:
            return "Goodbye! Come back anytime to explore IAM PARIS data."
        if ql in self._SMALL_TALK_CAPABILITIES:
            return capabilities
        return f"Hello! {capabilities}"

    def _is_clarification_followup(self, query: str, context: Optional[Dict[str, Any]] = None) -> bool:
        q = str(query or "").strip()
        if not q:
            return False
        # Greetings/thanks must never be consumed as a clarification answer.
        if self._is_small_talk(q):
            return False
        if self._is_affirmation(q) or self._is_rejection(q) or self._is_generic_followup(q):
            return True

        option_count = len((context or {}).get("suggested_options", []) or [])
        if self._extract_option_choice(q, option_count) is not None:
            return True

        # Single values like "AUS", "GDP|MER", or scenario shorthand should still count as follow-ups.
        token_count = len(re.findall(r"\S+", q))
        if token_count <= 4:
            if "|" in q:
                return True
            if self._match_scenario_from_text(q):
                return True
            if not (
                _looks_like_data_request(q)
                or _looks_like_plot_request(q)
                or _looks_like_model_info_request(q)
                or _looks_like_category_list_request(q, "models")
                or _looks_like_category_list_request(q, "variables")
                or _looks_like_category_list_request(q, "regions")
                or _looks_like_category_list_request(q, "scenarios")
            ):
                return True

        return False

    def _is_affirmation(self, query: str) -> bool:
        ql = query.strip().lower()
        return ql in {
            "yes", "y", "yeah", "yep", "ok", "okay", "sure", "correct",
            "use it", "sounds good", "that's right", "right"
        } or ql.startswith("yes ")

    def _is_rejection(self, query: str) -> bool:
        ql = query.strip().lower()
        return ql in {"no", "n", "nope", "nah", "not that", "different"}

    def _normalize_text(self, value: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", (value or "").lower())

    @staticmethod
    def _match_catalog_value_from_text(text: str, values) -> str:
        """Return the longest runtime-catalog value explicitly named in text."""
        query = str(text or "")
        candidates = sorted(
            {str(value).strip() for value in (values or []) if str(value).strip()},
            key=lambda value: (-len(value), value.casefold()),
        )
        for value in candidates:
            # Short catalogue codes are collision-prone natural-language words
            # (for example a pronoun may equal a two-letter region code when
            # compared case-insensitively). Trust them only when the user keeps
            # the catalogue casing or explicitly labels the dimension.
            if re.fullmatch(r"[A-Z0-9-]{2,3}", value):
                exact_case = re.search(r"(?<![\w-])" + re.escape(value) + r"(?![\w-])", query)
                labelled = re.search(
                    r"\b(?:region|country|geography|area|code)\s+" + re.escape(value) + r"\b",
                    query,
                    flags=re.IGNORECASE,
                )
                if not exact_case and not labelled:
                    continue
            pattern = r"(?<![\w-])" + re.escape(value) + r"(?![\w-])"
            if re.search(pattern, query, flags=re.IGNORECASE):
                return value
        return ""

    @staticmethod
    def _best_supported_variable_path(
        parent: str,
        current: str,
        query: str,
        available_variables,
    ) -> str:
        """Choose the most specific catalogue path supported by user words.

        A literal parent name may occur inside a more specific request.  Rank
        its runtime descendants by evidence in each extra path segment, while
        penalising unsupported segments.  A broad parent-only query therefore
        stays broad, a directly supported child wins, and an extractor cannot
        invent an unrelated intermediate branch.  The taxonomy itself remains
        entirely runtime-driven.
        """
        parent = str(parent or "").strip()
        current = str(current or "").strip()
        if not parent:
            return current

        def lexemes(value: str) -> set[str]:
            return {
                token
                for token in re.findall(r"[a-z0-9]+", str(value or "").casefold())
                if len(token) >= 2 and token not in {"and", "or", "of", "the", "incl", "excl"}
            }

        query_tokens = lexemes(query)

        def overlaps(segment_token: str) -> bool:
            return any(
                segment_token == query_token
                or (
                    min(len(segment_token), len(query_token)) >= 4
                    and (
                        segment_token.startswith(query_token)
                        or query_token.startswith(segment_token)
                    )
                )
                for query_token in query_tokens
            )

        candidates = sorted({
            str(value).strip()
            for value in (available_variables or [])
            if str(value or "").strip() == parent
            or str(value or "").strip().startswith(parent + "|")
        })
        if parent not in candidates:
            candidates.append(parent)

        preferred = preferred_variable_from_query(query, candidates)
        ranked: List[Tuple[Tuple[int, int, int, int, int, str], str]] = []
        for value in candidates:
            if value == parent:
                ranked.append(((0, 0, 0, int(value == current), 0, value), value))
                continue
            suffix = value[len(parent) + 1:]
            segments = [lexemes(part) for part in suffix.split("|")]
            segments = [tokens for tokens in segments if tokens]
            supported = sum(
                1 for tokens in segments if any(overlaps(token) for token in tokens)
            )
            if supported == 0 and value != preferred:
                continue
            unsupported = max(len(segments) - supported, 0)
            rank = (
                int(value == preferred),
                -unsupported,
                supported,
                int(value == current),
                -len(segments),
                value,
            )
            ranked.append((rank, value))

        if not ranked:
            return parent
        ranked.sort(key=lambda row: row[0], reverse=True)
        return ranked[0][1]

    def _match_scenario_from_text(self, query: str) -> str:
        """
        Resolve shorthand scenario mentions in follow-ups, e.g. "pr wwh cp" -> "PR_WWH_CP".
        """
        scenarios = getattr(self.entity_extractor, "available_scenarios", []) or []
        if not scenarios:
            return ""
        q_norm = self._normalize_text(query)
        if not q_norm:
            return ""

        # A known region is never a scenario. Without this, a follow-up like
        # "same for EU" would let "eu" containment-match "NZE_EUPol_Stand".
        regions = getattr(self.entity_extractor, "available_regions", []) or []
        if any(self._normalize_text(str(r)) == q_norm for r in regions):
            return ""

        # Exact normalized match first
        for scen in scenarios:
            if self._normalize_text(str(scen)) == q_norm:
                return str(scen)

        # Containment match for shorthand follow-ups. Require a few characters so
        # a tiny token ("eu", "in") cannot match as a substring of a longer code.
        if len(q_norm) < 3:
            return ""
        for scen in scenarios:
            s_norm = self._normalize_text(str(scen))
            if not s_norm:
                continue
            if q_norm in s_norm or s_norm in q_norm:
                return str(scen)
        return ""

    def _extract_best_candidate(self, response: str) -> str:
        match = re.search(r"best match is `([^`]+)`", response or "", re.IGNORECASE)
        return match.group(1).strip() if match else ""

    def _extract_candidate_options(self, response: str) -> List[str]:
        """
        Parse numbered/backticked options from clarification prompts.
        """
        if not response:
            return []

        numbered_choices = self._extract_candidate_choices(response)
        if numbered_choices:
            deduped: List[str] = []
            for _kind, item in numbered_choices:
                val = str(item).strip()
                if val and val not in deduped:
                    deduped.append(val)
            return deduped

        options: List[str] = []
        best = self._extract_best_candidate(response)
        if best:
            options.append(best)

        other = re.search(r"Other close options:\s*([^\.]+)", response or "", re.IGNORECASE)
        if other:
            for raw in other.group(1).split(","):
                opt = raw.strip().strip("`")
                if opt and opt not in options:
                    options.append(opt)
        return options

    @staticmethod
    def _extract_candidate_choices(response: str) -> List[Tuple[str, str]]:
        """Parse numbered choices while preserving each option's dimension."""
        choices: List[Tuple[str, str]] = []
        valid_kinds = {"variable", "region", "scenario", "model"}
        for kind, value in re.findall(
            r"\b\d+\.\s*(?:(variable|region|scenario|model)\s+)?`([^`]+)`",
            str(response or ""),
            flags=re.IGNORECASE,
        ):
            normalized_kind = str(kind or "").strip().lower()
            normalized_value = str(value or "").strip()
            if normalized_value:
                choices.append((
                    normalized_kind if normalized_kind in valid_kinds else "",
                    normalized_value,
                ))

        # Plot recovery prompts use compact typed groups instead of one
        # numbered list. Parse those groups into the same typed representation
        # so data and plot clarification share one state machine.
        plural_to_kind = {
            "variables": "variable",
            "regions": "region",
            "scenarios": "scenario",
            "models": "model",
        }
        for plural_kind, values in re.findall(
            r"Closest\s+(variables|regions|scenarios|models):\s*([^\n]+)",
            str(response or ""),
            flags=re.IGNORECASE,
        ):
            kind = plural_to_kind[plural_kind.casefold()]
            for value in re.findall(r"`([^`]+)`", values):
                normalized_value = str(value or "").strip()
                choice = (kind, normalized_value)
                if normalized_value and choice not in choices:
                    choices.append(choice)
        return choices

    def _extract_named_option_choice(
        self,
        query: str,
        options: List[str],
    ) -> Optional[int]:
        """Resolve a short reply that names one pending catalogue option."""
        normalized_query = self._normalize_text(
            re.sub(
                r"^(?:use|choose|select|pick)(?:\s+the)?\s+",
                "",
                str(query or "").strip(),
                flags=re.IGNORECASE,
            )
        )
        if not normalized_query:
            return None
        matches = [
            index for index, option in enumerate(options)
            if self._normalize_text(option) == normalized_query
        ]
        return matches[0] if len(matches) == 1 else None

    @staticmethod
    def _apply_clarification_option(
        context: Dict[str, Any],
        option_index: int,
    ) -> bool:
        """Apply one typed pending choice without leaking another option's type."""
        options = list(context.get("suggested_options", []) or [])
        if not (0 <= option_index < len(options)):
            return False
        option_kinds = list(context.get("suggested_option_kinds", []) or [])
        kind = (
            str(option_kinds[option_index]).strip().lower()
            if option_index < len(option_kinds)
            else str(context.get("suggested_kind", "variable") or "variable").strip().lower()
        )
        if kind not in {"variable", "region", "scenario", "model"}:
            kind = "variable"

        original_entities = dict(context.get("entities", {}) or {})
        for field in ("variable", "region", "scenario", "model"):
            context[f"suggested_{field}"] = str(original_entities.get(field) or "").strip()
        context["suggested_kind"] = kind
        context[f"suggested_{kind}"] = str(options[option_index]).strip()
        context["selected_option_index"] = option_index
        return True

    @staticmethod
    def _clarification_options_prompt(
        prefix: str,
        choices: List[Tuple[str, str]],
    ) -> str:
        """Render homogeneous or mixed typed options from conversation state."""
        cleaned = [
            (str(kind or "variable").strip().lower(), str(value or "").strip())
            for kind, value in choices
            if str(value or "").strip()
        ]
        if not cleaned:
            return prefix
        kinds = {kind for kind, _value in cleaned}
        if len(kinds) == 1:
            kind = next(iter(kinds))
            return _choice_prompt(prefix, kind, [value for _kind, value in cleaned])
        lines = [prefix, "", "Closest valid options:", ""]
        lines.extend(
            f"{index}. {kind} `{value}`"
            for index, (kind, value) in enumerate(cleaned, start=1)
        )
        lines.extend([
            "",
            f"Reply with a number (1-{len(cleaned)}), or type the option you want.",
        ])
        return "\n".join(lines)

    def _extract_option_choice(self, query: str, option_count: int) -> Optional[int]:
        """
        Return 0-based option index when user replies with a number like "2" or "option 2".
        """
        if option_count <= 0:
            return None
        ql = (query or "").strip().lower()
        ordinal_map = {
            "first": 0,
            "1st": 0,
            "second": 1,
            "2nd": 1,
            "third": 2,
            "3rd": 2,
        }
        for word, index in ordinal_map.items():
            if re.search(r"\b" + re.escape(word) + r"\b", ql) and index < option_count:
                return index
        # A bare number or "option N"/"number N" selects an option. A number
        # embedded in a longer sentence ("show me 3 scenarios") must not.
        match = re.fullmatch(
            r"(?:yes,?\s+)?(?:(?:use\s+)?(?:the\s+)?(?:option|choice|number|no\.?)\s*)?([1-9][0-9]*)\s*[.)]?",
            ql,
        )
        if not match:
            return None
        num = int(match.group(1))
        if 1 <= num <= option_count:
            return num - 1
        return None

    @staticmethod
    def _finalize_clarification_entities(
        entities: Optional[Dict[str, Any]],
        selected_kind: str = "",
        selected_value: str = "",
        *,
        confirmed: bool = False,
    ) -> Dict[str, Any]:
        """Return entity state with resolved clarification diagnostics removed.

        Candidate lists and unmatched-term diagnostics describe a pending
        decision. They must not survive after the user supplies that decision,
        otherwise later turns can re-open or rank against stale alternatives.
        """
        resolved = dict(entities or {})
        for key in list(resolved):
            normalized = str(key).casefold()
            if (
                normalized.startswith("unmatched_")
                or normalized.startswith("candidate_")
                or normalized.endswith("_candidate")
                or normalized.endswith("_candidates")
            ):
                resolved.pop(key, None)

        kind = str(selected_kind or "").strip().casefold()
        value = str(selected_value or "").strip()
        plural_keys = {
            "variable": "variables",
            "region": "regions",
            "scenario": "scenarios",
            "model": "models",
        }
        if kind in plural_keys and value:
            resolved[kind] = value
            resolved.pop(plural_keys[kind], None)
            if confirmed:
                resolved.pop(f"{kind}_matched", None)

        confidence = dict(resolved.get("entity_confidence") or {})
        if kind in plural_keys and value and confirmed:
            confidence[kind] = 1.0

        # Confidence is meaningful only for dimensions still present in the
        # resolved entity state (plus the extractor's structural fields).
        structural_confidence = {
            "action": bool(resolved.get("action")),
            "comparison": bool(resolved.get("comparison")),
            "years": resolved.get("start_year") is not None or resolved.get("end_year") is not None,
        }
        for field in list(confidence):
            if field in structural_confidence:
                if not structural_confidence[field]:
                    confidence.pop(field, None)
            elif not resolved.get(field):
                confidence.pop(field, None)

        if confidence:
            resolved["entity_confidence"] = confidence
        else:
            resolved.pop("entity_confidence", None)
        scored = [
            score
            for field, score in confidence.items()
            if field != "action" and isinstance(score, (int, float))
        ]
        if not scored and isinstance(confidence.get("action"), (int, float)):
            scored = [confidence["action"]]
        resolved["confidence"] = round(sum(scored) / len(scored), 3) if scored else 0.0
        return resolved

    def _record_clarification_result_route(
        self,
        agent_name: str,
        response: str,
        reason: str = "clarification selection",
    ) -> None:
        routed_agent = agent_name if agent_name in VALID_AGENT_NAMES else "data_query"
        needs_more_detail = self._looks_like_clarification_response(response)
        self._record_route_decision(
            routed_agent,
            0.72 if needs_more_detail else 0.98,
            "conversation_state",
            f"{reason}; " + ("additional clarification required" if needs_more_detail else "resolved"),
        )

    def _update_clarification_context(
        self,
        agent_name: str,
        query: str,
        response: str,
        entities: Optional[Dict[str, Any]] = None,
        base_query: Optional[str] = None,
    ) -> None:
        entities = self._finalize_clarification_entities(entities or {})
        if agent_name in {"data_query", "data_plotting"} and self._looks_like_clarification_response(response):
            typed_choices = self._extract_candidate_choices(response)
            options = (
                [value for _kind, value in typed_choices]
                if typed_choices
                else self._extract_candidate_options(response)
            )
            suggested_kind = "variable"
            if typed_choices and typed_choices[0][0]:
                suggested_kind = typed_choices[0][0]
            elif "Choose the region:" in response:
                suggested_kind = "region"
            elif "Choose the scenario:" in response:
                suggested_kind = "scenario"
            elif "Which region should I use?" in response:
                suggested_kind = "region"
            elif "Which scenario should I use?" in response:
                suggested_kind = "scenario"
            elif "Closest valid options:" in response:
                first_option = re.search(
                    r"\b1\.\s*(variable|region|scenario)\s+`",
                    response,
                    re.IGNORECASE,
                )
                if first_option:
                    suggested_kind = first_option.group(1).lower()
            option_kinds = [
                kind if kind else suggested_kind
                for kind, _value in typed_choices
            ] if typed_choices else [suggested_kind for _value in options]
            suggested_variable = ""
            suggested_region = str(entities.get("region", "") or "").strip()
            suggested_scenario = str(entities.get("scenario", "") or "").strip()
            if suggested_kind == "variable":
                suggested_variable = options[0] if options else self._extract_best_candidate(response)
            elif suggested_kind == "region":
                suggested_region = options[0] if options else suggested_region
            elif suggested_kind == "scenario":
                suggested_scenario = options[0] if options else suggested_scenario
            self.clarification_context = {
                "clarification_id": f"clarification-{getattr(self, 'current_turn', 0)}",
                "original_query": query,
                "base_query": base_query or query,
                "agent_type": agent_name,
                "entities": entities,
                "suggested_variable": suggested_variable,
                "suggested_options": options,
                "suggested_option_kinds": option_kinds,
                "suggested_kind": suggested_kind,
                "suggested_region": suggested_region,
                "suggested_scenario": suggested_scenario,
                "response": response,
                "ambiguous_response": response,
                "issued_turn": getattr(self, "current_turn", 0),
            }
            if options:
                self._apply_clarification_option(self.clarification_context, 0)

    def _route_single(
        self,
        query: str,
        history: Optional[List[Tuple[str, str]]] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Route a single-intent query."""
        # Be tolerant of copied chat prefixes like "YOU: YOU: 1"
        query = re.sub(r"^(?:\s*you:\s*)+", "", str(query or ""), flags=re.IGNORECASE).strip()
        # Per-turn API presentation state.  A navigation answer may suppress
        # data entities without erasing the successful scope needed by a later
        # "plot it" follow-up.
        self._conversation().response_scope_override = None
        original_user_query = query
        q_lower = query.strip().lower()
        # Restored sessions can contain a choice issued by the former plot
        # route. Resolve that choice through the numeric-data agent as well.
        if self.clarification_context and self.clarification_context.get("agent_type") == "data_plotting":
            self.clarification_context["agent_type"] = "data_query"
            self.clarification_context.setdefault("entities", {})["action"] = "query"
        available_models = getattr(self.entity_extractor, "available_models", []) or []
        plan = build_query_plan(query, available_models=available_models)
        if plan.replacement_dimension == "auto" and plan.replacement_value:
            replacement = self._resolve_carry_dimension(plan.replacement_value)
            if replacement:
                plan = replace(
                    plan,
                    replacement_dimension=replacement[0],
                    replacement_value=replacement[1],
                    scope_patch=ScopePatch(
                        replacement_dimension=replacement[0],
                        replacement_value=replacement[1],
                        output_mode=plan.output_mode,
                        year_filter=plan.year_filter,
                    ),
                )

        # Charts are intentionally kept in the IAM PARIS data explorer.  The
        # explorer provides the full study context and interactive filters, so
        # do not create a partial static figure inside the chat.
        if _looks_like_plot_request(query) and not self._result_scope_dimension(query):
            self.clarification_context = None
            self.last_links = []
            self._conversation().response_scope_override = {}
            self._record_route_decision(
                "general_qa", 1.0, "deterministic", "charts delegated to data explorer",
            )
            workspace_entries = _workspace_entries(list(self.shared_resources.get("ts") or []))
            named_workspace = _matched_workspace(original_user_query, workspace_entries)
            carried_workspace = str((self.last_entities or {}).get("workspace_code") or "").strip()
            workspace_entry = named_workspace or next(
                (
                    entry for entry in workspace_entries
                    if str(entry.get("code") or "").strip() == carried_workspace
                ),
                {},
            )
            if named_workspace:
                # A named study in a redirect is an explicit context switch.
                # Retain it for the next "this study" request while keeping
                # this navigation response free of data-scope metadata.
                self._conversation().record_success({
                    "workspace_code": str(named_workspace.get("code") or "").strip(),
                })
                self._conversation().response_scope_override = {}
            explorer_url = str(workspace_entry.get("explorer_url") or "https://iamparis.eu/results")
            return (
                "I do not generate figures in the chat. Please use the "
                f"[IAM PARIS data explorer]({explorer_url}) to explore, filter, and plot the data. "
                "Ask about a specific study first if you would like its direct explorer link."
            )

        # “What results are available?” is study discovery, not a general
        # knowledge question. Keep it on the deterministic data path so the
        # user sees public study titles before choosing a workspace.
        if re.fullmatch(
            r"\s*(?:(?:what|which)\s+)?(?:results?|stud(?:y|ies)|workspaces?)\s+"
            r"(?:are\s+)?(?:available|loaded|included)(?:\s+(?:here|now))?[?.!]?\s*",
            q_lower,
        ):
            agent = self.agents.get("data_query")
            if agent:
                self._record_route_decision(
                    "data_query", 1.0, "deterministic", "study/results catalogue request",
                )
                response = agent.handle(query, history)
                self._persist_last_entities({}, response)
                return self._append_relevant_links(response, query, {}, "data_query")

        # Greetings/thanks/help: answer directly and keep any pending
        # clarification context intact for the next real message.
        if self._is_small_talk(query):
            self._record_route_decision("general_qa", 0.95, "deterministic", "greeting/small talk")
            self.last_links = []
            if self.clarification_context:
                # A greeting should not burn the clarification window.
                self.clarification_context["issued_turn"] = getattr(self, "current_turn", 0)
            return self._small_talk_answer(query)

        # A request for the meaning of the generic model class is not a model
        # catalogue lookup. Resolve it before navigation and entity extraction,
        # where the noun "model" can otherwise become the Models page or a
        # fuzzy runtime model alias.
        if _is_integrated_assessment_model_concept_question(original_user_query):
            self.clarification_context = None
            self.last_links = []
            self._conversation().response_scope_override = {}
            self._record_route_decision(
                "general_qa",
                1.0,
                "deterministic",
                "integrated-assessment-model concept definition",
            )
            return _integrated_assessment_model_concept_answer()

        if _is_iam_vs_energy_system_question(original_user_query):
            self.clarification_context = None
            self.last_links = []
            self._conversation().response_scope_override = {}
            self._record_route_decision(
                "general_qa", 1.0, "deterministic", "model-class comparison",
            )
            return _iam_vs_energy_system_answer()

        if _is_scenario_assumptions_concept_question(original_user_query):
            self.clarification_context = None
            self.last_links = []
            self._conversation().response_scope_override = {}
            self._record_route_decision(
                "general_qa", 1.0, "deterministic", "scenario-assumptions concept explanation",
            )
            return _scenario_assumptions_concept_answer()

        # An explicit destination grounded in the live link catalogue is a
        # navigation request even when its text also names multiple models and
        # contains comparison language.  Resolve this before model metadata so
        # routing follows the requested page rather than the nouns on it.
        contextual_plural_model_reference = bool(re.search(
            r"\b(?:(?:both|these|those|the\s+two)\s+models?|"
            r"both\s+of\s+them|(?:these|those)\s+ones)\b",
            original_user_query,
            flags=re.IGNORECASE,
        ))
        explicitly_names_model = bool(plan.mentioned_models) or self._mentions_known_model(
            original_user_query
        )
        if (
            self.shared_resources.get("link_catalog")
            and self._is_site_navigation_request(original_user_query)
            and not (contextual_plural_model_reference and not explicitly_names_model)
        ):
            self._record_route_decision(
                "general_qa", 0.97, "deterministic", "explicit site navigation request",
            )
            self._conversation().response_scope_override = {}
            return self._grounded_site_navigation_answer(original_user_query, {})

        unresolved_model = str((self.last_entities or {}).get("unresolved_model") or "").strip()
        unresolved_reference = bool(re.search(
            r"\b(?:it|its|this\s+model|that\s+model|the\s+model)\b",
            query,
            re.IGNORECASE,
        ))
        if unresolved_model and unresolved_reference and not self._mentions_known_model(query):
            asks_catalog_scope = bool(re.search(
                r"\b(?:scenario|scenarios|variable|variables|region|regions|data|available|availability)\b",
                query,
                re.IGNORECASE,
            ))
            asks_reference_link = bool(re.search(
                r"\b(?:link|page|documentation|docs|read|browse|website)\b",
                query,
                re.IGNORECASE,
            ))
            if asks_catalog_scope or asks_reference_link:
                response = (
                    f"I still cannot resolve `{unresolved_model}` to a model in the currently loaded "
                    "IAM PARIS catalogue, so I cannot safely attach model-specific scenarios, data, or a "
                    "model-specific URL. Ask `list models` or give me the exact catalogue name."
                )
                self._record_route_decision(
                    "model_explanation", 0.98, "conversation_state", "unresolved model reference",
                )
                if asks_reference_link:
                    return self._append_relevant_links(
                        response, query, {"models": [unresolved_model]}, "model_explanation",
                    )
                self.last_links = []
                return response

        prior_scope = dict((context or {}).get("last_entities") or self.last_entities or {})
        prior_models = list(prior_scope.get("models") or [])
        if not prior_models and prior_scope.get("model"):
            prior_models = [prior_scope.get("model")]
        singular_deictic_reference = bool(re.search(
            r"\b(?:it|its|this\s+model|that\s+model|the\s+model)\b",
            original_user_query,
            re.IGNORECASE,
        ))
        elliptical_model_followup = bool(
            not singular_deictic_reference
            and (getattr(self, "last_route_decision", {}) or {}).get("agent") == "model_explanation"
        )
        qualitative_comparison = bool(re.search(
            r"\b(?:compar(?:e|ed|es|ing|ison)|contrast(?:ed|ing|s)?|"
            r"differ(?:s|ed|ent|ently|ence|ences)?|versus|vs\.?)\b",
            original_user_query,
            re.IGNORECASE,
        ))
        singular_qualitative_reference = bool(
            len(prior_models) == 1
            and not (self._runtime_model_profiles(original_user_query) or find_model_profiles(original_user_query))
            and (singular_deictic_reference or elliptical_model_followup)
            and _looks_like_model_info_request(original_user_query)
            and not qualitative_comparison
            and not _looks_like_plot_request(original_user_query)
            and not _looks_like_data_request(original_user_query)
            and plan.intent != "availability"
            and not self._is_site_navigation_request(original_user_query)
        )
        if singular_qualitative_reference:
            resolved = self._runtime_model_profiles(str(prior_models[0])) or find_model_profiles(
                str(prior_models[0])
            )
            if resolved:
                profile = resolved[0]
                response = format_model_profile_answer(
                    profile,
                    requested_name=str(profile.get("name") or prior_models[0]),
                    asks_assumptions=bool(re.search(r"\bassumptions?\b", q_lower)),
                    asks_limitations=bool(re.search(
                        r"\b(?:limitations?|caveats?|constraints?)\b", q_lower,
                    )),
                    query=original_user_query,
                )
                self.last_entities = {"model": str(profile.get("name") or prior_models[0])}
                self._record_route_decision(
                    "model_explanation", 0.97, "conversation_state", "singular-model qualitative follow-up",
                )
                return self._append_relevant_links(
                    response, original_user_query, self.last_entities, "model_explanation",
                )

        profiles = self._runtime_model_profiles(query) or find_model_profiles(query)
        family_subject, family_candidates = self._unqualified_model_family_candidates(query)
        if family_subject and family_candidates:
            if len(family_candidates) > 1:
                response = self._clarification_options_prompt(
                    f"`{family_subject}` matches multiple loaded IAM PARIS model entries. Which one do you mean?",
                    [("model", value) for value in family_candidates],
                )
                self.clarification_context = {
                    "clarification_id": f"clarification-{getattr(self, 'current_turn', 0)}",
                    "original_query": query,
                    "base_query": query,
                    "agent_type": "model_explanation",
                    "entities": {},
                    "suggested_options": list(family_candidates),
                    "suggested_option_kinds": ["model"] * len(family_candidates),
                    "suggested_kind": "model",
                    "suggested_model": family_candidates[0],
                    "response": response,
                    "ambiguous_response": response,
                    "issued_turn": getattr(self, "current_turn", 0),
                }
                self.last_entities = {"models": list(family_candidates)}
                self._record_route_decision(
                    "model_explanation", 0.72, "runtime_catalog", "ambiguous model family",
                )
                self.last_links = []
                return response

            matched_name = family_candidates[0]
            matched_profiles = self._runtime_model_profiles(matched_name) or find_model_profiles(matched_name)
            if matched_profiles:
                profile = matched_profiles[0]
                response = (
                    f"I matched `{family_subject}` to the only loaded catalogue entry in that family, "
                    f"`{matched_name}`.\n\n"
                    + format_model_profile_answer(
                        profile,
                        requested_name=matched_name,
                        query=query,
                    )
                )
                self.last_entities = {"model": matched_name}
                self._record_route_decision(
                    "model_explanation", 0.96, "runtime_catalog", "explicit model-family match",
                )
                return self._append_relevant_links(
                    response, query, self.last_entities, "model_explanation",
                )
        model_subject = self._extract_model_like_subject(query)
        if model_subject and not profiles:
            runtime_models = getattr(self.entity_extractor, "available_models", []) or []
            if not resolve_model_candidates(model_subject, runtime_models):
                response = (
                    f"I cannot resolve `{model_subject}` to any model in the currently loaded IAM PARIS "
                    "catalogue. I will not infer a description, scenarios, or data for an unverified name. "
                    "Ask `list models` or provide the exact catalogue name."
                )
                self.last_entities = {"unresolved_model": model_subject}
                self._record_route_decision(
                    "model_explanation", 0.98, "runtime_catalog", "unresolved model-like subject",
                )
                self.last_links = []
                return response
        requested_data_variable = preferred_variable_from_query(
            query, getattr(self.entity_extractor, "available_variables", []) or []
        )
        metadata_comparison_language = bool(re.search(
            r"\b(?:systems?|sectors?|cover(?:s|ed|ing|age)?|applications?|use\s+cases?|"
            r"frameworks?|methodology|methodologies|assumptions?|limitations?|"
            r"technolog(?:y|ies|ical)|treatment|approach|design|purpose|architecture|"
            r"differ(?:s|ed|ent|ently|ence|ences)?)\b",
            q_lower,
        ))
        qualitative_model_choice = bool(
            len(profiles) >= 2
            and re.search(
                r"\b(?:better|best|more\s+suitable|better\s+suited|prefer(?:able|red)?)\b",
                q_lower,
            )
            and re.search(r"\bmodels?\b", q_lower)
        )

        # Resolve a singular reference in a model comparison from typed
        # conversation state. Example shape: a previously discussed model,
        # followed by "How does it compare with <new model>?". Both model
        # values come from runtime/profile matching, never from literals here.
        carried_model_name = str((self.last_entities or {}).get("model") or "").strip()
        if not carried_model_name:
            prior_models = list((self.last_entities or {}).get("models") or [])
            if len(prior_models) == 1:
                carried_model_name = str(prior_models[0]).strip()
        singular_model_reference = bool(re.search(r"\b(?:it|this|that)(?:\s+model)?\b", q_lower))
        comparison_morphology = bool(re.search(
            r"\b(?:compar(?:e|ed|es|ing|ison)|differ(?:s|ed|ent|ently|ence|ences)?|versus|vs\.?)\b",
            q_lower,
        ))
        if (
            carried_model_name
            and singular_model_reference
            and comparison_morphology
            and profiles
            and not requested_data_variable
        ):
            # The carried model may come only from the live model catalogue and
            # therefore have no curated profile.  Resolve it through the same
            # runtime-backed path used for newly mentioned models so a pronoun
            # comparison never drops either side.
            carried_profiles = self._runtime_model_profiles(carried_model_name)
            combined_profiles = []
            for profile in carried_profiles + profiles:
                name = str((profile or {}).get("name") or "")
                if name and all(str(item.get("name") or "") != name for item in combined_profiles):
                    combined_profiles.append(profile)
            response = format_model_comparison_answer(combined_profiles, query)
            if response:
                names = [profile.get("name") for profile in combined_profiles]
                self._record_route_decision("model_explanation", 0.96, "conversation_state", "referenced model comparison")
                self.last_entities = {"models": names}
                return self._append_relevant_links(response, query, self.last_entities, "model_explanation")

        if (
            (not requested_data_variable or qualitative_model_choice)
            and (not _looks_like_data_request(query) or qualitative_model_choice)
            and not _looks_like_plot_request(query)
            and (
                plan.intent == "model_comparison"
                or qualitative_model_choice
                or (
                    len(profiles) >= 2
                    and re.search(
                        r"\b(?:compar(?:e|ed|es|ing|ison)|"
                        r"differ(?:s|ed|ent|ently|ence|ences)?|versus|vs\.?|between)\b",
                        q_lower,
                    )
                )
            )
        ):
            response = format_model_comparison_answer(profiles, query)
            if response:
                self._record_route_decision("model_explanation", 0.95, "query_plan", "multi-model metadata comparison")
                self.last_entities = {"models": [profile.get("name") for profile in profiles]}
                return self._append_relevant_links(response, query, self.last_entities, "model_explanation")

        # A named model plus explanatory language is a metadata question even
        # when ordinary prose overlaps a variable label (for example "useful").
        model_info_language = bool(re.search(
            r"\b(?:explain|describe|overview|framework|methodology|systems?|sectors?|coverage|"
            r"useful\s+for|used\s+(?:for|to)|intended\s+for|designed\s+(?:for|to)|"
            r"use\s+cases?|applications?|policy\s+questions?|problems?|assumptions?|limitations?|"
            r"technolog(?:y|ies|ical)|treatment|approach|design|purpose|architecture|"
            r"developers?|develop(?:ed|s|ing)?|institution|organisation|organization|"
            r"general[-\s]+equilibrium|partial[-\s]+equilibrium|cge|"
            r"model(?:ling|ing)?\s+types?|what\s+(?:kind|type|sort)\s+of\s+model|"
            r"optimi[sz](?:ation|e|ed|es|ing)|"
            r"simulation|bottom-?up|top-?down|energy\s+system\s+model)\b",
            q_lower,
        ))
        descriptive_model_coverage_question = bool(re.match(
            r"^\s*does\s+.+?\s+(?:model|cover|include|represent|simulate|capture|account\s+for)\s+"
            r"(?!data\b|results?\b|outputs?\b|values?\b|timeseries\b|time\s+series\b).+",
            q_lower,
        )) and not _looks_like_plot_request(query)
        # E3ME's curated profile is the grounded answer for the common
        # purpose question. Keep this narrow: generic model-information
        # requests still use the established model agent, while specific
        # methodology/developer/technology dimensions above can be answered
        # deterministically from structured profile fields.
        e3me_purpose_question = bool(
            len(profiles) == 1
            and model_family_key(str(profiles[0].get("name") or ""))
            == model_family_key("E3ME-FTT")
            and re.search(
                r"\bwhat\s+does\b[^?]*\b(?:e3me(?:-ftt)?\s+)?model\s+do\b",
                q_lower,
            )
        )
        if len(profiles) == 1 and (
            model_info_language or e3me_purpose_question or descriptive_model_coverage_question
        ) and not re.search(
            r"\b(?:plot|chart|table|value|values|trajectory|timeseries|time\s+series)\b", q_lower
        ):
            profile = profiles[0]
            response = format_model_profile_answer(
                profile,
                requested_name=str(profile.get("name", "")),
                asks_assumptions=bool(re.search(r"\bassumptions?\b", q_lower)),
                asks_limitations=bool(re.search(r"\b(?:limitations?|caveats?|constraints?)\b", q_lower)),
                query=query,
            )
            self._record_route_decision("model_explanation", 0.96, "query_plan", "named-model metadata request")
            self.last_entities = {"model": profile.get("name")}
            return self._append_relevant_links(response, query, self.last_entities, "model_explanation")

        early_carried = {}
        if context:
            early_carried = context.get("last_entities", {})
        if not early_carried and self.last_entities:
            early_carried = self.last_entities

        scenario_comparison_values = self._scenario_comparison_followup_values(
            original_user_query,
            early_carried,
        )

        # A model-metadata pronoun follow-up is already fully scoped by the
        # active model. Answer its catalogue dimension directly from loaded
        # timeseries records before generic variable-confidence checks can
        # misread words such as "it" and "regions" as a data variable.
        model_scope_followup = bool(
            early_carried
            and self._is_model_scope_followup(original_user_query)
            and (
                early_carried.get("model")
                or len(list(early_carried.get("models") or [])) == 1
            )
        )
        if model_scope_followup and "ts" in self.shared_resources:
            scoped_model = str(
                early_carried.get("model")
                or list(early_carried.get("models") or [""])[0]
            ).strip()
            scoped_category = _model_scoped_category(original_user_query)
            if scoped_model and scoped_category:
                response = _list_model_category(
                    scoped_category,
                    scoped_model,
                    list(self.shared_resources.get("ts") or []),
                    show_all=bool(re.search(r"\b(?:all|every)\b", original_user_query, re.IGNORECASE)),
                )
                scoped_entities = {"model": scoped_model, "action": "query"}
                self.clarification_context = None
                self._record_route_decision(
                    "data_query",
                    0.98,
                    "conversation_state",
                    "model-scoped catalogue follow-up",
                )
                self._persist_last_entities(scoped_entities, response)
                return self._append_relevant_links(
                    response,
                    original_user_query,
                    scoped_entities,
                    "data_query",
                )

        # Resolve typed plural references before broad catalogue discovery. The
        # model names come only from conversation state and are resolved against
        # the runtime timeseries catalogue by the bounded availability helper.
        referenced_models = list(early_carried.get("models") or []) if early_carried else []
        if not referenced_models and early_carried and early_carried.get("model"):
            referenced_models = [early_carried.get("model")]
        explicit_model_choice_reference = bool(re.search(
            r"\b(?:which(?:\s+one)?\s+of\s+(?:them|(?:these|those|the)"
            r"(?:\s+(?:two|both))?(?:\s+models?)?)|which\s+one|"
            r"either(?:\s+(?:one|model))?|both(?:\s+models?|\s+of\s+them)?|"
            r"(?:these|those)(?:\s+(?:two|both))?\s+models?|the\s+two\s+models?|"
            r"(?:these|those)\s+two)\b",
            original_user_query,
            re.IGNORECASE,
        ))
        state_qualitative_model_reference = bool(
            len(referenced_models) == 2
            and re.search(
                r"\b(?:they|their|them|the\s+pair)(?:selves)?\b",
                original_user_query,
                re.IGNORECASE,
            )
            and re.search(
                r"\b(?:compar(?:e|ed|es|ing|ison)|differ(?:s|ed|ent|ently|ence|ences)?|"
                r"assumptions?|limitations?|caveats?|methodolog(?:y|ies|ical)|methods?|"
                r"approaches?|model(?:ling|ing)?\s+types?|technolog(?:y|ies|ical)|"
                r"coverage|scope|sectors?|systems?|developers?|institutions?|"
                r"applications?|use\s+cases?|used\s+(?:for|to)|works?)\b",
                original_user_query,
                re.IGNORECASE,
            )
        )
        refers_to_model_choice = bool(
            explicit_model_choice_reference or state_qualitative_model_reference
        )
        explicit_plural_model_reference = bool(re.search(
            r"\b(?:both|(?:these|those)(?:\s+(?:two|both))?|the\s+two)\s+models?\b|"
            r"\bwhich(?:\s+one)?\s+of\s+(?:these|those|the)"
            r"(?:\s+(?:two|both))?\s+models?\b|"
            r"\b(?:both|either)\s+of\s+(?:the\s+)?models?\b",
            original_user_query,
            re.IGNORECASE,
        ))
        asks_model_availability = bool(re.search(
            r"\b(?:available|availability|report|reports|reported|reporting|has|have|"
            r"contain|contains|cover|covers|provide|provides|data\s+for)\b",
            original_user_query,
            re.IGNORECASE,
        ))
        available_variables = getattr(self.entity_extractor, "available_variables", []) or []
        bounded_availability_variable = (
            self._match_catalog_value_from_text(original_user_query, available_variables)
            or requested_data_variable
        )
        explicit_data_availability_language = bool(re.search(
            r"\b(?:report|reports|reported|reporting|contain|contains|dataset|data|"
            r"variable|variables|value|values|timeseries|time\s+series)\b",
            original_user_query,
            re.IGNORECASE,
        ))
        asks_bounded_data_availability = bool(
            asks_model_availability
            and (bounded_availability_variable or explicit_data_availability_language)
        )
        if (
            refers_to_model_choice
            and len(referenced_models) >= 2
            and bounded_availability_variable
            and asks_bounded_data_availability
            and str((self.clarification_context or {}).get("agent_type") or "")
            in {"data_query", "data_plotting"}
        ):
            # A fully grounded bounded-model availability request is a new
            # typed intent, not an answer to an older data clarification.
            self.clarification_context = None
        if (
            refers_to_model_choice
            and asks_bounded_data_availability
            and not getattr(self, "clarification_context", None)
        ):
            if len(referenced_models) >= 2:
                response = self._referenced_models_availability_answer(original_user_query, referenced_models)
                if response:
                    updated_scope = dict(early_carried)
                    if bounded_availability_variable:
                        updated_scope["variable"] = bounded_availability_variable
                    self.last_entities = updated_scope
                    self._record_route_decision(
                        "data_query", 0.96, "conversation_state", "referenced-model availability",
                    )
                    return self._append_relevant_links(
                        response, original_user_query, updated_scope, "data_query",
                    )
                self.last_entities = dict(early_carried)
                self.last_links = []
                self._record_route_decision(
                    "data_query", 0.92, "conversation_state", "referenced-model availability needs variable",
                )
                return (
                    "I have the referenced model set, but I could not resolve a data variable from "
                    "that request. Please name the variable you want to check."
                )
            if explicit_plural_model_reference or not referenced_models:
                self.last_links = []
                self._record_route_decision(
                    "data_query", 0.95, "conversation_state", "missing referenced model set",
                )
                return (
                    "I do not have a referenced set of two models in this conversation yet. "
                    "Name or compare the models first, then ask which of them reports the variable."
                )

        asks_link_or_read = bool(re.search(
            r"\b(?:link|links|page|pages|documentation|docs|read|browse|learn\s+more)\b",
            original_user_query,
            re.IGNORECASE,
        ))
        refers_to_plural_prior = bool(re.search(
            r"\b(?:these|those|both|the\s+two|them|their)\b(?:\s+of\s+them)?|"
            r"\b(?:these|those|both)\s+models?\b",
            original_user_query,
            re.IGNORECASE,
        ))
        asks_links_for_models = asks_link_or_read and refers_to_plural_prior
        if (
            asks_links_for_models
            and len(referenced_models) >= 2
            and str((self.clarification_context or {}).get("agent_type") or "")
            in {"data_query", "data_plotting"}
        ):
            self.clarification_context = None
        if asks_links_for_models and not getattr(self, "clarification_context", None):
            if len(referenced_models) >= 2 and self.shared_resources.get("link_catalog"):
                entities = {"models": referenced_models, "model": referenced_models[0]}
                names = ", ".join(str(name) for name in referenced_models)
                response = f"IAM PARIS model links for: {names}."
                self._record_route_decision(
                    "model_explanation", 0.96, "conversation_state", "referenced model links",
                )
                self.last_entities = dict(early_carried)
                return self._append_relevant_links(
                    response, original_user_query, entities, "model_explanation",
                )
            if explicit_plural_model_reference and len(referenced_models) < 2:
                self.last_links = []
                self._record_route_decision(
                    "model_explanation", 0.95, "conversation_state", "missing referenced model set",
                )
                return (
                    "I do not have a referenced set of two models in this conversation yet. "
                    "Name or compare the models first, then ask for their documentation links."
                )

        # A deictic qualitative question ("these two models", "which one")
        # must stay bounded to the typed model set in conversation state.  It
        # is deliberately routed after explicit data-availability and link
        # requests, but before broad catalogue discovery, whose generic
        # ``which + models`` grammar would otherwise discard the pair.
        explicit_quantitative_data_request = bool(
            _looks_like_plot_request(original_user_query)
            or re.search(
                r"\b(?:table|dataset|data|values?|rows?|observations?|timeseries|"
                r"time\s+series|trajectory|trend|year|years)\b",
                original_user_query,
                re.IGNORECASE,
            )
            or (
                bounded_availability_variable
                and re.search(
                    r"\b(?:compar(?:e|ed|es|ing|ison)|contrast|"
                    r"differ(?:s|ed|ent|ently|ence|ences)?|versus|vs\.?)\b",
                    original_user_query,
                    re.IGNORECASE,
                )
            )
        )
        typed_qualitative_model_interrupt = bool(
            refers_to_model_choice
            and len(referenced_models) >= 2
            and (
                state_qualitative_model_reference
                or metadata_comparison_language
                or _looks_like_model_info_request(original_user_query)
            )
            and not asks_bounded_data_availability
            and not asks_link_or_read
            and not explicit_quantitative_data_request
            and not self._is_site_navigation_request(original_user_query)
        )
        if (
            typed_qualitative_model_interrupt
            and str((self.clarification_context or {}).get("agent_type") or "")
            in {"data_query", "data_plotting"}
        ):
            self.clarification_context = None
        if (
            typed_qualitative_model_interrupt
            and not getattr(self, "clarification_context", None)
        ):
            bounded_profiles: List[Dict[str, Any]] = []
            unresolved_references: List[str] = []
            for reference in referenced_models:
                resolved = self._runtime_model_profiles(str(reference))
                if not resolved:
                    resolved = find_model_profiles(str(reference))
                selected = next(
                    (
                        profile for profile in resolved
                        if str(profile.get("name") or "").casefold()
                        == str(reference).casefold()
                    ),
                    resolved[0] if resolved else None,
                )
                if not selected:
                    unresolved_references.append(str(reference))
                    continue
                selected_name = str(selected.get("name") or "").casefold()
                if selected_name and all(
                    str(profile.get("name") or "").casefold() != selected_name
                    for profile in bounded_profiles
                ):
                    bounded_profiles.append(selected)

            response = format_model_comparison_answer(
                bounded_profiles,
                original_user_query,
            )
            if not response:
                unresolved_text = ", ".join(
                    f"`{name}`" for name in (unresolved_references or referenced_models)
                )
                response = (
                    "I am keeping this question scoped to the referenced models, but the "
                    "loaded catalogue does not contain enough qualitative metadata to compare "
                    f"{unresolved_text}. I will not substitute unrelated models."
                )
            updated_scope = dict(early_carried)
            updated_scope["models"] = list(referenced_models)
            updated_scope.pop("model", None)
            self.last_entities = updated_scope
            self._record_route_decision(
                "model_explanation",
                0.96,
                "conversation_state",
                "referenced-model qualitative comparison",
            )
            return self._append_relevant_links(
                response,
                original_user_query,
                updated_scope,
                "model_explanation",
            )

        if plan.intent == "availability" and any(
            target in plan.availability_targets for target in ("variable", "region", "scenario", "model")
        ):
            availability_carried = early_carried if plan.followup else {}
            metadata = self.shared_resources.get("metadata")
            available_variables = getattr(metadata, "all_variables", set()) if metadata else set()
            normalized_query = re.sub(r"\s+", " ", str(query or "").casefold()).strip()
            # An exact catalogue variable in the user's wording is authoritative.
            # Prefer its scoped availability over a broader topic-family summary;
            # the vocabulary remains entirely runtime-driven.
            explicit_variable = next(
                (
                    str(variable)
                    for variable in sorted(available_variables, key=lambda value: -len(str(value)))
                    if re.search(
                        rf"(?<![\w|]){re.escape(re.sub(r'\s+', ' ', str(variable).casefold()).strip())}(?![\w|])",
                        normalized_query,
                    )
                ),
                "",
            )

            def _availability_state(variable: str = "") -> Dict[str, Any]:
                updated = dict(availability_carried or {})
                if variable:
                    updated["variable"] = variable
                regions = getattr(self.entity_extractor, "available_regions", []) or []
                region = self._match_catalog_value_from_text(query, regions) or canonical_region_from_query(
                    query, regions,
                )
                if region:
                    updated["region"] = region
                scenarios = getattr(self.entity_extractor, "available_scenarios", []) or []
                scenario_values = explicit_scenarios_from_query(query, scenarios)
                if not scenario_values:
                    scenario = self._match_catalog_value_from_text(query, scenarios) or canonical_scenario_from_query(
                        query, scenarios,
                    )
                    scenario_values = [scenario] if scenario else []
                if len(scenario_values) > 1:
                    updated["scenarios"] = list(scenario_values)
                    updated.pop("scenario", None)
                    updated["all_scenarios"] = False
                elif scenario_values:
                    updated["scenario"] = scenario_values[0]
                    updated["scenarios"] = [scenario_values[0]]
                    updated["all_scenarios"] = False
                models = getattr(self.entity_extractor, "available_models", []) or []
                model_matches = list(build_query_plan(
                    query, available_models=models,
                ).mentioned_models)
                if len(model_matches) > 1:
                    updated["models"] = list(model_matches)
                    updated.pop("model", None)
                elif model_matches:
                    updated["model"] = str(model_matches[0])
                    updated.pop("models", None)
                year_filter = extract_year_filter(query)
                if year_filter.explicit:
                    updated = year_filter.apply(updated)
                return updated

            if "variable" in plan.availability_targets:
                response = self._scoped_variable_availability_answer(query, availability_carried)
                if response:
                    self.last_entities = _availability_state()
                    self._record_route_decision(
                        "data_query", 0.95, "query_plan", "scope-filtered variable availability",
                    )
                    return self._append_relevant_links(
                        response, query, self.last_entities, "data_query",
                    )
            if explicit_variable:
                response = self._scoped_availability_answer(
                    query, plan.availability_targets, availability_carried,
                )
                if response:
                    self.last_entities = _availability_state(explicit_variable)
                    self._record_route_decision("data_query", 0.97, "query_plan", "exact-variable scoped availability")
                    return self._append_relevant_links(response, query, self.last_entities, "data_query")
            carried_variable = str((availability_carried or {}).get("variable") or "").strip()
            if carried_variable:
                response = self._scoped_availability_answer(
                    query, plan.availability_targets, availability_carried,
                )
                if response:
                    self.last_entities = _availability_state(carried_variable)
                    self._record_route_decision(
                        "data_query", 0.96, "conversation_state", "scoped availability request",
                    )
                    return self._append_relevant_links(
                        response, query, self.last_entities, "data_query",
                    )
            # A broad topic + model availability request is an aggregation over
            # every matching variable, not availability for whichever leaf the
            # extractor happened to choose first.
            if "model" in plan.availability_targets:
                topic_models = self._models_covering_topic_answer(query)
                if topic_models is not None:
                    self._record_route_decision("data_query", 0.95, "query_plan", "topic-wide model availability")
                    self.last_entities = {}
                    return self._append_relevant_links(topic_models, query, {}, "data_query")
            response = self._scoped_availability_answer(
                query, plan.availability_targets, availability_carried,
            )
            if response:
                self.last_entities = _availability_state(explicit_variable)
                self._record_route_decision("data_query", 0.94, "query_plan", "scoped availability request")
                return self._append_relevant_links(response, query, self.last_entities, "data_query")

        if hasattr(self, "clarification_context") and self.clarification_context:
            issued_turn = int((self.clarification_context or {}).get("issued_turn", getattr(self, "current_turn", 0)))
            # Keep the pending choice alive for a few turns so an intervening
            # message does not silently discard the user's next "2"/"yes".
            if getattr(self, "current_turn", 0) > issued_turn + 3:
                self.clarification_context = None
            elif not (
                plan.replacement_dimension
                and plan.replacement_value
            ) and not self._is_clarification_followup(query, self.clarification_context):
                self.clarification_context = None

        failed_comparison = self._failed_scope_comparison_answer(original_user_query)
        if failed_comparison:
            self._record_route_decision(
                "data_query", 0.98, "conversation_state", "comparison includes failed attempted scope",
            )
            self.last_links = []
            return failed_comparison

        result_dimension = self._result_scope_dimension(original_user_query)
        if result_dimension:
            scope_answer = self._result_scope_answer(result_dimension)
            if scope_answer:
                self._record_route_decision(
                    "data_query", 0.98, "conversation_state", "latest-result scope follow-up",
                )
                self.last_links = []
                return scope_answer

        # Provenance follow-up ("which model is this from?"): answer from the
        # models of the last data result instead of routing to a model listing.
        if self._is_result_provenance_question(query):
            provenance = self._result_provenance_answer()
            if provenance is not None:
                self._record_route_decision("data_query", 0.9, "deterministic", "result provenance follow-up")
                self.last_links = []
                return provenance

        prior_comparison_query = ""
        if (
            early_carried
            and plan.output_mode == "plot"
            and self._is_generic_followup(original_user_query)
            and bool(plan.diagnostics.get("comparison_language"))
            and not early_carried.get("comparison")
            and not scenario_comparison_values
        ):
            prior_comparison_query = self._render_previous_scope_comparison(early_carried)

        model_scope_followup = bool(
            early_carried
            and self._is_model_scope_followup(original_user_query)
            and (
                early_carried.get("model")
                or len(list(early_carried.get("models") or [])) == 1
            )
        )
        scope_patch = bool(model_scope_followup or (
            early_carried and plan.followup and (
                plan.replacement_dimension or plan.output_mode or plan.year_filter.explicit
            )
        ))
        if prior_comparison_query:
            query = prior_comparison_query
            q_lower = query.lower()
            scope_patch = False
        if model_scope_followup:
            query = self._compose_contextual_query(original_user_query, early_carried)
            q_lower = query.lower()
        if scope_patch:
            if not model_scope_followup:
                query = render_scope_query(early_carried, plan)
                q_lower = query.lower()
        # A scenario-comparison follow-up ("compare with net zero") pairs the
        # carried scenario with the newly named one. Route it with structured
        # entities (families expanded to their member codes) so the new scenario
        # is never re-parsed as a region by the text path ("net zero" -> "RO").
        if scenario_comparison_values and early_carried:
            comp_variable = str(early_carried.get("variable") or "").strip()
            if comp_variable:
                available_scenarios = getattr(self.entity_extractor, "available_scenarios", []) or []
                expanded: List[str] = []
                for family in scenario_comparison_values:
                    members = scenario_family_members(family, available_scenarios)
                    if not members:
                        members = [family] if family in set(available_scenarios) else [family]
                    for member in members:
                        if member not in expanded:
                            expanded.append(member)
                comp_entities: Dict[str, Any] = {
                    "action": "query",
                    "variable": comp_variable,
                    "scenarios": expanded,
                    "comparison": "scenario",
                    "all_scenarios": False,
                }
                comp_region = str(early_carried.get("region") or "").strip()
                if comp_region:
                    comp_entities["region"] = comp_region
                comp_model = str(early_carried.get("model") or "").strip()
                if comp_model:
                    comp_entities["model"] = comp_model
                comp_workspace = str(early_carried.get("workspace_code") or "").strip()
                if comp_workspace:
                    comp_entities["workspace_code"] = comp_workspace
                for year_key in ("start_year", "end_year"):
                    if early_carried.get(year_key) is not None:
                        comp_entities[year_key] = early_carried[year_key]
                follow_year_filter = extract_year_filter(original_user_query)
                if follow_year_filter.explicit:
                    comp_entities = follow_year_filter.apply(comp_entities)
                agent = self.agents.get("data_query")
                if agent and hasattr(agent, "handle_with_entities"):
                    self._record_route_decision(
                        "data_query", 0.95, "conversation_state", "scenario-comparison table follow-up",
                    )
                    response = agent.handle_with_entities(original_user_query, comp_entities, history)
                    self._persist_last_entities(comp_entities, response)
                    return self._append_relevant_links(
                        response, original_user_query, comp_entities, "data_query",
                    )

        was_contextual_followup = bool(early_carried and self._is_contextual_dimension_followup(query))
        if was_contextual_followup:
            query = self._compose_contextual_query(query, early_carried)
            q_lower = query.strip().lower()

        # "plot both models together" / "show both on a chart": a plot request
        # that references both the previous and the current scope without naming
        # anything new. Build the two-series comparison from the last two
        # successful answers when they differ in exactly one dimension.
        # Structured entities are handed straight to the plotting agent so
        # exact catalogue names survive. The phrasing must be fully generic —
        # a query that names its own scope (e.g. "plot all scenarios for CO2")
        # is not a previous-scope reference.
        _both_plot_followup = bool(re.fullmatch(
            r"(?i)(?:plot|chart|graph|draw|show|display|visuali[sz]e)\s+(?:me\s+)?(?:the\s+)?"
            r"(?:both|two|all)\b(?:\s+(?:of\s+them|models|variables|regions|scenarios|series))?"
            r"(?:\s+together)?(?:\s+(?:on|in)\s+(?:a|one|the\s+same)\s+(?:chart|plot|graph|figure))?",
            re.sub(self._FOLLOWUP_FILLER, "", original_user_query.strip(), flags=re.IGNORECASE).strip().rstrip("?.!").strip(),
        ))
        if (
            not was_contextual_followup
            and early_carried
            and not early_carried.get("comparison")
            and _both_plot_followup
        ):
            comparison_entities = self._previous_scope_comparison_entities(early_carried)
            if comparison_entities:
                agent = self.agents.get("data_plotting")
                if agent and hasattr(agent, "handle_with_entities"):
                    self._record_route_decision(
                        "data_plotting", 0.95, "conversation_state", "previous-scope comparison plot",
                    )
                    response = agent.handle_with_entities(
                        original_user_query, comparison_entities, history,
                    )
                    self._persist_last_entities(comparison_entities, response)
                    return self._append_relevant_links(
                        response, original_user_query, comparison_entities, "data_plotting",
                    )
            comparison_query = self._render_previous_scope_comparison(early_carried)
            if comparison_query:
                query = comparison_query
                q_lower = query.lower()

        explicit_navigation = self._is_site_navigation_request(query)
        if explicit_navigation and self.shared_resources.get("link_catalog"):
            self._record_route_decision("general_qa", 0.97, "deterministic", "explicit site navigation request")
            self._conversation().response_scope_override = {}
            return self._grounded_site_navigation_answer(query, {})

        if not scope_patch and _looks_like_category_list_request(query, "variables") and _looks_like_plot_request(query):
            agent = self.agents.get("data_query")
            if not agent:
                return "Sorry, the requested agent is not available."
            self.last_entities = {}
            self._record_route_decision("data_query", 0.95, "deterministic", "plot variable discovery request")
            response = agent.handle(query, history)
            return self._append_relevant_links(response, query, {}, "data_query")
        if not scope_patch and _looks_like_category_list_request(query, "models") and not was_contextual_followup:
            self.last_entities = {}
            sector_answer = self._models_covering_topic_answer(query)
            if sector_answer is not None:
                self._record_route_decision("data_query", 0.92, "deterministic", "sector-filtered model availability request")
                return self._append_relevant_links(sector_answer, query, {}, "data_query")
            agent = self.agents.get("data_query")
            if not agent:
                return "Sorry, the requested agent is not available."
            self._record_route_decision("data_query", 0.95, "deterministic", "model availability request")
            response = agent.handle(query, history)
            return self._append_relevant_links(response, query, {}, "data_query")
        if (
            _looks_like_category_list_request(query, "scenarios")
            and not scope_patch
            and not re.search(r"\b(?:for|across)\s+(?:all\s+)?available\s+scenarios\b", query, re.IGNORECASE)
        ):
            agent = self.agents.get("data_query")
            if not agent:
                return "Sorry, the requested agent is not available."
            self.last_entities = {}
            self._record_route_decision("data_query", 0.95, "deterministic", "scenario availability request")
            response = agent.handle(query, history)
            response = self._maybe_add_followup_guidance(response, query, "data_query")
            return self._append_relevant_links(response, query, {}, "data_query")
        if not scope_patch and _looks_like_category_list_request(query, "variables"):
            agent = self.agents.get("data_query")
            if not agent:
                return "Sorry, the requested agent is not available."
            self.last_entities = {}
            self._record_route_decision("data_query", 0.95, "deterministic", "variable availability request")
            response = agent.handle(query, history)
            return self._append_relevant_links(response, query, {}, "data_query")
        if not scope_patch and _looks_like_category_list_request(query, "regions"):
            agent = self.agents.get("data_query")
            if not agent:
                return "Sorry, the requested agent is not available."
            self.last_entities = {}
            self._record_route_decision("data_query", 0.95, "deterministic", "region availability request")
            response = agent.handle(query, history)
            return self._append_relevant_links(response, query, {}, "data_query")

        fresh_interrupt = bool(
            self._is_site_navigation_request(query)
            or (
                _looks_like_model_info_request(query)
                and (find_model_profile(query) or self._mentions_known_model(query) or re.search(r"\bmodel\b", query, flags=re.IGNORECASE))
            )
        )
        if fresh_interrupt:
            self.clarification_context = None

        # Check for clarification responses first
        if hasattr(self, 'clarification_context') and self.clarification_context:
            clar_ctx = self.clarification_context
            pending_kind = str(clar_ctx.get("suggested_kind", "") or "").strip().lower()
            pending_catalog = {
                "variable": getattr(self.entity_extractor, "available_variables", []) or [],
                "region": getattr(self.entity_extractor, "available_regions", []) or [],
                "scenario": getattr(self.entity_extractor, "available_scenarios", []) or [],
                "model": getattr(self.entity_extractor, "available_models", []) or [],
            }.get(pending_kind, [])
            exact_pending_value = self._match_catalog_value_from_text(query, pending_catalog)
            if exact_pending_value:
                entities = dict(clar_ctx.get("entities", {}) or {})
                entities[pending_kind] = exact_pending_value
                clar_ctx["entities"] = entities
                clar_ctx[f"suggested_{pending_kind}"] = exact_pending_value
                query = "yes"
            option_choice_idx = self._extract_option_choice(
                query,
                len(clar_ctx.get("suggested_options", []) or []),
            )
            if option_choice_idx is None:
                option_choice_idx = self._extract_named_option_choice(
                    query,
                    list(clar_ctx.get("suggested_options", []) or []),
                )
            if option_choice_idx is not None:
                if self._apply_clarification_option(clar_ctx, option_choice_idx):
                    query = "yes"
            if self._is_affirmation(query):
                pending_type = clar_ctx.get("agent_type", "")
                original_query = str(clar_ctx.get("original_query", "")).strip()
                base_query = str(clar_ctx.get("base_query", "") or original_query).strip()
                selected_kind = str(clar_ctx.get("suggested_kind", "variable") or "variable").strip().lower()
                suggested_variable = str(clar_ctx.get("suggested_variable", "")).strip()
                suggested_region = str(clar_ctx.get("suggested_region", "")).strip()
                suggested_scenario = str(clar_ctx.get("suggested_scenario", "")).strip()
                suggested_model = str(clar_ctx.get("suggested_model", "")).strip()
                merged_entities = dict(clar_ctx.get("entities", {}) or {})
                if suggested_variable:
                    merged_entities["variable"] = suggested_variable
                if suggested_region:
                    merged_entities["region"] = suggested_region
                if suggested_scenario:
                    merged_entities["scenario"] = suggested_scenario
                if suggested_model:
                    merged_entities["model"] = suggested_model
                selected_value = str(merged_entities.get(selected_kind) or "").strip()
                merged_entities = self._finalize_clarification_entities(
                    merged_entities,
                    selected_kind,
                    selected_value,
                    confirmed=bool(selected_value),
                )
                followup_query = base_query or original_query or query
                self.clarification_context = None
                if pending_type == "data_plotting":
                    agent = self.agents.get("data_plotting")
                    if not agent:
                        return "Sorry, the requested agent is not available."
                    if hasattr(agent, "handle_with_entities"):
                        response = agent.handle_with_entities(followup_query, merged_entities, history)
                    else:
                        response = agent.handle(followup_query, history)
                    if not str(response or "").strip():
                        response = "I need one more detail to continue. Please specify the variable, region, or scenario."
                    self._record_clarification_result_route("data_plotting", response)
                    self._persist_last_entities(merged_entities, response)
                    self._update_clarification_context("data_plotting", followup_query, response, merged_entities, base_query=base_query)
                    return self._append_relevant_links(response, followup_query, merged_entities, "data_plotting")
                if pending_type == "data_query":
                    agent = self.agents.get("data_query")
                    if not agent:
                        return "Sorry, the requested agent is not available."
                    if hasattr(agent, "handle_with_entities"):
                        response = agent.handle_with_entities(followup_query, merged_entities, history)
                    else:
                        response = agent.handle(followup_query, history)
                    if not str(response or "").strip():
                        response = "I need one more detail to continue. Please specify the variable, region, or scenario."
                    self._record_clarification_result_route("data_query", response)
                    self._persist_last_entities(merged_entities, response)
                    self._update_clarification_context("data_query", followup_query, response, merged_entities, base_query=base_query)
                    return self._append_relevant_links(response, followup_query, merged_entities, "data_query")
                if pending_type == "model_explanation" and suggested_model:
                    profiles = self._runtime_model_profiles(suggested_model) or find_model_profiles(suggested_model)
                    if profiles:
                        profile = profiles[0]
                        response = format_model_profile_answer(
                            profile,
                            requested_name=suggested_model,
                            query=base_query or original_query,
                        )
                        self.last_entities = {"model": suggested_model}
                        self._record_route_decision(
                            "model_explanation", 0.98, "conversation_state", "model-family clarification resolved",
                        )
                        return self._append_relevant_links(
                            response,
                            base_query or original_query,
                            self.last_entities,
                            "model_explanation",
                        )
                return self._route_single(followup_query, history, context={"last_entities": merged_entities})

            if self._is_rejection(query):
                pending_type = clar_ctx.get("agent_type", "")
                options = list(clar_ctx.get("suggested_options", []) or [])
                option_kinds = list(clar_ctx.get("suggested_option_kinds", []) or [])
                used_kind = str(clar_ctx.get("suggested_kind", "variable") or "variable")
                typed_options = [
                    (
                        str(option_kinds[index] if index < len(option_kinds) else used_kind),
                        str(value),
                    )
                    for index, value in enumerate(options)
                ]
                selected_index = int(clar_ctx.get("selected_option_index", 0) or 0)
                remaining_choices = [
                    choice for index, choice in enumerate(typed_options)
                    if index != selected_index
                ]

                self.clarification_context = None
                if pending_type == "data_query" and remaining_choices:
                    response = self._clarification_options_prompt(
                        "Okay, here are the next closest options.",
                        remaining_choices[:3],
                    )
                    updated_entities = self._finalize_clarification_entities(
                        clar_ctx.get("entities", {}) or {},
                    )
                    self._record_clarification_result_route(
                        "data_query", response, "clarification rejection",
                    )
                    self._update_clarification_context(
                        "data_query",
                        str(clar_ctx.get("base_query", "") or clar_ctx.get("original_query", "") or ""),
                        response,
                        updated_entities,
                        base_query=str(clar_ctx.get("base_query", "") or clar_ctx.get("original_query", "") or ""),
                    )
                    return response
                if pending_type == "data_query":
                    self._record_route_decision(
                        "data_query", 0.72, "conversation_state", "clarification rejected",
                    )
                    return "Okay. Which variable should I use instead?"
                if pending_type == "data_plotting":
                    self._record_route_decision(
                        "data_plotting", 0.72, "conversation_state", "clarification rejected",
                    )
                    return "Okay. Which variable or region should I use instead?"
                self._record_route_decision(
                    "data_query", 0.72, "conversation_state", "clarification rejected",
                )
                return "Okay. Please give me the variable you want."

            # A grounded scope patch can arrive while another dimension still
            # needs clarification. Update only the explicitly named dimension
            # and keep the pending choice alive; inferred entities from the
            # patch wording must not replace the unresolved variable/region.
            patch_dimension = str(plan.replacement_dimension or "").strip().lower()
            patch_value = str(plan.replacement_value or "").strip()
            if patch_dimension in {"variable", "region", "scenario", "model"} and patch_value:
                raw_patch_value = patch_value
                catalog_values: List[str] = []
                resolved_patch_value = ""
                if patch_dimension == "region":
                    catalog_values = list(
                        getattr(self.entity_extractor, "available_regions", []) or []
                    )
                    resolved_patch_value = (
                        self._resolve_region_from_text(patch_value, clar_ctx.get("entities", {}))
                        or canonical_region_from_query(
                            patch_value,
                            catalog_values,
                        )
                    )
                elif patch_dimension == "scenario":
                    catalog_values = list(
                        getattr(self.entity_extractor, "available_scenarios", []) or []
                    )
                    resolved_patch_value = (
                        self._match_catalog_value_from_text(patch_value, catalog_values)
                        or canonical_scenario_from_query(patch_value, catalog_values)
                        or self._match_scenario_from_text(patch_value)
                    )
                elif patch_dimension == "variable":
                    catalog_values = list(
                        getattr(self.entity_extractor, "available_variables", []) or []
                    )
                    resolved_patch_value = (
                        self._match_catalog_value_from_text(patch_value, catalog_values)
                        or preferred_variable_from_query(patch_value, catalog_values)
                    )
                elif patch_dimension == "model":
                    catalog_values = list(
                        getattr(self.entity_extractor, "available_models", []) or []
                    )
                    candidates = resolve_model_candidates(
                        patch_value,
                        catalog_values,
                    )
                    if candidates:
                        resolved_patch_value = str(candidates[0])

                if catalog_values and not resolved_patch_value:
                    pending_prompt = str(
                        clar_ctx.get("response")
                        or clar_ctx.get("ambiguous_response")
                        or "I still need the pending choice before I can continue."
                    )
                    response = (
                        f"I could not resolve `{raw_patch_value}` as an available "
                        f"{patch_dimension}, so I kept the existing scope.\n\n"
                        f"{pending_prompt}"
                    )
                    self._record_route_decision(
                        str(clar_ctx.get("agent_type") or "data_query"),
                        0.9,
                        "conversation_state",
                        "invalid clarification scope patch rejected",
                    )
                    self.last_links = []
                    return response
                patch_value = resolved_patch_value or raw_patch_value

                pending_kind = str(clar_ctx.get("suggested_kind", "") or "").strip().lower()
                patched_entities = dict(clar_ctx.get("entities", {}) or {})
                patched_entities[patch_dimension] = patch_value
                plural_key = {
                    "variable": "variables", "region": "regions",
                    "scenario": "scenarios", "model": "models",
                }[patch_dimension]
                patched_entities.pop(plural_key, None)
                if patch_dimension == "scenario":
                    patched_entities["all_scenarios"] = False
                    clar_ctx["suggested_scenario"] = patch_value
                elif patch_dimension == "region":
                    clar_ctx["suggested_region"] = patch_value
                elif patch_dimension == "variable":
                    clar_ctx["suggested_variable"] = patch_value
                confidence = dict(patched_entities.get("entity_confidence") or {})
                confidence[patch_dimension] = 1.0
                patched_entities["entity_confidence"] = confidence
                clar_ctx["entities"] = patched_entities
                clar_ctx["issued_turn"] = getattr(self, "current_turn", 0)

                # A valid explicit patch of the pending dimension is itself a
                # grounded clarification answer. Resolve it immediately rather
                # than leaving the preselected first suggestion in control.
                if patch_dimension == pending_kind:
                    return self._route_single("yes", history, context)

                if patch_dimension != pending_kind:
                    pending_prompt = str(
                        clar_ctx.get("response")
                        or clar_ctx.get("ambiguous_response")
                        or "I still need the pending choice before I can continue."
                    )
                    response = (
                        f"Updated {patch_dimension} to `{patch_value}`. "
                        "I still need the pending choice.\n\n"
                        f"{pending_prompt}"
                    )
                    self._record_route_decision(
                        str(clar_ctx.get("agent_type") or "data_query"),
                        0.9,
                        "conversation_state",
                        "clarification scope patch preserved pending choice",
                    )
                    self.last_links = []
                    return response

            # An action-only contextual reply does not answer a pending entity
            # question.  In particular, do not pass words such as "that" back
            # through fuzzy region matching, where a short catalogue code could
            # be invented from ordinary grammar.  Keep the pending prompt alive
            # until the user supplies a typed option or a grounded dimension.
            if self._is_generic_followup(query) and _looks_like_plot_request(query):
                pending_type = str(clar_ctx.get("agent_type", "data_query") or "data_query")
                response = str(
                    clar_ctx.get("response")
                    or clar_ctx.get("ambiguous_response")
                    or "I still need the pending variable, region, or scenario before I can plot it."
                )
                self._record_route_decision(
                    pending_type if pending_type in VALID_AGENT_NAMES else "data_query",
                    0.72,
                    "conversation_state",
                    "plot deferred until clarification is resolved",
                )
                self.last_links = []
                return response

            # Treat non-yes/no follow-up text as clarification details to merge with clar_ctx.
            pending_type = clar_ctx.get("agent_type", "")
            original_query = str(clar_ctx.get("original_query", "")).strip()
            base_query = str(clar_ctx.get("base_query", "") or original_query).strip()
            merged_entities = dict(clar_ctx.get("entities", {}) or {})
            try:
                follow_entities = self.entity_extractor.extract(query)
            except Exception:
                follow_entities = {}

            explicitly_supplied_kinds: List[str] = []
            for key in ("variable", "region", "scenario", "model"):
                value = str((follow_entities or {}).get(key, "") or "").strip()
                if value:
                    merged_entities[key] = value
                    explicitly_supplied_kinds.append(key)

            # Scenario shorthand fallback (e.g., "pr wwh cp")
            if not merged_entities.get("scenario"):
                scen = self._match_scenario_from_text(query)
                if scen:
                    merged_entities["scenario"] = scen
                    if "scenario" not in explicitly_supplied_kinds:
                        explicitly_supplied_kinds.append("scenario")

            selected_kind = str(clar_ctx.get("suggested_kind", "") or "").strip().lower()
            if len(explicitly_supplied_kinds) == 1:
                selected_kind = explicitly_supplied_kinds[0]
            selected_value = str(merged_entities.get(selected_kind) or "").strip()
            merged_entities = self._finalize_clarification_entities(
                merged_entities,
                selected_kind,
                selected_value,
                confirmed=bool(selected_kind and selected_value),
            )

            followup_query = base_query or original_query or query
            self.clarification_context = None
            if pending_type == "data_plotting":
                agent = self.agents.get("data_plotting")
                if not agent:
                    return "Sorry, the requested agent is not available."
                if hasattr(agent, "handle_with_entities"):
                    response = agent.handle_with_entities(followup_query, merged_entities, history)
                else:
                    response = agent.handle(followup_query, history)
                if not str(response or "").strip():
                    response = "I need one more detail to continue. Please specify the variable, region, or scenario."
                self._record_clarification_result_route(
                    "data_plotting", response, "clarification detail",
                )
                self._persist_last_entities(merged_entities, response)
                self._update_clarification_context("data_plotting", followup_query, response, merged_entities, base_query=base_query)
                return self._append_relevant_links(response, followup_query, merged_entities, "data_plotting")
            if pending_type == "data_query":
                agent = self.agents.get("data_query")
                if not agent:
                    return "Sorry, the requested agent is not available."
                if hasattr(agent, "handle_with_entities"):
                    response = agent.handle_with_entities(followup_query, merged_entities, history)
                else:
                    response = agent.handle(followup_query, history)
                if not str(response or "").strip():
                    response = "I need one more detail to continue. Please specify the variable, region, or scenario."
                self._record_clarification_result_route(
                    "data_query", response, "clarification detail",
                )
                self._persist_last_entities(merged_entities, response)
                self._update_clarification_context("data_query", followup_query, response, merged_entities, base_query=base_query)
                return self._append_relevant_links(response, followup_query, merged_entities, "data_query")

        if re.fullmatch(r"\s*\d+\s*", query):
            self._record_route_decision("data_query", 0.75, "deterministic", "number without active clarification")
            response = (
                "I don't have an active numbered choice right now. "
                "Reply with a full variable, region, or scenario so I can continue."
            )
            return self._append_relevant_links(response, query, {}, "data_query")

        if re.fullmatch(r"\s*use\s+the\s+(first|second|third|fourth|fifth)\s+scenario\s*", query, flags=re.IGNORECASE):
            self._record_route_decision("data_query", 0.75, "deterministic", "scenario ordinal without active clarification")
            response = (
                "I don't have an active scenario choice right now. "
                "Reply with a scenario name or ask me to list scenarios first."
            )
            return self._append_relevant_links(response, query, {}, "data_query")

        carried = {}
        if context:
            carried = context.get("last_entities", {})
        if not carried and self.last_entities:
            carried = self.last_entities

        named_workspace = _matched_workspace(
            original_user_query,
            _workspace_entries(list(self.shared_resources.get("ts") or [])),
        )
        carried_workspace_code = str(carried.get("workspace_code") or "").strip()
        explicit_study_reference = bool(re.search(
            r"\b(?:this|that|the)\s+(?:study|workspace)\b", original_user_query, re.IGNORECASE,
        ))
        # Once a user has opened a study, subsequent numeric data questions
        # remain within it by default. A different named study is an explicit
        # replacement and must not inherit the prior workspace filter.
        workspace_study_followup = bool(
            carried_workspace_code
            and (
                explicit_study_reference
                or (
                    _looks_like_data_request(original_user_query)
                    and (
                        not named_workspace
                        or str(named_workspace.get("code") or "") == carried_workspace_code
                    )
                )
            )
        )
        generic_followup = (
            was_contextual_followup
            or plan.followup
            or self._is_generic_followup(query)
            or self._is_contextual_dimension_followup(query)
            or self._is_model_scope_followup(query)
            or bool(scenario_comparison_values)
            or workspace_study_followup
        )

        # Carry context into generic follow-ups like "plot it" or "show me data".
        if generic_followup and carried and not was_contextual_followup and not scope_patch:
            query = self._compose_contextual_query(query, carried)
            q_lower = query.strip().lower()

        # Extract entities from query using the new extractor
        try:
            entities = self.entity_extractor.extract(query)
            self.logger.debug(f"Extracted entities: {entities}")

            # A study named in the current message is an explicit scope
            # replacement and must override a workspace carried from history.
            if named_workspace:
                entities = dict(entities or {})
                entities["workspace_code"] = str(named_workspace.get("code") or "").strip()

            if plan.replacement_dimension and plan.replacement_value:
                entities = dict(entities or {})
                replacement = plan.replacement_value
                if plan.replacement_dimension == "region":
                    replacement = canonical_region_from_query(
                        replacement, getattr(self.entity_extractor, "available_regions", []) or []
                    ) or replacement
                elif plan.replacement_dimension == "scenario":
                    available_scenarios = getattr(self.entity_extractor, "available_scenarios", []) or []
                    replacement = (
                        canonical_scenario_from_query(replacement, available_scenarios)
                        or self._match_scenario_from_text(replacement)
                        or replacement
                    )
                elif plan.replacement_dimension == "variable":
                    replacement = preferred_variable_from_query(
                        replacement, getattr(self.entity_extractor, "available_variables", []) or []
                    ) or replacement
                elif plan.replacement_dimension == "model":
                    runtime_models = getattr(self.entity_extractor, "available_models", []) or []
                    model_matches = resolve_model_candidates(replacement, runtime_models)
                    if model_matches:
                        replacement = str(model_matches[0])
                entities[plan.replacement_dimension] = replacement
                # A singular dimension replacement invalidates any stale
                # plural selection for that same dimension.
                plural_key = {"scenario": "scenarios", "model": "models", "variable": "variables"}.get(
                    plan.replacement_dimension
                )
                if plural_key:
                    entities.pop(plural_key, None)
                if plan.replacement_dimension == "scenario":
                    entities["all_scenarios"] = False

            if generic_followup and carried:
                entities = dict(entities or {})
                confidence = dict(entities.get("entity_confidence") or {})
                # Patch semantics: an extracted value may replace carried state
                # only when the original follow-up explicitly supplied that
                # dimension. This prevents a family alias inferred from words
                # such as "same data" replacing an exact prior catalog value.
                original_variables = getattr(self.entity_extractor, "available_variables", []) or []
                original_variable = self._match_catalog_value_from_text(
                    original_user_query, original_variables
                ) or preferred_variable_from_query(original_user_query, original_variables)
                # Within an explicitly referenced study, plain-language
                # "emissions" is a request for the standard CO2 indicator.
                # This makes the common question "global emissions ... in this
                # study for 2050" deterministic without widening the study.
                if workspace_study_followup and not original_variable and re.search(
                    r"\bemissions?\b", original_user_query, re.IGNORECASE,
                ):
                    gas_target = "Emissions|CO2"
                    if re.search(r"\b(?:methane|ch4)\b", original_user_query, re.IGNORECASE):
                        gas_target = "Emissions|CH4"
                    elif re.search(r"\b(?:nitrous\s+oxide|n2o)\b", original_user_query, re.IGNORECASE):
                        gas_target = "Emissions|N2O"
                    original_variable = next(
                        (
                            str(value) for value in original_variables
                            if str(value).casefold() == gas_target.casefold()
                        ),
                        "",
                    )
                original_regions = getattr(self.entity_extractor, "available_regions", []) or []
                original_region = self._resolve_region_from_text(original_user_query, carried)
                original_scenarios = getattr(self.entity_extractor, "available_scenarios", []) or []
                original_scenario = self._match_catalog_value_from_text(
                    original_user_query, original_scenarios
                ) or canonical_scenario_from_query(original_user_query, original_scenarios)
                original_models = find_model_profiles(original_user_query)
                # A "from <model>" follow-up ("now the same from POLES") is an
                # explicit model switch even when the model is not in the profile
                # catalogue. Without this, the carried model would overwrite the
                # freshly switched one below.
                followup_switch_model = self._from_switch_model(
                    re.sub(self._FOLLOWUP_FILLER, "", original_user_query.strip().lower())
                    .strip().rstrip("?.!").strip()
                )
                explicitly_replaced = {
                    "variable": bool(original_variable) or plan.replacement_dimension == "variable",
                    "region": bool(original_region) or plan.replacement_dimension == "region",
                    "scenario": bool(original_scenario) or plan.replacement_dimension == "scenario",
                    "model": bool(original_models) or bool(followup_switch_model) or plan.replacement_dimension == "model",
                }
                for key in ("variable", "region", "scenario", "model"):
                    value = str(carried.get(key, "") or "").strip()
                    if value and not explicitly_replaced[key]:
                        entities[key] = value
                        confidence[key] = max(float(confidence.get(key, 0) or 0), 0.85)
                if followup_switch_model:
                    entities["model"] = followup_switch_model
                    entities.pop("models", None)
                    confidence["model"] = max(float(confidence.get("model", 0) or 0), 0.9)

                # Do not accept an entity inferred only from the enriched
                # follow-up when that dimension was absent from both the user's
                # patch and the carried scope. This blocks fuzzy extractor drift
                # (for example a scenario phrase being mistaken for a model).
                for key in ("variable", "region", "scenario", "model"):
                    if not explicitly_replaced[key] and not carried.get(key):
                        entities.pop(key, None)
                        confidence.pop(key, None)

                # Values explicitly present in the original follow-up take
                # precedence over carried scope and extraction from the
                # enriched query. Runtime catalogs resolve the values, so this
                # remains independent of any particular region or query.
                if original_variable:
                    entities["variable"] = original_variable
                if original_region:
                    entities["region"] = original_region
                if original_scenario:
                    entities["scenario"] = original_scenario
                for key in (
                    "start_year", "end_year", "all_scenarios", "scenarios", "models",
                    "variables", "regions", "chart_type", "workspace_code",
                ):
                    if key == "scenarios" and explicitly_replaced["scenario"]:
                        continue
                    if key == "models" and explicitly_replaced["model"]:
                        continue
                    if key == "variables" and explicitly_replaced["variable"]:
                        continue
                    if key == "regions" and explicitly_replaced["region"]:
                        continue
                    if key in carried and not entities.get(key):
                        entities[key] = carried[key]

                # Plural selections are an explicit bounded scope, not an
                # extractor suggestion. A patch to another dimension (for
                # example changing only the region) must preserve those exact
                # values even if the enriched query yields a catalogue-wide
                # list. False is meaningful for all_scenarios and must also
                # overwrite an inferred True value.
                if carried.get("scenarios") and not explicitly_replaced["scenario"]:
                    carried_scenarios = list(dict.fromkeys(
                        str(value).strip()
                        for value in (carried.get("scenarios") or [])
                        if str(value or "").strip()
                    ))
                    if len(carried_scenarios) == 1:
                        # Data-query filtering consumes the singular scenario;
                        # a one-item plural is not a comparison and must not
                        # erase that exact filter during a region/variable patch.
                        entities["scenario"] = carried_scenarios[0]
                        entities.pop("scenarios", None)
                    else:
                        entities["scenarios"] = carried_scenarios
                        entities.pop("scenario", None)
                    entities["all_scenarios"] = bool(carried.get("all_scenarios", False))
                if carried.get("models") and not explicitly_replaced["model"]:
                    entities["models"] = list(carried.get("models") or [])
                    entities.pop("model", None)
                if carried.get("variables") and not explicitly_replaced["variable"]:
                    entities["variables"] = list(carried.get("variables") or [])
                if carried.get("regions") and not explicitly_replaced["region"]:
                    carried_regions = list(dict.fromkeys(
                        str(value).strip()
                        for value in (carried.get("regions") or [])
                        if str(value or "").strip()
                    ))
                    if len(carried_regions) == 1:
                        # The plotter treats `regions` as a comparison only for
                        # two or more values. Preserve a one-region plot as the
                        # singular hard filter instead of widening it globally.
                        entities["region"] = carried_regions[0]
                        entities.pop("regions", None)
                    else:
                        entities["regions"] = carried_regions
                        entities.pop("region", None)

                # A direct reference to the existing comparison (for example
                # "plot the comparison") preserves its structured dimension
                # and bounded values.  The normalized query text is allowed to
                # omit those fields, but it must not widen them to the catalogue.
                carried_comparison = carried.get("comparison")
                comparison_dimension = (
                    str(carried_comparison.get("dimension") or "")
                    if isinstance(carried_comparison, dict)
                    else str(carried_comparison or "")
                )
                if (
                    carried_comparison
                    and not explicitly_replaced.get(comparison_dimension, False)
                    and (
                        bool(plan.diagnostics.get("comparison_language"))
                        or self._is_generic_followup(original_user_query)
                    )
                ):
                    entities["comparison"] = carried["comparison"]
                    if not plan.output_mode and carried.get("action") in {"plot", "query"}:
                        entities["action"] = carried["action"]

                # A comparison follow-up naming one new scenario is a patch to
                # the carried singular scenario, not a request to compare
                # variables.  Both values are resolved from the runtime
                # scenario catalogue and retained structurally for plotting.
                if scenario_comparison_values:
                    entities["scenarios"] = list(scenario_comparison_values)
                    entities.pop("scenario", None)
                    entities["comparison"] = "scenario"
                    entities["all_scenarios"] = False
                    entities["action"] = "query"
                    confidence["scenario"] = max(
                        float(confidence.get("scenario", 0) or 0),
                        0.9,
                    )
                    confidence["comparison"] = max(
                        float(confidence.get("comparison", 0) or 0),
                        0.9,
                    )

                # Explicit years in the follow-up are a patch and therefore
                # always override carried years, independent of extractor output.
                if plan.year_filter.explicit:
                    entities = plan.year_filter.apply(entities)
                else:
                    # A dimension-only patch (for example changing just the
                    # region) must not let extractor defaults widen an exact
                    # previously selected time range.
                    if carried.get("start_year") is not None:
                        entities["start_year"] = carried["start_year"]
                    if carried.get("end_year") is not None:
                        entities["end_year"] = carried["end_year"]

                year_compare = re.fullmatch(
                    r"(?i)\s*compare\s+(?:(?:that|it|this)\s+)?(?:with|to|against)\s+(\d{4})[?.!]?\s*",
                    original_user_query,
                )
                if year_compare:
                    target = int(year_compare.group(1))
                    prior_years = [
                        int(y) for y in (carried.get("start_year"), carried.get("end_year"))
                        if y is not None
                    ]
                    entities["start_year"] = min(prior_years + [target])
                    entities["end_year"] = max(prior_years + [target])
                    entities["comparison"] = {"dimension": "year", "values": sorted(set(prior_years + [target]))}

                explicit_chart_match = re.search(
                    r"\b(bar|column|line|scatter|area)(?:\s+chart)?\b",
                    original_user_query,
                    re.IGNORECASE,
                )
                if explicit_chart_match and re.search(
                    r"\b(?:plot|graph|chart|draw|visuali[sz]e)\b",
                    original_user_query,
                    re.IGNORECASE,
                ):
                    requested_chart = explicit_chart_match.group(1).casefold()
                    entities["chart_type"] = "bar" if requested_chart == "column" else requested_chart

                if self._is_generic_followup(original_user_query) and re.search(
                    r"\b(?:plot|graph|chart|draw|visuali[sz]e)\b", original_user_query, re.IGNORECASE
                ):
                    entities["action"] = "plot"
                    # A contextual plot of one requested year is a categorical
                    # comparison across the selected series.  Tell the plotter
                    # that explicitly; its safe default is a line chart, which
                    # is appropriate for trajectories but misleading for a
                    # single x-value.  An explicit chart type always wins.
                    start_year = entities.get("start_year")
                    end_year = entities.get("end_year")
                    if (
                        not explicit_chart_match
                        and not entities.get("chart_type")
                        and start_year is not None
                        and start_year == end_year
                        and not YearFilter(
                            start_year,
                            end_year,
                            explicit=True,
                            operator="exact",
                        ).is_latest
                    ):
                        entities["chart_type"] = "bar"
                if confidence:
                    entities["entity_confidence"] = confidence

            if prior_comparison_query:
                previous_scope = dict(getattr(self, "previous_entities", {}) or {})
                current_scope = dict(carried or {})
                plural_keys = {"region": "regions", "scenario": "scenarios", "model": "models"}

                def _ordered_scope_values(scope, dimension):
                    raw = scope.get(plural_keys[dimension])
                    if not raw:
                        raw = scope.get(dimension)
                    values = raw if isinstance(raw, (list, tuple, set)) else [raw]
                    return [str(value).strip() for value in values if str(value or "").strip()]

                comparison_dimensions = []
                for dimension in plural_keys:
                    old_values = _ordered_scope_values(previous_scope, dimension)
                    new_values = _ordered_scope_values(current_scope, dimension)
                    old_normalized = {value.casefold() for value in old_values}
                    new_normalized = {value.casefold() for value in new_values}
                    if old_normalized and new_normalized and old_normalized != new_normalized:
                        comparison_dimensions.append(dimension)

                # _render_previous_scope_comparison already established that
                # exactly one dimension changed. Re-apply that comparison as
                # structured state so unchanged plural scopes do not have to
                # survive a lossy round-trip through generated query text.
                if len(comparison_dimensions) == 1:
                    comparison_dimension = comparison_dimensions[0]
                    for dimension, plural_key in plural_keys.items():
                        old_values = _ordered_scope_values(previous_scope, dimension)
                        current_values = _ordered_scope_values(current_scope, dimension)
                        if dimension == comparison_dimension:
                            entities[plural_key] = old_values + [
                                value for value in current_values
                                if value.casefold() not in {item.casefold() for item in old_values}
                            ]
                            entities.pop(dimension, None)
                        elif len(current_values) > 1:
                            entities[plural_key] = current_values
                            entities.pop(dimension, None)
                        elif len(current_values) == 1:
                            entities[dimension] = current_values[0]
                            entities.pop(plural_key, None)
                    entities["comparison"] = comparison_dimension
                    entities["variable"] = current_scope.get("variable", entities.get("variable"))
                    entities["start_year"] = current_scope.get("start_year")
                    entities["end_year"] = current_scope.get("end_year")
                    entities["action"] = "plot"

            # Sanity-check extracted entities against explicit query cues
            ql = query.lower()
            if entities.get("variable"):
                available_variables = getattr(self.entity_extractor, "available_variables", []) or []
                exact_query_variable = self._match_catalog_value_from_text(
                    query, available_variables
                )
                current_variable = str(entities.get("variable") or "").strip()
                # A literal parent path in natural language must not replace a
                # more specific descendant already supported by the extractor
                # (e.g. "Family Carrier" -> ``Family|Carrier``).  Exact peers
                # and more specific literal paths still take precedence.
                if exact_query_variable:
                    entities["variable"] = self._best_supported_variable_path(
                        exact_query_variable,
                        current_variable,
                        query,
                        available_variables,
                    )
                elif (
                    workspace_study_followup
                    and current_variable.casefold() in {
                        "emissions|co2", "emissions|ch4", "emissions|n2o",
                    }
                    and re.search(r"\bemissions?\b", original_user_query, re.IGNORECASE)
                ):
                    # Preserve the explicit study-scoped gas resolved above.
                    # The general sanitizer cannot see a literal IAMC path in
                    # natural wording such as “global methane emissions”.
                    entities["variable"] = current_variable
                else:
                    entities["variable"] = sanitize_variable_for_query(current_variable, query)

            # A unit is meaningful only for the variable that survived the
            # query-evidence guard above.  The extractor may ground a unit to
            # an LLM-proposed catalogue variable and then correctly reject that
            # variable as unrelated (for example bare ``oil`` resolving to a
            # BEUR-valued capacity series).  Never carry that orphaned unit into
            # a clarification or its provenance.
            resolved_variable = str(entities.get("variable") or "").strip()
            variable_units = getattr(self.entity_extractor, "variable_units", {}) or {}
            if resolved_variable and resolved_variable in variable_units:
                entities["unit"] = variable_units[resolved_variable]
            elif not resolved_variable:
                entities.pop("unit", None)

            entities = self._repair_comparison_entities(query, entities)

            if re.search(r"\b(world|global|globally)\b", ql):
                existing_region = str(entities.get("region") or "").strip()
                # Only override when no region was resolved, or the resolved one
                # is not literally present in the query (i.e. it was guessed).
                if not existing_region or existing_region.lower() not in ql:
                    entities["region"] = "World"
            runtime_models = getattr(self.entity_extractor, "available_models", []) or []
            current_model = str(entities.get("model") or "").strip()
            exact_runtime_model = next(
                (
                    str(value) for value in runtime_models
                    if str(value).casefold() == current_model.casefold()
                ),
                "",
            )
            if exact_runtime_model:
                entities["model"] = exact_runtime_model
                confidence = dict(entities.get("entity_confidence") or {})
                confidence["model"] = max(float(confidence.get("model", 0) or 0), 1.0)
                entities["entity_confidence"] = confidence
            profile = find_model_profile(query)
            if profile and not exact_runtime_model:
                entities["model"] = str(profile.get("name", "") or entities.get("model") or "")
                confidence = dict(entities.get("entity_confidence") or {})
                confidence["model"] = max(float(confidence.get("model", 0) or 0), 0.9)
                entities["entity_confidence"] = confidence
                entities["confidence"] = max(float(entities.get("confidence", 0) or 0), 0.9)
            if not entities.get("model"):
                profile = find_model_profile(query)
                if profile:
                    entities["model"] = str(profile.get("name", ""))
            if entities.get("model") and _looks_like_model_info_request(query) and find_model_profile(query):
                confidence = dict(entities.get("entity_confidence") or {})
                confidence["model"] = max(float(confidence.get("model", 0) or 0), 0.9)
                entities["entity_confidence"] = confidence

            low_confidence_prompt = self._low_confidence_entity_prompt(
                entities,
                original_user_query,
            )
            if low_confidence_prompt:
                self._record_route_decision("data_query", 0.45, "deterministic", "low confidence entity clarification")
                self._update_clarification_context("data_query", query, low_confidence_prompt, entities)
                self._persist_last_entities(entities, low_confidence_prompt)
                return self._append_relevant_links(
                    low_confidence_prompt, query, entities, "data_query",
                )

            textual_comparison = self._textual_comparison_answer(query, entities)
            if textual_comparison:
                self._record_route_decision("data_query", 0.9, "deterministic", "textual comparison question")
                self._update_clarification_context("data_query", query, textual_comparison, entities)
                self._persist_last_entities(entities, textual_comparison)
                return self._append_relevant_links(textual_comparison, query, entities, "data_query")

            route_decision = self._deterministic_route_decision(query, entities)
            if route_decision:
                if route_decision.get("clear_entities"):
                    # Capability questions carry no data scope; a scenario-like
                    # noun such as "policy" must not leak into follow-up state.
                    entities = {}
                agent_name = self._record_route_decision(
                    route_decision["agent"],
                    route_decision["confidence"],
                    route_decision["source"],
                    route_decision["reason"],
                )
            else:
                agent_name = self._route_with_llm_fallback(query, entities)

        except Exception as e:
            self.logger.error(f"Routing error: {e}")
            agent_name = self._classify_route_heuristic(query, {})
            self._record_route_decision(agent_name, 0.45, "heuristic", "routing exception")
            entities = {}

        # Any implicit comparison that a classifier still labels as plotting
        # is returned as a numeric table. Explicit chart requests were already
        # redirected to the explorer at the start of this method.
        if agent_name == "data_plotting":
            agent_name = self._record_route_decision(
                "data_query", 0.95, "policy", "chat plotting disabled; return numeric table",
            )
            entities = dict(entities or {})
            entities["action"] = "query"

        self.logger.debug("Routing query to %s agent.", agent_name)
        agent = self.agents.get(agent_name)
        if not agent:
            return "Sorry, the requested agent is not available."

        try:
            if (
                agent_name == "general_qa"
                and self._is_site_navigation_request(query)
                and self.shared_resources.get("link_catalog")
            ):
                entities = {}
                response = self._grounded_site_navigation_answer(query, entities)
                self._conversation().response_scope_override = {}
                return response

            # Pass entities to agent if it supports them
            if hasattr(agent, 'handle_with_entities'):
                response = agent.handle_with_entities(query, entities, history)
            else:
                response = agent.handle(query, history)

            response = self._workspace_result_answer(query, response)
            if agent_name == "model_explanation":
                response = self._model_metadata_fallback_answer(query, response, entities)
            if not str(response or "").strip():
                response = "I need one more detail to continue. Please specify the variable, region, or scenario."

            response = self._maybe_add_followup_guidance(response, query, agent_name)
            self._update_clarification_context(agent_name, query, response, entities)
            # Persist last entities for follow-up context
            self._persist_last_entities(entities, response)
            return self._append_relevant_links(response, query, entities, agent_name)
        except Exception as e:
            self.logger.error(f"Error handling query with {agent_name}: {e}")
            # Navigation/general_qa failures (e.g. embeddings provider down) can
            # still be answered from the link catalog without any LLM call.
            if (
                agent_name == "general_qa"
                and self.shared_resources.get("link_catalog")
                and (self._is_site_navigation_request(query) or self._is_provider_error(e))
            ):
                nav_answer = self._grounded_site_navigation_answer(query, entities or {})
                if self.last_links:
                    return nav_answer
            # When provider/API calls fail, fallback to deterministic data_query where possible.
            if self._is_provider_error(e) and agent_name != "data_query":
                fallback = self.agents.get("data_query")
                if fallback:
                    try:
                        if hasattr(fallback, "handle_with_entities"):
                            response = fallback.handle_with_entities(query, entities, history)
                        else:
                            response = fallback.handle(query, history)
                        if str(response or "").strip():
                            return self._append_relevant_links(response, query, entities, "data_query")
                    except Exception as fallback_err:
                        self.logger.error("Fallback data_query failed: %s", fallback_err)
            # Do not surface the raw exception text to the user (may contain
            # internal details); it is already logged above. Distinguish an
            # outage (retry later) from a query problem (rephrase).
            if self._is_provider_error(e):
                return (
                    "The AI service is temporarily unavailable, so I could not complete this request. "
                    "Please try again in a moment — your question was fine."
                )
            return "Sorry, I encountered an error while processing your request. Please try rephrasing your question."

    def _initialize_agents(self):
        """Initialize all agents with shared resources."""
        self.agents["data_query"] = DataQueryAgent(self.shared_resources, self.streaming)
        self.agents["model_explanation"] = ModelExplanationAgent(self.shared_resources, self.streaming)
        self.agents["data_plotting"] = DataPlottingAgent(self.shared_resources, self.streaming)
        self.agents["general_qa"] = GeneralQAAgent(self.shared_resources, self.streaming)
        self.agents["modelling_suggestions"] = ModellingSuggestionsAgent(self.shared_resources, self.streaming)
        self.logger.debug("All agents initialized successfully.")

    _GREEK_CHARS = re.compile(r"[Ͱ-Ͽἀ-῿]")

    def _translate_to_english(self, query: str) -> str:
        """Translate a non-English (Greek) query to English via the router LLM."""
        try:
            response = self.router_llm.invoke(
                "Translate the following question for a climate-data chatbot into English. "
                "Keep model, scenario, variable and region names unchanged. "
                "Return ONLY the English translation, nothing else.\n\n"
                f"Question: {query}"
            )
            return str(getattr(response, "content", "") or "").strip()
        except Exception as err:
            self.logger.warning("Query translation failed: %s", err)
            return ""

    def route_query(self, query: str, history: Optional[List[Tuple[str, str]]] = None) -> str:
        """Route the query to the appropriate agent using LLM-based classification."""
        self.last_links = []
        self.turn_counter = getattr(self, "turn_counter", 0) + 1
        self.current_turn = self.turn_counter
        consume_resolved_scope()  # drop any stale scope from a previous turn

        # Greek queries: the normalizer/heuristics only understand English, so
        # translate first instead of silently mis-routing a garbled query.
        if self._GREEK_CHARS.search(str(query or "")):
            translated = self._translate_to_english(query)
            if translated and not self._GREEK_CHARS.search(translated):
                self.logger.info("Translated Greek query to English: %s", translated)
                query = translated
            else:
                return (
                    "Προς το παρόν απαντώ αξιόπιστα μόνο σε ερωτήσεις στα αγγλικά. "
                    "Παρακαλώ ξαναδιατύπωσε την ερώτησή σου στα αγγλικά.\n\n"
                    "I currently answer questions in English only. "
                    "Please rephrase your question in English."
                )

        subqueries = self._split_multi_intent(query)
        if len(subqueries) > 1:
            responses = []
            parent_plan = build_query_plan(
                query,
                available_models=getattr(self.entity_extractor, "available_models", []) or [],
            )
            shared_scope = dict(self.last_entities or {})
            parent_profiles = find_model_profiles(query)
            if len(parent_profiles) == 1:
                shared_scope["model"] = str(parent_profiles[0].get("name") or "")
                shared_scope.pop("models", None)
            elif len(parent_profiles) > 1:
                shared_scope["models"] = [
                    str(profile.get("name")) for profile in parent_profiles if profile.get("name")
                ]
                shared_scope.pop("model", None)
            elif len(parent_plan.mentioned_models) == 1:
                shared_scope["model"] = parent_plan.mentioned_models[0]
                shared_scope.pop("models", None)
            elif len(parent_plan.mentioned_models) > 1:
                shared_scope["models"] = list(parent_plan.mentioned_models)
                shared_scope.pop("model", None)
            context: Dict[str, Any] = {"last_entities": shared_scope} if shared_scope else {}
            for idx, subq in enumerate(subqueries, start=1):
                routed_subq = subq
                prior_entities = dict(context.get("last_entities") or {})
                if prior_entities and re.search(
                    r"\b(?:it|its|this|that|these|those|them|their|the\s+model|the\s+result)\b",
                    subq,
                    flags=re.IGNORECASE,
                ):
                    routed_subq = self._compose_contextual_query(subq, prior_entities)
                resp = self._route_single(routed_subq, history, context=context)
                responses.append(f"**{idx}. {subq}**\n{resp}")
                # The structured state persisted by the completed segment is
                # authoritative. Re-extracting only the raw subquery can erase
                # model/region context resolved by deterministic routing.
                if self.last_entities:
                    context["last_entities"] = dict(self.last_entities)
            return "\n\n".join(responses)

        return self._route_single(query, history, context={"last_entities": self.last_entities})

    def get_agent_names(self) -> List[str]:
        """Return the list of available agent names."""
        return list(self.agents.keys())
