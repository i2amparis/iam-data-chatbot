import json
import logging
import re
from typing import Dict, Any, List, Tuple, Optional
from agents import BaseAgent, DataQueryAgent, ModelExplanationAgent, DataPlottingAgent, GeneralQAAgent, ModellingSuggestionsAgent
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from pathlib import Path
from data_utils import (
    _choice_prompt,
    _infer_variable_intent,
    _looks_like_comparison_request,
    _looks_like_category_list_request,
    _looks_like_data_request,
    _looks_like_model_info_request,
    _looks_like_plot_request,
    _variable_matches_query_signal,
    sanitize_variable_for_query,
)
from canonical_aliases import canonical_scenario_from_query, preferred_variable_from_query
from link_router import format_relevant_links, suggest_links
from model_profiles import find_model_profile, format_model_profile_answer
from year_filters import extract_year_range
from llm_config import ROUTER_MODEL
from resolved_scope import consume_resolved_scope, record_resolved_scope


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
    "generic_site_targets": (
        "documentation", "docs", "user guide", "scenario explorer",
        "model documentation", "application library", "data portal",
        "dashboard", "tutorial", "iam paris results", "paris results",
        "results page", "scenario database",
    ),
    "strong_nav_phrases": (
        "where can i find", "where do i find", "where is",
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


def _looks_like_site_navigation_request(query: str) -> bool:
    q = str(query or "").strip().lower()
    if not q:
        return False
    navigation_terms = _NAV_TERMS["navigation_terms"]
    named_site_items = _NAV_TERMS["named_site_items"]
    data_story_items = _NAV_TERMS["data_story_items"]
    project_workspace_items = _NAV_TERMS["project_workspace_items"]
    analysis_contact_items = _NAV_TERMS["analysis_contact_items"]
    generic_site_targets = _NAV_TERMS["generic_site_targets"]
    strong_nav_phrases = _NAV_TERMS["strong_nav_phrases"]

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
    def __init__(self, shared_resources: Dict[str, Any], streaming: bool = True):
        self.shared_resources = shared_resources
        self.streaming = streaming
        self.logger = logging.getLogger(self.__class__.__name__)
        self.agents: Dict[str, BaseAgent] = {}
        self._initialize_agents()
        self.last_entities: Dict[str, Any] = {}
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
            self.entity_extractor = QueryEntityExtractor(
                models=shared_resources.get("models", []),
                ts_data=shared_resources.get("ts", []),
                api_key=shared_resources["env"]["OPENAI_API_KEY"]
            )
            shared_resources["entity_extractor"] = self.entity_extractor

        # LLM for intelligent query routing
        self.router_llm = ChatOpenAI(
            model_name=ROUTER_MODEL,
            temperature=0,
            streaming=False,
            timeout=30,
            max_retries=1,
            api_key=self.shared_resources["env"]["OPENAI_API_KEY"],
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

    def _looks_like_clarification_response(self, response: str) -> bool:
        text = str(response or "")
        markers = (
            "Choose the variable:",
            "Choose the region:",
            "Choose the scenario:",
            "Closest valid options:",
            "Which variable should I use?",
            "Which variable or region should I use instead?",
            "Please provide the",
            "I need one more detail",
            "I don't have an active numbered choice",
            "Reply with a number",
        )
        return any(marker.lower() in text.lower() for marker in markers)

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
            links = suggest_links(
                query,
                catalog,
                agent_name="general_qa",
                entities=entities or {},
                variable_intent=_infer_variable_intent(query),
            )
        except Exception as err:
            self.logger.warning("Could not build grounded IAM PARIS link answer: %s", err)
            self.last_links = []
            return "I could not match this request to a reliable IAM PARIS link."

        self.last_links = links
        if not links:
            return "I could not match this request to a reliable IAM PARIS link."

        lines = ["Use these IAM PARIS links for this request:"]
        formatted = format_relevant_links(links)
        if formatted:
            lines.extend(["", formatted])
        search_hints = [
            str(link.get("search_hint", "")).strip()
            for link in links
            if str(link.get("search_hint", "")).strip()
        ]
        if search_hints:
            lines.extend([
                "",
                "If the direct detail page is not available, open the Application Library and search for: "
                + ", ".join(dict.fromkeys(search_hints))
                + ".",
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
        link_text = " ".join(
            " ".join(str(link.get(key) or "") for key in ("title", "url", "reason", "search_hint"))
            for link in links
            if isinstance(link, dict)
        ).lower()
        if "results" in link_text or "/results" in link_text:
            return links

        fallback = next((item for item in catalog if item.get("url") == "https://iamparis.eu/results"), None)
        if not fallback:
            return links
        result_link = {
            "title": str(fallback.get("title", "IAM PARIS Results")),
            "url": str(fallback.get("url", "https://iamparis.eu/results")),
            "reason": "General IAM PARIS results page for data follow-ups.",
            "confidence": 0.25,
            "search_hint": str(fallback.get("search_hint", "")),
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
        if (has_year_filter and names_scenario) or "latest" in q:
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
            )
        return (
            f"I matched `{model}`, but IAM PARIS does not expose a dedicated assumptions/metadata page "
            "for that model in the local model catalog. Use the IAM PARIS model and results links below "
            "to inspect related documentation or available data."
        )

    def _models_covering_topic_answer(self, query: str) -> Optional[str]:
        """N4: when a model-list request carries a sector/topic qualifier, return a
        deterministic subset of models that report data for that topic instead of the
        full model list. Returns None when no topic is detected or metadata is missing."""
        metadata = self.shared_resources.get("metadata")
        if not metadata or not hasattr(metadata, "models_covering_topic"):
            return None
        category, models = metadata.models_covering_topic(query)
        if not category or not models:
            return None
        total = len(metadata.all_model_names) if hasattr(metadata, "all_model_names") else None
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
        if val_a == val_b:
            verdict = f"Both are equal in {year}."
        else:
            higher = var_a if val_a > val_b else var_b
            verdict = f"`{higher}` is higher in {year}."
        record_resolved_scope(
            variable=var_a,
            region=region_key,
            scenario=scenario_key,
            model=model_key,
        )
        lines = [
            f"### Comparison — {var_a} vs {var_b} ({region_key})",
            "",
            f"In {year} under scenario `{scenario_key}` (model `{model_key}`):",
            f"- `{var_a}`: {val_a:,.2f} {unit_a}".rstrip(),
            f"- `{var_b}`: {val_b:,.2f} {unit_b}".rstrip(),
            "",
            verdict,
        ]
        if unit_a and unit_b and unit_a != unit_b:
            lines.append("Note: the two variables use different units, so compare with care.")
        lines.append(f"Ask `plot compare {var_a} versus {var_b}` to see the full trajectories.")
        return "\n".join(lines)

    def _low_confidence_entity_prompt(self, entities: Optional[Dict[str, Any]]) -> str:
        entities = entities or {}
        confidence = entities.get("entity_confidence") or {}
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
        return any(
            re.search(r"(?<!\w)" + re.escape(name) + r"(?!\w)", q)
            for name in model_names[:200]
            if name
        )

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

        if _looks_like_site_navigation_request(query):
            return {
                "agent": "general_qa",
                "confidence": 0.88,
                "source": "deterministic",
                "reason": "site/navigation link request",
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
        if _looks_like_site_navigation_request(query):
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
            if self._is_intentful_segment(part):
                merged.append(part)
            else:
                merged[-1] = f"{merged[-1]} {part}".strip()

        return merged if len(merged) > 1 else [q]

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

        if self._is_contextual_dimension_followup(query):
            same_for = re.search(r"\bsame\s+for\s+(.+)$", query, re.IGNORECASE)
            what_about = re.search(r"\b(?:what|how)\s+about\s+(.+)$", query, re.IGNORECASE)
            compare_with = re.search(r"\bcompare\s+(?:with|to|against)\s+(.+)$", query, re.IGNORECASE)
            scope_year = re.match(r"\s*((?:after|before|by|in|from)\s+\d{4}(?:\s*(?:to|until|-)\s*\d{4})?)\s*$", query, re.IGNORECASE)
            # Dimension switches ("under PR_NDC_CP", "now for CHN"): keep the
            # carried scope and replace only the named dimension. Strip filler on
            # the original-case query so scenario codes keep their casing.
            _stripped = re.sub(self._FOLLOWUP_FILLER, "", query.strip(), flags=re.IGNORECASE).strip().rstrip("?").strip()
            under_switch = re.fullmatch(r"(?i)under\s+(.+)", _stripped)
            for_switch = re.fullmatch(r"(?i)for\s+(.+)", _stripped)

            if "show all scenarios" in ql:
                parts = ["show all scenarios"]
                if variable:
                    parts.append(f"for {variable}")
                if region:
                    parts.append(f"in {region}")
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
            if scenario:
                parts.append(f"under {scenario}")
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

        # N5: a failed/clarification/no-data turn must not wipe the scope carried
        # over from the last *successful* answer. Seed the merge with the previous
        # known scope so an incomplete failed turn keeps variable/region/etc.
        if self._is_unsuccessful_response(text):
            merged = prior
        else:
            merged = dict(entities or {}) if entities is not None else prior

        for key in ("variable", "region", "scenario", "model"):
            value = str((entities or {}).get(key, "") or "").strip()
            if value:
                merged[key] = value

        # Preferred channel: the scope the answer formatter actually resolved,
        # reported structurally by data_utils/simple_plotter.
        structured_scope = consume_resolved_scope()
        if structured_scope and not self._is_unsuccessful_response(text):
            for key in ("variable", "region", "scenario", "model"):
                value = str(structured_scope.get(key, "") or "").strip()
                if value:
                    merged[key] = value
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

        if merged:
            self.last_entities = merged

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
        "sorry, the requested agent",
    )

    def _is_unsuccessful_response(self, text: str) -> bool:
        """Heuristic: did this turn fail to produce a real data/model answer?"""
        body = str(text or "").strip()
        if not body:
            return True
        lowered = body.lower()
        return any(marker in lowered for marker in self._UNSUCCESSFUL_RESPONSE_MARKERS)

    def _is_generic_followup(self, query: str) -> bool:
        ql = query.strip().lower()
        # Strip leading filler so "now plot it" / "ok show me that" are still
        # recognised as context-carrying follow-ups.
        ql = re.sub(r"^(?:(?:ok(?:ay)?|now|then|so|and|please|just|also|yeah|yes)\s+)+", "", ql).strip()
        if ql in {"continue", "keep going", "what about it"}:
            return True
        return bool(
            re.fullmatch(
                r"(?:plot|graph|chart|show|display|give|use)\s+(?:me\s+)?(?:it|this|that|those|them|the same)"
                r"(?:\s+(?:both|all|together|again|too|data))?",
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

    _FOLLOWUP_FILLER = r"^(?:(?:ok(?:ay)?|now|then|so|and|also|please)\s+)+"

    def _is_contextual_dimension_followup(self, query: str) -> bool:
        ql = query.strip().lower()
        if (
            re.fullmatch(r"same\s+for\s+.+", ql)
            or re.fullmatch(r"(?:what|how)\s+about\s+.+", ql)
            or re.fullmatch(r"compare\s+(?:with|to|against)\s+.+", ql)
            or re.fullmatch(r"(?:after|before|by|in|from)\s+\d{4}(?:\s*(?:to|until|-)\s*\d{4})?", ql)
            or ql == "show all scenarios"
        ):
            return True
        # Dimension switches: "under PR_NDC_CP", "now for CHN", "and under Baseline".
        # A short trailing phrase is the new scenario ("under X") or region ("for X").
        stripped = re.sub(self._FOLLOWUP_FILLER, "", ql).strip().rstrip("?").strip()
        m = re.fullmatch(r"(?:under|for)\s+(.+)", stripped)
        return bool(m and 1 <= len(m.group(1).split()) <= 4)

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

        # Exact normalized match first
        for scen in scenarios:
            if self._normalize_text(str(scen)) == q_norm:
                return str(scen)

        # Containment match for shorthand follow-ups
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

        # Preferred format: "1. `...` 2. `...`"
        numbered_backticked = re.findall(r"\b\d+\.\s*(?:[A-Za-z]+\s+)?`([^`]+)`", response)
        if numbered_backticked:
            deduped: List[str] = []
            for item in numbered_backticked:
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

    def _update_clarification_context(
        self,
        agent_name: str,
        query: str,
        response: str,
        entities: Optional[Dict[str, Any]] = None,
        base_query: Optional[str] = None,
    ) -> None:
        entities = entities or {}
        if agent_name == "data_plotting" and ("Please clarify" in response or "matched multiple" in response):
            self.clarification_context = {
                "original_query": query,
                "base_query": base_query or query,
                "ambiguous_response": response,
                "agent_type": agent_name,
                "entities": entities,
                "issued_turn": getattr(self, "current_turn", 0),
            }
            return
        if agent_name == "data_query" and (
            "Should I use it?" in response
            or "I think the best match is" in response
            or "Choose the variable:" in response
            or "Choose the region:" in response
            or "Choose the scenario:" in response
            or "Closest valid options:" in response
        ):
            options = self._extract_candidate_options(response)
            suggested_kind = "variable"
            if "Choose the region:" in response:
                suggested_kind = "region"
            elif "Choose the scenario:" in response:
                suggested_kind = "scenario"
            elif "Closest valid options:" in response:
                first_option = re.search(
                    r"\b1\.\s*(variable|region|scenario)\s+`",
                    response,
                    re.IGNORECASE,
                )
                if first_option:
                    suggested_kind = first_option.group(1).lower()
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
                "original_query": query,
                "base_query": base_query or query,
                "agent_type": agent_name,
                "entities": entities,
                "suggested_variable": suggested_variable,
                "suggested_options": options,
                "suggested_kind": suggested_kind,
                "suggested_region": suggested_region,
                "suggested_scenario": suggested_scenario,
                "response": response,
                "issued_turn": getattr(self, "current_turn", 0),
            }

    def _route_single(
        self,
        query: str,
        history: Optional[List[Tuple[str, str]]] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Route a single-intent query."""
        # Be tolerant of copied chat prefixes like "YOU: YOU: 1"
        query = re.sub(r"^(?:\s*you:\s*)+", "", str(query or ""), flags=re.IGNORECASE).strip()
        q_lower = query.strip().lower()

        # Greetings/thanks/help: answer directly and keep any pending
        # clarification context intact for the next real message.
        if self._is_small_talk(query):
            self._record_route_decision("general_qa", 0.95, "deterministic", "greeting/small talk")
            self.last_links = []
            if self.clarification_context:
                # A greeting should not burn the clarification window.
                self.clarification_context["issued_turn"] = getattr(self, "current_turn", 0)
            return self._small_talk_answer(query)

        if hasattr(self, "clarification_context") and self.clarification_context:
            issued_turn = int((self.clarification_context or {}).get("issued_turn", getattr(self, "current_turn", 0)))
            # Keep the pending choice alive for a few turns so an intervening
            # message does not silently discard the user's next "2"/"yes".
            if getattr(self, "current_turn", 0) > issued_turn + 3:
                self.clarification_context = None
            elif not self._is_clarification_followup(query, self.clarification_context):
                self.clarification_context = None

        early_carried = {}
        if context:
            early_carried = context.get("last_entities", {})
        if not early_carried and self.last_entities:
            early_carried = self.last_entities
        was_contextual_followup = bool(early_carried and self._is_contextual_dimension_followup(query))
        if was_contextual_followup:
            query = self._compose_contextual_query(query, early_carried)
            q_lower = query.strip().lower()

        if _looks_like_category_list_request(query, "variables") and _looks_like_plot_request(query):
            agent = self.agents.get("data_query")
            if not agent:
                return "Sorry, the requested agent is not available."
            self.last_entities = {}
            self._record_route_decision("data_query", 0.95, "deterministic", "plot variable discovery request")
            response = agent.handle(query, history)
            return self._append_relevant_links(response, query, {}, "data_query")
        if _looks_like_category_list_request(query, "models") and not was_contextual_followup:
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
        if _looks_like_category_list_request(query, "scenarios"):
            agent = self.agents.get("data_query")
            if not agent:
                return "Sorry, the requested agent is not available."
            self.last_entities = {}
            self._record_route_decision("data_query", 0.95, "deterministic", "scenario availability request")
            response = agent.handle(query, history)
            response = self._maybe_add_followup_guidance(response, query, "data_query")
            return self._append_relevant_links(response, query, {}, "data_query")
        if _looks_like_category_list_request(query, "variables"):
            agent = self.agents.get("data_query")
            if not agent:
                return "Sorry, the requested agent is not available."
            self.last_entities = {}
            self._record_route_decision("data_query", 0.95, "deterministic", "variable availability request")
            response = agent.handle(query, history)
            return self._append_relevant_links(response, query, {}, "data_query")
        if _looks_like_category_list_request(query, "regions"):
            agent = self.agents.get("data_query")
            if not agent:
                return "Sorry, the requested agent is not available."
            self.last_entities = {}
            self._record_route_decision("data_query", 0.95, "deterministic", "region availability request")
            response = agent.handle(query, history)
            return self._append_relevant_links(response, query, {}, "data_query")

        fresh_interrupt = bool(
            _looks_like_site_navigation_request(query)
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
            option_choice_idx = self._extract_option_choice(
                query,
                len(clar_ctx.get("suggested_options", []) or []),
            )
            if option_choice_idx is not None:
                options = clar_ctx.get("suggested_options", []) or []
                selected = str(options[option_choice_idx]).strip()
                if selected:
                    kind = clar_ctx.get("suggested_kind", "variable")
                    if kind == "region":
                        clar_ctx["suggested_region"] = selected
                    elif kind == "scenario":
                        clar_ctx["suggested_scenario"] = selected
                    else:
                        clar_ctx["suggested_variable"] = selected
                    query = "yes"
            if self._is_affirmation(query):
                pending_type = clar_ctx.get("agent_type", "")
                original_query = str(clar_ctx.get("original_query", "")).strip()
                base_query = str(clar_ctx.get("base_query", "") or original_query).strip()
                suggested_variable = str(clar_ctx.get("suggested_variable", "")).strip()
                suggested_region = str(clar_ctx.get("suggested_region", "")).strip()
                suggested_scenario = str(clar_ctx.get("suggested_scenario", "")).strip()
                merged_entities = dict(clar_ctx.get("entities", {}) or {})
                if suggested_variable:
                    merged_entities["variable"] = suggested_variable
                if suggested_region:
                    merged_entities["region"] = suggested_region
                if suggested_scenario:
                    merged_entities["scenario"] = suggested_scenario
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
                    self._persist_last_entities(merged_entities, response)
                    self._update_clarification_context("data_query", followup_query, response, merged_entities, base_query=base_query)
                    return self._append_relevant_links(response, followup_query, merged_entities, "data_query")
                return self._route_single(followup_query, history, context={"last_entities": merged_entities})

            if self._is_rejection(query):
                pending_type = clar_ctx.get("agent_type", "")
                remaining_options = list(clar_ctx.get("suggested_options", []) or [])
                used_kind = clar_ctx.get("suggested_kind", "variable")
                if remaining_options:
                    rejected = ""
                    if used_kind == "region":
                        rejected = str(clar_ctx.get("suggested_region", "")).strip()
                    elif used_kind == "scenario":
                        rejected = str(clar_ctx.get("suggested_scenario", "")).strip()
                    else:
                        rejected = str(clar_ctx.get("suggested_variable", "")).strip()
                    if rejected and rejected in remaining_options:
                        remaining_options = [opt for opt in remaining_options if opt != rejected]
                    elif remaining_options:
                        remaining_options = remaining_options[1:]

                self.clarification_context = None
                if pending_type == "data_query" and remaining_options:
                    response = _choice_prompt(
                        "Okay, here are the next closest options.",
                        used_kind,
                        remaining_options[:3],
                    )
                    updated_entities = dict(clar_ctx.get("entities", {}) or {})
                    self._update_clarification_context(
                        "data_query",
                        str(clar_ctx.get("base_query", "") or clar_ctx.get("original_query", "") or ""),
                        response,
                        updated_entities,
                        base_query=str(clar_ctx.get("base_query", "") or clar_ctx.get("original_query", "") or ""),
                    )
                    return response
                if pending_type == "data_query":
                    return "Okay. Which variable should I use instead?"
                if pending_type == "data_plotting":
                    return "Okay. Which variable or region should I use instead?"
                return "Okay. Please give me the variable you want."

            # Treat non-yes/no follow-up text as clarification details to merge with clar_ctx.
            pending_type = clar_ctx.get("agent_type", "")
            original_query = str(clar_ctx.get("original_query", "")).strip()
            base_query = str(clar_ctx.get("base_query", "") or original_query).strip()
            merged_entities = dict(clar_ctx.get("entities", {}) or {})
            try:
                follow_entities = self.entity_extractor.extract(query)
            except Exception:
                follow_entities = {}

            for key in ("variable", "region", "scenario", "model"):
                value = str((follow_entities or {}).get(key, "") or "").strip()
                if value:
                    merged_entities[key] = value

            # Scenario shorthand fallback (e.g., "pr wwh cp")
            if not merged_entities.get("scenario"):
                scen = self._match_scenario_from_text(query)
                if scen:
                    merged_entities["scenario"] = scen

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

        generic_followup = (
            was_contextual_followup
            or self._is_generic_followup(query)
            or self._is_contextual_dimension_followup(query)
            or self._is_model_scope_followup(query)
        )

        # Carry context into generic follow-ups like "plot it" or "show me data".
        if generic_followup and carried and not was_contextual_followup:
            query = self._compose_contextual_query(query, carried)
            q_lower = query.strip().lower()

        # Extract entities from query using the new extractor
        try:
            entities = self.entity_extractor.extract(query)
            self.logger.debug(f"Extracted entities: {entities}")

            if generic_followup and carried:
                entities = dict(entities or {})
                confidence = dict(entities.get("entity_confidence") or {})
                for key in ("variable", "region", "scenario", "model"):
                    value = str(carried.get(key, "") or "").strip()
                    if value and (not entities.get(key) or key == "variable"):
                        entities[key] = value
                        confidence[key] = max(float(confidence.get(key, 0) or 0), 0.85)
                if confidence:
                    entities["entity_confidence"] = confidence

            # Sanity-check extracted entities against explicit query cues
            ql = query.lower()
            if entities.get("variable"):
                entities["variable"] = sanitize_variable_for_query(entities["variable"], query)

            entities = self._repair_comparison_entities(query, entities)

            if re.search(r"\b(world|global|globally)\b", ql):
                existing_region = str(entities.get("region") or "").strip()
                # Only override when no region was resolved, or the resolved one
                # is not literally present in the query (i.e. it was guessed).
                if not existing_region or existing_region.lower() not in ql:
                    entities["region"] = "World"
            profile = find_model_profile(query)
            if profile:
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

            low_confidence_prompt = self._low_confidence_entity_prompt(entities)
            if low_confidence_prompt:
                self._record_route_decision("data_query", 0.45, "deterministic", "low confidence entity clarification")
                self._update_clarification_context("data_query", query, low_confidence_prompt, entities)
                self._persist_last_entities(entities, low_confidence_prompt)
                return low_confidence_prompt

            textual_comparison = self._textual_comparison_answer(query, entities)
            if textual_comparison:
                self._record_route_decision("data_query", 0.9, "deterministic", "textual comparison question")
                self._update_clarification_context("data_query", query, textual_comparison, entities)
                self._persist_last_entities(entities, textual_comparison)
                return self._append_relevant_links(textual_comparison, query, entities, "data_query")

            route_decision = self._deterministic_route_decision(query, entities)
            if route_decision:
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

        self.logger.debug("Routing query to %s agent.", agent_name)
        agent = self.agents.get(agent_name)
        if not agent:
            return "Sorry, the requested agent is not available."

        try:
            if (
                agent_name == "general_qa"
                and _looks_like_site_navigation_request(query)
                and self.shared_resources.get("link_catalog")
            ):
                entities = {}
                response = self._grounded_site_navigation_answer(query, entities)
                self.last_entities = {}
                return response

            # Pass entities to agent if it supports them
            if hasattr(agent, 'handle_with_entities'):
                response = agent.handle_with_entities(query, entities, history)
            else:
                response = agent.handle(query, history)

            response = self._workspace_result_answer(query, response)
            if agent_name == "model_explanation":
                response = self._model_metadata_fallback_answer(query, response, entities)
                # Conceptual questions (e.g. "what is an integrated assessment
                # model") are not about a specific catalog model. If the model
                # agent could not match one, answer the concept via general_qa.
                if "couldn't match that to a known model" in str(response or "").lower():
                    gqa = self.agents.get("general_qa")
                    if gqa is not None:
                        try:
                            gqa_resp = (
                                gqa.handle_with_entities(query, {}, history)
                                if hasattr(gqa, "handle_with_entities")
                                else gqa.handle(query, history)
                            )
                            if str(gqa_resp or "").strip():
                                response = gqa_resp
                                agent_name = "general_qa"
                        except Exception as gqa_err:
                            self.logger.warning("general_qa fallback failed: %s", gqa_err)
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
                and (_looks_like_site_navigation_request(query) or self._is_provider_error(e))
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
            context: Dict[str, Any] = {}
            for idx, subq in enumerate(subqueries, start=1):
                resp = self._route_single(subq, history, context=context)
                responses.append(f"**{idx}. {subq}**\n{resp}")
                # Update context from entity extractor for simple carry-over
                try:
                    context["last_entities"] = self.entity_extractor.extract(subq)
                    # If response contains a header like "### Variable in Region", use it as authoritative
                    if isinstance(resp, str):
                        first_line = resp.splitlines()[0] if resp else ""
                        if first_line.startswith("### "):
                            header = first_line[4:].strip()
                            if " in " in header:
                                var, region = header.split(" in ", 1)
                                context["last_entities"]["variable"] = var.strip()
                                context["last_entities"]["region"] = region.strip()
                            else:
                                context["last_entities"]["variable"] = header.strip()
                except Exception:
                    pass
            return "\n\n".join(responses)

        return self._route_single(query, history, context={"last_entities": self.last_entities})

    def get_agent_names(self) -> List[str]:
        """Return the list of available agent names."""
        return list(self.agents.keys())
