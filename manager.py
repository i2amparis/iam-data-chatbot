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
    _looks_like_category_list_request,
    _looks_like_data_request,
    _looks_like_model_info_request,
    _looks_like_plot_request,
    _variable_matches_query_signal,
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


class MultiAgentManager:
    def __init__(self, shared_resources: Dict[str, Any], streaming: bool = True):
        self.shared_resources = shared_resources
        self.streaming = streaming
        self.logger = logging.getLogger(self.__class__.__name__)
        self.agents: Dict[str, BaseAgent] = {}
        self._initialize_agents()
        self.last_entities: Dict[str, Any] = {}
        self.turn_counter: int = 0
        self.current_turn: int = 0

        # Initialize Query Entity Extractor
        self.entity_extractor = QueryEntityExtractor(
            models=shared_resources.get("models", []),
            ts_data=shared_resources.get("ts", []),
            api_key=shared_resources["env"]["OPENAI_API_KEY"]
        )

        # LLM for intelligent query routing
        self.router_llm = ChatOpenAI(
            model_name="gpt-4-turbo",
            temperature=0,
            streaming=False,
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
    {skill_guidance}

    Question: {{query}}
    Answer:"""),
                HumanMessagePromptTemplate.from_template("Query: {query}")
            ])

    def _classify_route_heuristic(self, query: str, entities: Optional[Dict[str, Any]] = None) -> str:
        """
        Local, no-network route classifier used when router LLM is unavailable.
        """
        q = (query or "").strip().lower()
        entities = entities or {}

        if _looks_like_plot_request(query):
            return "data_plotting"
        model_names = [
            str(m.get("modelName", "")).lower()
            for m in self.shared_resources.get("models", [])
            if m and m.get("modelName")
        ]
        mentions_model = any(
            re.search(r"(?<!\w)" + re.escape(name) + r"(?!\w)", q)
            for name in model_names[:200]
            if name
        )
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
        if " and plot " in lower or lower.endswith(" and plot it") or " and plot it" in lower:
            import re
            parts = re.split(r"\s+and\s+", q)
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

        import re
        parts = re.split(r"\s+(?:and|then|also)\s+|;|\n", q)
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
                parts.append(f"for model {model}")
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
        merged = dict(entities or {}) if entities else dict(self.last_entities or {})
        for key in ("variable", "region", "scenario", "model"):
            value = str((entities or {}).get(key, "") or "").strip()
            if value:
                merged[key] = value

        text = str(response or "")
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

    def _is_generic_followup(self, query: str) -> bool:
        ql = query.strip().lower()
        if ql in {"continue", "keep going", "what about it"}:
            return True
        return bool(
            re.fullmatch(
                r"(?:plot|graph|chart|show|display|give|use)\s+(?:me\s+)?(?:it|this|that|those|them|the same)",
                ql,
            )
        )

    def _is_clarification_followup(self, query: str, context: Optional[Dict[str, Any]] = None) -> bool:
        q = str(query or "").strip()
        if not q:
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
        numbered_backticked = re.findall(r"\b\d+\.\s*`([^`]+)`", response)
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
        match = re.search(r"\b([1-9][0-9]*)\b", query or "")
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
        ):
            options = self._extract_candidate_options(response)
            suggested_kind = "variable"
            if "Choose the region:" in response:
                suggested_kind = "region"
            elif "Choose the scenario:" in response:
                suggested_kind = "scenario"
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

        if hasattr(self, "clarification_context") and self.clarification_context:
            issued_turn = int((self.clarification_context or {}).get("issued_turn", getattr(self, "current_turn", 0)))
            if getattr(self, "current_turn", 0) > issued_turn + 1:
                self.clarification_context = None
            elif not self._is_clarification_followup(query, self.clarification_context):
                self.clarification_context = None

        if _looks_like_category_list_request(query, "variables") and _looks_like_plot_request(query):
            agent = self.agents.get("data_query")
            if not agent:
                return "Sorry, the requested agent is not available."
            return agent.handle(query, history)
        if _looks_like_category_list_request(query, "models"):
            agent = self.agents.get("data_query")
            if not agent:
                return "Sorry, the requested agent is not available."
            return agent.handle(query, history)
        if _looks_like_category_list_request(query, "scenarios"):
            agent = self.agents.get("data_query")
            if not agent:
                return "Sorry, the requested agent is not available."
            return agent.handle(query, history)
        if _looks_like_category_list_request(query, "variables"):
            agent = self.agents.get("data_query")
            if not agent:
                return "Sorry, the requested agent is not available."
            return agent.handle(query, history)
        if _looks_like_category_list_request(query, "regions"):
            agent = self.agents.get("data_query")
            if not agent:
                return "Sorry, the requested agent is not available."
            return agent.handle(query, history)

        # Check for clarification responses first
        if hasattr(self, 'clarification_context') and self.clarification_context:
            context = self.clarification_context
            option_choice_idx = self._extract_option_choice(
                query,
                len(context.get("suggested_options", []) or []),
            )
            if option_choice_idx is not None:
                options = context.get("suggested_options", []) or []
                selected = str(options[option_choice_idx]).strip()
                if selected:
                    kind = context.get("suggested_kind", "variable")
                    if kind == "region":
                        context["suggested_region"] = selected
                    elif kind == "scenario":
                        context["suggested_scenario"] = selected
                    else:
                        context["suggested_variable"] = selected
                    query = "yes"
            if self._is_affirmation(query):
                pending_type = context.get("agent_type", "")
                original_query = str(context.get("original_query", "")).strip()
                base_query = str(context.get("base_query", "") or original_query).strip()
                suggested_variable = str(context.get("suggested_variable", "")).strip()
                suggested_region = str(context.get("suggested_region", "")).strip()
                suggested_scenario = str(context.get("suggested_scenario", "")).strip()
                merged_entities = dict(context.get("entities", {}) or {})
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
                    return response
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
                    return response
                return self._route_single(followup_query, history, context={"last_entities": merged_entities})

            if self._is_rejection(query):
                pending_type = context.get("agent_type", "")
                remaining_options = list(context.get("suggested_options", []) or [])
                used_kind = context.get("suggested_kind", "variable")
                if remaining_options:
                    rejected = ""
                    if used_kind == "region":
                        rejected = str(context.get("suggested_region", "")).strip()
                    elif used_kind == "scenario":
                        rejected = str(context.get("suggested_scenario", "")).strip()
                    else:
                        rejected = str(context.get("suggested_variable", "")).strip()
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
                    updated_entities = dict(context.get("entities", {}) or {})
                    self._update_clarification_context(
                        "data_query",
                        str(context.get("base_query", "") or context.get("original_query", "") or ""),
                        response,
                        updated_entities,
                        base_query=str(context.get("base_query", "") or context.get("original_query", "") or ""),
                    )
                    return response
                if pending_type == "data_query":
                    return "Okay. Which variable should I use instead?"
                if pending_type == "data_plotting":
                    return "Okay. Which variable or region should I use instead?"
                return "Okay. Please give me the variable you want."

            # Treat non-yes/no follow-up text as clarification details to merge with context.
            pending_type = context.get("agent_type", "")
            original_query = str(context.get("original_query", "")).strip()
            base_query = str(context.get("base_query", "") or original_query).strip()
            merged_entities = dict(context.get("entities", {}) or {})
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
                return response
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
                return response

        if re.fullmatch(r"\s*\d+\s*", query):
            return (
                "I don't have an active numbered choice right now. "
                "Ask a new query, or repeat the variable, region, or scenario you want."
            )

        carried = {}
        if context:
            carried = context.get("last_entities", {})
        if not carried and self.last_entities:
            carried = self.last_entities

        generic_followup = self._is_generic_followup(query)

        # Carry context into generic follow-ups like "plot it" or "show me data".
        if generic_followup and carried:
            query = self._compose_contextual_query(query, carried)
            q_lower = query.strip().lower()

        # Extract entities from query using the new extractor
        try:
            entities = self.entity_extractor.extract(query)
            self.logger.debug(f"Extracted entities: {entities}")

            if generic_followup and carried:
                entities = dict(entities or {})
                for key in ("variable", "region", "scenario", "model"):
                    value = str(carried.get(key, "") or "").strip()
                    if value:
                        entities[key] = value

            # Sanity-check extracted entities against explicit query cues
            ql = query.lower()
            var = entities.get("variable")
            if var:
                v = str(var).lower()
                if any(t in ql for t in ["co2", "emission", "emissions"]) and not ("co2" in v or "emission" in v):
                    entities["variable"] = None
                if "solar" in ql and "solar" not in v:
                    entities["variable"] = None
                if "wind" in ql and "wind" not in v:
                    entities["variable"] = None
                if "capacity" in ql and "capacity" not in v:
                    entities["variable"] = None
                if entities.get("variable"):
                    intent = _infer_variable_intent(query)
                    if not _variable_matches_query_signal(
                        str(entities["variable"]),
                        query,
                        intent,
                    ):
                        entities["variable"] = None

            if "world" in ql or "global" in ql:
                entities["region"] = "World"

            explicit_data_query = _looks_like_data_request(query)
            explicit_plot_query = _looks_like_plot_request(query)

            # Use extracted action to determine routing
            if explicit_data_query and not explicit_plot_query:
                agent_name = 'data_query'
            elif entities.get('action') == 'plot':
                agent_name = 'data_plotting'
            elif _looks_like_category_list_request(query, "variables") and explicit_plot_query:
                agent_name = 'data_query'
            else:
                # Use LLM for more complex routing, with local fallback on provider/network failures.
                try:
                    result = self.routing_prompt | self.router_llm
                    response_obj = result.invoke({"query": query})
                    agent_name = response_obj.content.strip().lower()
                except Exception as route_err:
                    self.logger.warning(
                        "Router LLM unavailable (%s). Falling back to heuristic routing.",
                        route_err,
                    )
                    agent_name = self._classify_route_heuristic(query, entities)

                # Validate agent name
                valid_agents = ["data_query", "data_plotting", "model_explanation", "general_qa", "modelling_suggestions"]
                if agent_name not in valid_agents:
                    agent_name = self._classify_route_heuristic(query, entities)

        except Exception as e:
            self.logger.error(f"Routing error: {e}")
            agent_name = self._classify_route_heuristic(query, {})
            entities = {}

        self.logger.debug(f"Routing query to {agent_name} agent.")
        agent = self.agents.get(agent_name)
        if not agent:
            return "Sorry, the requested agent is not available."

        try:
            # Pass entities to agent if it supports them
            if hasattr(agent, 'handle_with_entities'):
                response = agent.handle_with_entities(query, entities, history)
            else:
                response = agent.handle(query, history)

            if not str(response or "").strip():
                response = "I need one more detail to continue. Please specify the variable, region, or scenario."

            self._update_clarification_context(agent_name, query, response, entities)
            # Persist last entities for follow-up context
            self._persist_last_entities(entities, response)
            return response
        except Exception as e:
            self.logger.error(f"Error handling query with {agent_name}: {e}")
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
                            return response
                    except Exception as fallback_err:
                        self.logger.error("Fallback data_query failed: %s", fallback_err)
            return f"Sorry, I encountered an error: {str(e)}"

    def _initialize_agents(self):
        """Initialize all agents with shared resources."""
        self.agents["data_query"] = DataQueryAgent(self.shared_resources, self.streaming)
        self.agents["model_explanation"] = ModelExplanationAgent(self.shared_resources, self.streaming)
        self.agents["data_plotting"] = DataPlottingAgent(self.shared_resources, self.streaming)
        self.agents["general_qa"] = GeneralQAAgent(self.shared_resources, self.streaming)
        self.agents["modelling_suggestions"] = ModellingSuggestionsAgent(self.shared_resources, self.streaming)
        self.logger.debug("All agents initialized successfully.")

    def route_query(self, query: str, history: Optional[List[Tuple[str, str]]] = None) -> str:
        """Route the query to the appropriate agent using LLM-based classification."""
        self.turn_counter = getattr(self, "turn_counter", 0) + 1
        self.current_turn = self.turn_counter

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
