import logging
import os
import re
from typing import List, Tuple, Optional, Dict, Any
from pathlib import Path

from langchain.schema import Document
from langchain_core.retrievers import BaseRetriever
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from llm_factory import get_chat_openai as ChatOpenAI

from llm_config import QA_MODEL
from model_aliases import is_presentable_model_label


HISTORY_ANSWER_MAX_CHARS = 300

_REFERS_BACK = re.compile(
    r"\b(?:it|its|this|that|these|those|they|them|their|there|above|previous|"
    r"same|also|else|more|what\s+about|how\s+about|and\s+for)\b",
    re.IGNORECASE,
)


def _compact_history_answer(answer: object, max_chars: int = HISTORY_ANSWER_MAX_CHARS) -> str:
    """Past answer reduced to its prose: tables and links dropped, length capped."""
    lines = [
        line for line in str(answer or "").splitlines()
        if line.strip() and not line.lstrip().startswith("|")
    ]
    text = re.sub(r"!?\[([^\]]*)\]\([^)]*\)", r"\1", " ".join(lines))
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > max_chars:
        text = text[:max_chars].rsplit(" ", 1)[0] + " …"
    return text


class BaseAgent:
    def __init__(self, shared_resources: Dict[str, Any], streaming: bool = True):
        self.resources = shared_resources
        self.streaming = streaming
        self.logger = logging.getLogger(self.__class__.__name__)

    def handle(self, query: str, history: Optional[List[Tuple[str, str]]] = None) -> str:
        raise NotImplementedError("handle method must be implemented by subclasses")


class DataQueryAgent(BaseAgent):
    """Agent for querying IAM PARIS data using LLM intelligence."""
    
    def __init__(self, shared_resources: Dict[str, Any], streaming: bool = True):
        super().__init__(shared_resources, streaming)
        # Data answers come from the deterministic data_utils pipeline.

    def handle(self, query: str, history: Optional[List[Tuple[str, str]]] = None) -> str:
        from data_utils import data_query
        models = self.resources.get("models", [])
        ts = self.resources.get("ts", [])
        metadata = self.resources.get("metadata")
        return data_query(query, models, ts, history=history, metadata=metadata, allow_plots=False).strip()

    def handle_with_entities(
        self,
        query: str,
        entities: Dict[str, Any],
        history: Optional[List[Tuple[str, str]]] = None,
    ) -> str:
        from data_utils import data_query
        models = self.resources.get("models", [])
        ts = self.resources.get("ts", [])
        metadata = self.resources.get("metadata")
        return data_query(
            query,
            models,
            ts,
            history=history,
            forced_entities=entities,
            metadata=metadata,
            allow_plots=False,
        ).strip()


class ModelExplanationAgent(BaseAgent):
    def __init__(self, shared_resources: Dict[str, Any], streaming: bool = True):
        super().__init__(shared_resources, streaming)
        # Model answers come from deterministic model metadata.

    def handle(self, query: str, history: Optional[List[Tuple[str, str]]] = None) -> str:
        from data_utils import data_query
        models = self.resources.get("models", [])
        ts = self.resources.get("ts", [])
        metadata = self.resources.get("metadata")
        # Charts live in the IAM PARIS data explorer; never render one here.
        return data_query(query, models, ts, history=history, metadata=metadata, allow_plots=False).strip()


class DataPlottingAgent(BaseAgent):
    def handle(self, query: str, history: Optional[List[Tuple[str, str]]] = None) -> str:
        # Use the plotting function directly instead of data_query
        from simple_plotter import simple_plot_query
        models = self.resources.get("models", [])
        ts = self.resources.get("ts", [])
        return simple_plot_query(query, models, ts)

    def handle_with_entities(self, query: str, entities: Dict[str, Any], history: Optional[List[Tuple[str, str]]] = None) -> str:
        """
        Handle plotting with pre-extracted entities for better accuracy.
        """
        from simple_plotter import simple_plot_query_with_entities, simple_plot_query
        from data_utils import sanitize_variable_for_query
        models = self.resources.get("models", [])
        ts = self.resources.get("ts", [])
        sanitized = dict(entities or {})
        # Keyword sanitization cross-checks an *extracted* variable against the
        # words of the query. A structured comparison built from the resolved
        # scope of previous turns (e.g. "plot both models together") carries a
        # trusted variable that the follow-up text never names, so checking it
        # against that text would wrongly discard it.
        structured_comparison = bool(
            sanitized.get("comparison")
            and (
                sanitized.get("variables")
                or sanitized.get("models")
                or sanitized.get("regions")
                or sanitized.get("scenarios")
            )
        )
        if sanitized.get("variable") and not structured_comparison:
            sanitized["variable"] = sanitize_variable_for_query(sanitized["variable"], query)

        if not sanitized.get("variable") and not sanitized.get("variables") and not sanitized.get("models"):
            return simple_plot_query(query, models, ts)

        return simple_plot_query_with_entities(query, models, ts, sanitized)

    def handle_clarification(self, query: str, context: Dict[str, Any], history: Optional[List[Tuple[str, str]]] = None) -> str:
        """
        Handle clarification responses for ambiguous queries.
        """
        # Extract the specific variable from the clarification
        clarification_lower = query.lower().strip()

        # Get the original ambiguous matches from context
        original_response = context.get('ambiguous_response', '')
        if 'matched multiple variables' in original_response:


            # Re-run the plot with the clarified variable
            models = self.resources.get("models", [])
            ts = self.resources.get("ts", [])

            # Import the plotting function
            from simple_plotter import simple_plot_query
            return simple_plot_query(context['original_query'], models, ts)
        else:
            return (
                "I couldn't understand your clarification. Here are some tips:\n\n"
                "**For energy variables, try specifying:**\n"
                "- 'solar PV' or 'photovoltaic capacity'\n"
                "- 'wind power' or 'wind capacity'\n"
                "- 'total electricity' or 'power generation'\n\n"
                "**For regions, try:**\n"
                "- Country names: 'Germany', 'China', 'United States'\n"
                "- Regions: 'Europe', 'Asia', 'OECD & EU'\n\n"
                "Or try rephrasing your original request with more specific terms."
            )


QA_MAX_OUTPUT_TOKENS = 350

# Short and grounded on purpose. The previous prompt injected ~4,000 chars of
# developer-facing skill notes and asked for links; a small local model copied
# that structure into answers and invented URLs.
GENERAL_QA_SYSTEM_PROMPT = """You are the IAM PARIS assistant (https://iamparis.eu), helping people explore integrated assessment model (IAM) results on climate and energy.

Answer the question in 2-5 plain sentences.
Rules:
- Base the answer on the context below whenever it covers the question; prefer its wording for definitions.
- Never invent numbers, model names, study names or web links. The only link you may give is https://iamparis.eu/results.
- For IAM PARIS data values, tell the user to ask for a variable and region, for example "CO2 emissions for World in 2050".
- If the context does not cover the question and you are not sure, say so briefly.
- Do not name specific models, studies or variable codes unless the question asks about them.
- When the context starts with an "Explanation", build the answer on it and do not contradict it.

Context:
{context}"""


class _CuratedFirstRetriever(BaseRetriever):
    """Matching curated explanations, or similarity retrieval without one.

    A matched explanation is given on its own: any retrieved document next to
    it (removal text beside CCS, carbon pricing beside electricity prices)
    was blended into the small model's answer and made it wrong.
    """

    base: Any
    k: int = 4

    def _get_relevant_documents(self, query: str, *, run_manager: Any = None) -> List[Document]:
        from explanations import matching_explanations

        curated = matching_explanations(query)
        if curated:
            return curated
        return self.base.invoke(query)[:self.k]


class GeneralQAAgent(BaseAgent):
    def __init__(self, shared_resources: Dict[str, Any], streaming: bool = True):
        super().__init__(shared_resources, streaming)
        # Built lazily: a missing vector store must not crash agent/manager
        # initialization, only general-QA answers.
        self.chain = None

    def _ensure_chain(self) -> ConversationalRetrievalChain:
        if self.chain is None:
            self.chain = self._create_qa_chain()
        return self.chain

    def _create_qa_chain(self) -> ConversationalRetrievalChain:
        vs = self.resources.get("vector_store")
        if not vs:
            raise ValueError("Vector store not found in shared resources")

        llm = ChatOpenAI(
            model_name=QA_MODEL,
            temperature=0,
            streaming=self.streaming,
            timeout=30,
            max_retries=1,
            # Bounded answers: long generations are slow on CPU and small
            # models drift into unsupported detail.
            max_tokens=QA_MAX_OUTPUT_TOKENS,
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessagePromptTemplate.from_template(GENERAL_QA_SYSTEM_PROMPT),
                HumanMessagePromptTemplate.from_template("{question}"),
            ]
        )

        retriever = _CuratedFirstRetriever(
            base=vs.as_retriever(search_type="similarity", search_kwargs={"k": 4}),
        )

        # No internal memory: the session's chat_history (passed to handle) is
        # the single source of conversation state, so the two never diverge.
        return ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            chain_type="stuff",
            combine_docs_chain_kwargs={"prompt": prompt},
            verbose=False,
        )

    def handle(self, query: str, history: Optional[List[Tuple[str, str]]] = None) -> str:
        if history is None:
            history = []
        try:
            chain = self._ensure_chain()
        except ValueError:
            return (
                "General Q&A is unavailable because the knowledge index is not loaded. "
                "You can still ask data questions like `show CO2 emissions for Europe`."
            )
        # Keep only the recent turns, and only the prose of past answers, to
        # bound the prompt: the question-rewrite step reads the whole history
        # and data answers can carry multi-kilobyte tables.
        # A self-contained question gets no history: the chain then skips its
        # question-rewrite LLM call, which made a small model turn unrelated
        # input into a question about the previous turn (and cost a call).
        refers_back = bool(_REFERS_BACK.search(str(query or "")))
        compact_history = [
            (question, _compact_history_answer(answer))
            for question, answer in history[-6:]
        ] if refers_back else []
        resp = chain.invoke({"question": query, "chat_history": compact_history})
        return resp.get("answer", "").strip()


class ModellingSuggestionsAgent(BaseAgent):
    DEFAULT_SUGGESTIONS = [
        "Explore the impact of different carbon pricing scenarios on emission reductions.",
        "Investigate the role of renewable energy adoption in achieving climate targets.",
        "Analyze the effects of land-use changes on greenhouse gas emissions.",
        "Study the implications of energy efficiency improvements across sectors.",
        "Examine the potential of negative emissions technologies in climate mitigation pathways.",
        "Assess the outcomes of different policy mixes on achieving net-zero targets.",
    ]

    TOPIC_SUGGESTIONS = {
        "transport": (
            ("transport", "transportation", "mobility", "vehicle", "vehicles", "freight", "aviation", "shipping"),
            [
                "Compare transport final energy demand across scenarios (try `show Final Energy|Transportation for Europe`).",
                "Investigate electrification of road transport and its effect on transport CO2 emissions.",
                "Compare transport decarbonisation pathways between models (e.g. `compare GCAM and REMIND for transport emissions`).",
            ],
        ),
        "buildings": (
            ("building", "buildings", "residential", "commercial", "heating", "cooling"),
            [
                "Study energy demand reductions in the buildings sector across scenarios.",
                "Investigate heat-pump and electrification uptake in residential energy use.",
                "Compare buildings final energy between regions (try `show Final Energy|Residential and Commercial`).",
            ],
        ),
        "industry": (
            ("industry", "industrial", "steel", "cement", "manufacturing"),
            [
                "Analyze industrial emissions pathways under different policy scenarios.",
                "Investigate fuel switching and efficiency improvements in industry.",
                "Compare industrial final energy demand across models and regions.",
            ],
        ),
        "land use": (
            ("land", "land-use", "afolu", "forest", "forestry", "agriculture", "agricultural", "crop"),
            [
                "Analyze the effects of land-use changes on greenhouse gas emissions.",
                "Investigate afforestation and land-based mitigation potential across scenarios.",
                "Explore the AFOLU transformation results workspace for sectoral pathways.",
            ],
        ),
        "power and renewables": (
            ("electricity", "power", "renewable", "renewables", "solar", "wind", "grid"),
            [
                "Investigate the role of renewable energy adoption in achieving climate targets (try `compare solar and wind capacity for EU`).",
                "Compare electricity generation mixes across scenarios and models.",
                "Study the pace of coal phase-out in power generation under different policies.",
            ],
        ),
        "emissions and carbon pricing": (
            ("emission", "emissions", "co2", "ghg", "carbon", "price", "pricing", "tax"),
            [
                "Explore the impact of different carbon pricing scenarios on emission reductions.",
                "Compare CO2 emissions pathways between scenarios (try `show Emissions|CO2 for World under Baseline`).",
                "Examine the potential of negative emissions technologies in mitigation pathways.",
            ],
        ),
    }

    def handle(self, query: str, history: Optional[List[Tuple[str, str]]] = None) -> str:
        q = str(query or "").lower()
        matched_topics: List[str] = []
        suggestions: List[str] = []
        for topic, (keywords, topic_suggestions) in self.TOPIC_SUGGESTIONS.items():
            if any(keyword in q for keyword in keywords):
                matched_topics.append(topic)
                for suggestion in topic_suggestions:
                    if suggestion not in suggestions:
                        suggestions.append(suggestion)

        if matched_topics:
            intro = (
                "Here are modelling study suggestions related to "
                + ", ".join(matched_topics)
                + ":"
            )
        else:
            intro = "Here are some modelling study suggestions you could explore:"
            suggestions = list(self.DEFAULT_SUGGESTIONS)

        lines = [intro, ""]
        for idx, suggestion in enumerate(suggestions[:6], 1):
            lines.append(f"{idx}. {suggestion}")

        # Ground the answer in what the loaded data actually covers.
        metadata = self.resources.get("metadata")
        if metadata is not None and hasattr(metadata, "models_covering_topic"):
            try:
                category, models = metadata.models_covering_topic(query)
                if category and models:
                    lines.append("")
                    lines.append(
                        f"{len(models)} IAM PARIS model(s) report {category.lower()} variables — "
                        f"ask `which models cover {category.lower()}?` to see them."
                    )
            except Exception:
                pass

        lines.append("")
        lines.append("Explore the underlying data at https://iamparis.eu/results.")
        return "\n".join(lines)
