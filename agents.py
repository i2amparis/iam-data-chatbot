import logging
from typing import List, Tuple, Optional, Dict, Any

from langchain.schema import Document
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationSummaryBufferMemory
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain_openai import ChatOpenAI
from langchain_community.chat_message_histories import ChatMessageHistory

from data_utils import data_query  # Import the existing data_query function


class BaseAgent:
    def __init__(self, shared_resources: Dict[str, Any], streaming: bool = True):
        self.resources = shared_resources
        self.streaming = streaming
        self.logger = logging.getLogger(self.__class__.__name__)

    def handle(self, query: str, history: Optional[List[Tuple[str, str]]] = None) -> str:
        raise NotImplementedError("handle method must be implemented by subclasses")


class DataQueryAgent(BaseAgent):
    def handle(self, query: str, history: Optional[List[Tuple[str, str]]] = None) -> str:
        models = self.resources.get("models", [])
        ts = self.resources.get("ts", [])
        return data_query(query, models, ts)


class ModelExplanationAgent(BaseAgent):
    def __init__(self, shared_resources: Dict[str, Any], streaming: bool = True):
        super().__init__(shared_resources, streaming)
        self.chain = self._create_qa_chain()

    def _create_qa_chain(self) -> ConversationalRetrievalChain:
        vs = self.resources.get("vector_store")
        if not vs:
            raise ValueError("Vector store not found in shared resources")

        llm = ChatOpenAI(
            model_name="gpt-4-turbo",
            temperature=0,
            streaming=self.streaming,
            api_key=self.resources["env"]["OPENAI_API_KEY"],
        )

        message_history = ChatMessageHistory()
        memory = ConversationSummaryBufferMemory(
            llm=llm,
            max_token_limit=2000,
            chat_memory=message_history,
            return_messages=True,
            memory_key="chat_history"
        )

        system_tpl = """You are an expert climate policy assistant focused on IAM PARIS data and models (https://iamparis.eu/).

Always:
- Provide direct answers without restating the question
- Use Markdown formatting for responses with proper headers (##) and lists (-)
- Reference specific IAM PARIS data points when available 
- Clearly indicate when information comes from external sources
- Include relevant IAM PARIS links when referencing specific studies
- Format numerical values with proper units
- Keep answers focused and data-driven

Available IAM PARIS resources:
- Model documentation: https://iamparis.eu/models
- Results database: https://iamparis.eu/results
- Study descriptions: https://iamparis.eu/studies

Context: ```{context}```"""

        user_tpl = "Question: ```{question}```"

        prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessagePromptTemplate.from_template(system_tpl),
                HumanMessagePromptTemplate.from_template(user_tpl),
            ]
        )

        retriever = vs.as_retriever(search_type="similarity", search_kwargs={"k": 5})

        return ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=memory,
            chain_type="stuff",
            combine_docs_chain_kwargs={"prompt": prompt},
            verbose=False,
        )

    def handle(self, query: str, history: Optional[List[Tuple[str, str]]] = None) -> str:
        if history is None:
            history = []
        resp = self.chain.invoke({"question": query, "chat_history": history})
        return resp.get("answer", "").strip()


class DataPlottingAgent(BaseAgent):
    def handle(self, query: str, history: Optional[List[Tuple[str, str]]] = None) -> str:
        models = self.resources.get("models", [])
        ts = self.resources.get("ts", [])
        return data_query(query, models, ts)


class GeneralQAAgent(BaseAgent):
    def __init__(self, shared_resources: Dict[str, Any], streaming: bool = True):
        super().__init__(shared_resources, streaming)
        self.chain = self._create_qa_chain()

    def _create_qa_chain(self) -> ConversationalRetrievalChain:
        vs = self.resources.get("vector_store")
        if not vs:
            raise ValueError("Vector store not found in shared resources")

        llm = ChatOpenAI(
            model_name="gpt-4-turbo",
            temperature=0,
            streaming=True,
            api_key=self.resources["env"]["OPENAI_API_KEY"],
        )

        message_history = ChatMessageHistory()
        memory = ConversationSummaryBufferMemory(
            llm=llm,
            max_token_limit=2000,
            chat_memory=message_history,
            return_messages=True,
            memory_key="chat_history"
        )

        system_tpl = """You are an expert climate policy assistant focused on IAM PARIS data and models (https://iamparis.eu/).

Always:
- Provide direct answers without restating the question
- Use Markdown formatting with headers and lists
- Reference IAM PARIS data when available
- Include IAM PARIS links
- Format numbers with units

Context: ```{context}```"""

        user_tpl = "Question: ```{question}```"

        prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessagePromptTemplate.from_template(system_tpl),
                HumanMessagePromptTemplate.from_template(user_tpl),
            ]
        )

        retriever = vs.as_retriever(search_type="similarity", search_kwargs={"k": 5})

        return ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=memory,
            chain_type="stuff",
            combine_docs_chain_kwargs={"prompt": prompt},
            verbose=False,
        )

    def handle(self, query: str, history: Optional[List[Tuple[str, str]]] = None) -> str:
        if history is None:
            history = []
        resp = self.chain.invoke({"question": query, "chat_history": history})
        return resp.get("answer", "").strip()


class ModellingSuggestionsAgent(BaseAgent):
    def handle(self, query: str, history: Optional[List[Tuple[str, str]]] = None) -> str:
        suggestions = [
            "Explore the impact of different carbon pricing scenarios on emission reductions. See details at https://iamparis.eu/results.",
            "Investigate the role of renewable energy adoption in achieving climate targets. More info at https://iamparis.eu/results.",
            "Analyze the effects of land-use changes on greenhouse gas emissions. Relevant studies can be found at https://iamparis.eu/results.",
            "Study the implications of energy efficiency improvements across sectors. Visit https://iamparis.eu/results for related data.",
            "Examine the potential of negative emissions technologies in climate mitigation pathways. See https://iamparis.eu/results for studies.",
            "Assess the outcomes of different policy mixes on achieving net-zero targets. Explore https://iamparis.eu/results for modelling results."
        ]
        response = "Here are some modelling study suggestions you could explore:\n\n"
        for idx, suggestion in enumerate(suggestions, 1):
            response += f"{idx}. {suggestion}\n"
        return response
