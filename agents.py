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
        self.chain = self._create_qa_chain()

    def _create_qa_chain(self) -> ConversationalRetrievalChain:
        vs = self.resources.get("vector_store")
        if not vs:
            raise ValueError("Vector store not found in shared resources")
        
        # Get all available data for direct LLM access
        models = self.resources.get("models", [])
        ts = self.resources.get("ts", [])
        
        model_names = sorted([m.get('modelName', '') for m in models if m and m.get('modelName')])
        scenarios = sorted({r.get('scenario', '') for r in ts if r and r.get('scenario')})
        variables = sorted({str(r.get('variable', '')) for r in ts if r and r.get('variable')})
        regions = sorted({str(r.get('region', '')) for r in ts if r and r.get('region')})
        
        # Create concise summaries instead of full lists
        model_list = ", ".join(model_names[:20]) + (f" ... and {len(model_names)-20} more" if len(model_names) > 20 else "")
        scenario_list = ", ".join(scenarios[:15]) + (f" ... and {len(scenarios)-15} more" if len(scenarios) > 15 else "")
        variable_list = ", ".join(variables[:20]) + (f" ... and {len(variables)-20} more" if len(variables) > 20 else "")
        region_list = ", ".join(regions[:15]) + (f" ... and {len(regions)-15} more" if len(regions) > 15 else "")
        
        llm = ChatOpenAI(
            model_name="gpt-4-turbo",
            temperature=0,
            streaming=self.streaming,
            api_key=self.resources["env"]["OPENAI_API_KEY"],
        )

        message_history = ChatMessageHistory()
        memory = ConversationSummaryBufferMemory(
            llm=llm,
            max_token_limit=1000,
            chat_memory=message_history,
            return_messages=True,
            memory_key="chat_history"
        )

        system_tpl = f"""You are a data query assistant for IAM PARIS climate data (https://iamparis.eu/).

## Available Data Summary:

- **Models:** {len(model_names)} total - Examples: {model_list}
- **Scenarios:** {len(scenarios)} total - Examples: {scenario_list}
- **Variables:** {len(variables)} total - Examples: {variable_list}
- **Regions:** {len(regions)} total - Examples: {region_list}

## Your Task:

1. Answer questions about what data is available
2. Use the vector store context to find specific items
3. Provide counts and examples when asked

## Guidelines:

- For "which/what/list models": Provide count and list from context
- For "which/what/list scenarios": Provide count and examples
- For "which/what/list variables": Provide count and relevant examples
- For "which/what/list regions": Provide count and examples
- Use Markdown formatting
- Reference https://iamparis.eu/results for data access

Context from vector store: ```{{context}}```"""

        user_tpl = "Question: ```{question}```"

        prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessagePromptTemplate.from_template(system_tpl),
                HumanMessagePromptTemplate.from_template(user_tpl),
            ]
        )

        retriever = vs.as_retriever(search_type="mmr", search_kwargs={"k": 5, "fetch_k": 20, "lambda_mult": 0.5})

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


class ModelExplanationAgent(BaseAgent):
    def __init__(self, shared_resources: Dict[str, Any], streaming: bool = True):
        super().__init__(shared_resources, streaming)
        self.chain = self._create_qa_chain()

    def _create_qa_chain(self) -> ConversationalRetrievalChain:
        vs = self.resources.get("vector_store")
        if not vs:
            raise ValueError("Vector store not found in shared resources")
        
        # Get all model names for the system prompt
        models = self.resources.get("models", [])
        model_names = sorted([m.get('modelName', '') for m in models if m and m.get('modelName')])
        model_list = ", ".join(model_names)
        
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

        system_tpl = f"""You are an expert climate policy assistant focused on IAM PARIS data and models (https://iamparis.eu/).

Available models in IAM PARIS database ({len(model_names)} total):
{model_list}

When users ask about models:
- List ALL models by name when asked to list models
- Provide details about specific models using the modelName field
- Match user queries to the correct modelName

Always:
- Provide direct answers without restating the question
- Use Markdown formatting for responses with proper headers (##) and lists (-)
- Reference specific IAM PARIS data points when available
- Clearly indicate when information comes from external sources
- Include relevant IAM PARIS links when referencing specific studies
- Format numerical values with proper units
- Keep answers focused and data-driven

Available IAM PARIS resources:
- Results database: https://iamparis.eu/results

Context: ```{{context}}```"""

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
        # Use the plotting function directly instead of data_query
        from simple_plotter import simple_plot_query
        models = self.resources.get("models", [])
        ts = self.resources.get("ts", [])
        return simple_plot_query(query, models, ts)

    def handle_with_entities(self, query: str, entities: Dict[str, Any], history: Optional[List[Tuple[str, str]]] = None) -> str:
        """
        Handle plotting with pre-extracted entities for better accuracy.
        """
        from simple_plotter import simple_plot_query_with_entities
        models = self.resources.get("models", [])
        ts = self.resources.get("ts", [])
        return simple_plot_query_with_entities(query, models, ts, entities)

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


class GeneralQAAgent(BaseAgent):
    def __init__(self, shared_resources: Dict[str, Any], streaming: bool = True):
        super().__init__(shared_resources, streaming)
        self.chain = self._create_qa_chain()

    def _create_qa_chain(self) -> ConversationalRetrievalChain:
        vs = self.resources.get("vector_store")
        if not vs:
            raise ValueError("Vector store not found in shared resources")
        
        # Get all model names for the system prompt
        models = self.resources.get("models", [])
        model_names = sorted([m.get('modelName', '') for m in models if m and m.get('modelName')])
        model_list = ", ".join(model_names)
        
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

        system_tpl = f"""You are an expert climate policy assistant focused on IAM PARIS data and models (https://iamparis.eu/).

Available models in IAM PARIS database ({len(model_names)} total):
{model_list}

When users ask about models:
- List ALL models by name when asked to list models
- Provide details about specific models using the modelName field

Always:
- Provide direct answers without restating the question
- Use Markdown formatting with headers and lists
- Reference IAM PARIS data when available
- Include IAM PARIS links
- Format numbers with units
- Promote https://iamparis.eu/results for detailed data access

Context: ```{{context}}```"""

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
        # Always promote the results page
        for i, suggestion in enumerate(suggestions):
            if "https://iamparis.eu/results" not in suggestion:
                suggestions[i] = suggestion.replace("https://iamparis.eu/results", "https://iamparis.eu/results")
        response = "Here are some modelling study suggestions you could explore:\n\n"
        for idx, suggestion in enumerate(suggestions, 1):
            response += f"{idx}. {suggestion}\n"
        return response
