import logging
from typing import Dict, Any, List, Tuple, Optional
from agents import BaseAgent, DataQueryAgent, ModelExplanationAgent, DataPlottingAgent, GeneralQAAgent, ModellingSuggestionsAgent
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from query_extractor import QueryEntityExtractor


class MultiAgentManager:
    def __init__(self, shared_resources: Dict[str, Any], streaming: bool = True):
        self.shared_resources = shared_resources
        self.streaming = streaming
        self.logger = logging.getLogger(self.__class__.__name__)
        self.agents: Dict[str, BaseAgent] = {}
        self._initialize_agents()

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
        self.routing_prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template("""You are a query classifier for an IAM PARIS climate data chatbot.

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

    Question: {query}
    Answer:"""),
                HumanMessagePromptTemplate.from_template("Query: {query}")
            ])

    def _initialize_agents(self):
        """Initialize all agents with shared resources."""
        self.agents["data_query"] = DataQueryAgent(self.shared_resources, self.streaming)
        self.agents["model_explanation"] = ModelExplanationAgent(self.shared_resources, self.streaming)
        self.agents["data_plotting"] = DataPlottingAgent(self.shared_resources, self.streaming)
        self.agents["general_qa"] = GeneralQAAgent(self.shared_resources, self.streaming)
        self.agents["modelling_suggestions"] = ModellingSuggestionsAgent(self.shared_resources, self.streaming)
        self.logger.info("All agents initialized successfully.")

    def route_query(self, query: str, history: Optional[List[Tuple[str, str]]] = None) -> str:
        """Route the query to the appropriate agent using LLM-based classification."""
        
        # Check for clarification responses first
        if hasattr(self, 'clarification_context') and self.clarification_context:
            context = self.clarification_context
            if context['agent_type'] == 'data_plotting':
                response = self.agents["data_plotting"].handle_clarification(query, context, history)
                self.clarification_context = None
                return response

        # Extract entities from query using the new extractor
        try:
            entities = self.entity_extractor.extract(query)
            self.logger.info(f"Extracted entities: {entities}")
            
            # Use extracted action to determine routing
            if entities.get('action') == 'plot':
                agent_name = 'data_plotting'
            else:
                # Use LLM for more complex routing
                result = self.routing_prompt | self.router_llm
                response_obj = result.invoke({"query": query})
                agent_name = response_obj.content.strip().lower()
                
                # Validate agent name
                valid_agents = ["data_query", "data_plotting", "model_explanation", "general_qa", "modelling_suggestions"]
                if agent_name not in valid_agents:
                    agent_name = "general_qa"
                    
        except Exception as e:
            self.logger.error(f"Routing error: {e}")
            agent_name = "general_qa"
            entities = {}

        self.logger.info(f"Routing query to {agent_name} agent.")
        agent = self.agents.get(agent_name)
        if not agent:
            return "Sorry, the requested agent is not available."

        try:
            # Pass entities to agent if it supports them
            if hasattr(agent, 'handle_with_entities'):
                response = agent.handle_with_entities(query, entities, history)
            else:
                response = agent.handle(query, history)
            
            # Check if plotting response needs clarification
            if agent_name == "data_plotting" and ("Please clarify" in response or "matched multiple" in response):
                self.clarification_context = {
                    'original_query': query,
                    'ambiguous_response': response,
                    'agent_type': agent_name,
                    'entities': entities
                }
            return response
        except Exception as e:
            self.logger.error(f"Error handling query with {agent_name}: {e}")
            return f"Sorry, I encountered an error: {str(e)}"

    def get_agent_names(self) -> List[str]:
        """Return the list of available agent names."""
        return list(self.agents.keys())

