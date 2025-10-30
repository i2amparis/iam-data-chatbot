import logging
from typing import Dict, Any, List, Tuple, Optional
from agents import BaseAgent, DataQueryAgent, ModelExplanationAgent, DataPlottingAgent, GeneralQAAgent, ModellingSuggestionsAgent


class MultiAgentManager:
    def __init__(self, shared_resources: Dict[str, Any], streaming: bool = True):
        self.shared_resources = shared_resources
        self.streaming = streaming
        self.logger = logging.getLogger(__name__)
        self.agents: Dict[str, BaseAgent] = {}
        self._initialize_agents()

    def _initialize_agents(self):
        """Initialize all agents with shared resources."""
        self.agents["data_query"] = DataQueryAgent(self.shared_resources, self.streaming)
        self.agents["model_explanation"] = ModelExplanationAgent(self.shared_resources, self.streaming)
        self.agents["data_plotting"] = DataPlottingAgent(self.shared_resources, self.streaming)
        self.agents["general_qa"] = GeneralQAAgent(self.shared_resources, self.streaming)
        self.agents["modelling_suggestions"] = ModellingSuggestionsAgent(self.shared_resources, self.streaming)
        self.logger.info("All agents initialized successfully.")

    def route_query(self, query: str, history: Optional[List[Tuple[str, str]]] = None) -> str:
        """Route the query to the appropriate agent based on keywords."""
        query_lower = query.lower()

        # Check for clarification responses first
        if hasattr(self, 'clarification_context') and self.clarification_context:
            # This is a follow-up to a clarification request
            context = self.clarification_context
            if context['agent_type'] == 'data_plotting':
                response = self.agents["data_plotting"].handle_clarification(query, context, history)
                # Clear context after handling
                self.clarification_context = None
                return response

        # Enhanced routing logic to detect data queries
        plotting_keywords = ["plot", "graph", "visualize", "chart", "give me a plot", 
                             "create a plot", "make a plot"]
        data_listing_keywords = ["list models", "list variables", "list scenarios", 
                                 "available models", "available variables", "available scenarios", "what models", 
                                 "what variables", "what scenarios", "what are the models", "what are the variables", 
                                 "what are the scenarios", "what scenarios are there", "tell me the models", 
                                 "tell me the variables", "tell me the scenarios", "what data", "what can you plot", 
                                 "what can you graph", "what can you visualize", "what plots", "what graphs", 
                                 "what charts", "show me variables", "show me models", "show me scenarios"]

        # Check for data variable queries (contains variable-like terms + location)
        data_variable_keywords = ["capacity", "generation", "production", "emissions", "energy", 
                                  "electricity", "power", "solar", "wind", "gas", "coal", "nuclear", 
                                  "hydro", "biomass", "co2", "carbon", "greenhouse"]
        location_keywords = ["greece", "europe", "china", "india", "usa", "united states", "germany", 
                             "france", "japan", "russia", "brazil", "africa", "asia", "global", "world"]

        has_data_keywords = any(word in query_lower for word in data_variable_keywords)
        has_location = any(loc in query_lower for loc in location_keywords)
        has_for = "for" in query_lower or "in" in query_lower

        # Check if this is a data listing request (not plotting)
        is_data_listing = any(phrase in query_lower for phrase in data_listing_keywords)

        # Check if this contains plotting keywords but also data listing intent
        has_plotting_words = any(word in query_lower for word in plotting_keywords)
        has_show_me = "show me" in query_lower

        # Route to data_query for: data listings, variable queries with locations, or direct variable names
        if is_data_listing or (has_data_keywords and (has_location or has_for)) or "|" in query:
            agent_name = "data_query"
        # Special case: "show me" + data listing keywords should go to data_query, not plotting
        elif has_show_me and any(phrase in query_lower for phrase in ["models", "variables", "scenarios"]):
            agent_name = "data_query"
        elif has_show_me and any(word in query_lower for word in ["emissions", "data"]):
            # "show me emissions data" should go to plotting, not data_query
            agent_name = "data_plotting"
        elif has_plotting_words and not is_data_listing:
            agent_name = "data_plotting"
            response = self.agents[agent_name].handle(query, history)
            # Check if response indicates ambiguity that needs clarification
            if "Please clarify" in response or "matched multiple" in response:
                # Store clarification context for follow-up
                self.clarification_context = {
                    'original_query': query,
                    'ambiguous_response': response,
                    'agent_type': agent_name
                }
                return response
            return response
        elif "explain carbon pricing" in query_lower:
            agent_name = "general_qa"  # Route to LLM for carbon pricing explanation
        elif any(phrase in query_lower for phrase in ["explain", "describe", "info about", "details about", "tell me about", "what is"]):
            agent_name = "model_explanation"
        elif "suggest modelling studies" in query_lower:
            agent_name = "modelling_suggestions"  # Route to suggestions agent
        elif any(word in query_lower for word in ["suggest", "recommend", "modelling suggestions"]):
            agent_name = "general_qa"  # Route other suggestions to LLM
        else:
            agent_name = "general_qa"

        self.logger.info(f"Routing query to {agent_name} agent.")
        agent = self.agents.get(agent_name)
        if not agent:
            return "Sorry, the requested agent is not available."

        try:
            response = agent.handle(query, history)
            return response
        except Exception as e:
            self.logger.error(f"Error handling query with {agent_name}: {e}")
            return f"Sorry, I encountered an error: {str(e)}"

    def get_agent_names(self) -> List[str]:
        """Return the list of available agent names."""
        return list(self.agents.keys())
