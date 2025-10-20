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

        # Routing logic based on keywords - more specific to avoid false positives
        # Check for actual plotting requests first (highest priority)
        if any(word in query_lower for word in ["plot", "show me", "graph", "visualize", "chart", "give me a plot", "create a plot", "make a plot"]):
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
        elif any(phrase in query_lower for phrase in ["list models", "list variables", "list scenarios", "available models", "available variables", "available scenarios", "what models", "what variables", "what scenarios", "what are the scenarios", "what scenarios are there"]) or \
              any(word in query_lower for word in ["what data", "what can you plot", "what can you graph", "what can you visualize", "what plots", "what graphs", "what charts"]):
            agent_name = "data_query"
        elif any(phrase in query_lower for phrase in ["explain", "describe", "info about", "details about", "tell me about", "what is"]):
            agent_name = "model_explanation"
        elif any(word in query_lower for word in ["suggest", "recommend", "modelling studies", "modelling suggestions"]):
            agent_name = "modelling_suggestions"
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
