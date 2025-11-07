# In agents/base_agent.py

from typing import Dict, Any

class BaseAgent:
    """A base class for all our agents"""
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """The main execution method for the agent"""
        raise NotImplementedError("This method must be overridden by a subclass")

    def log_message(self, message: str, state: Dict[str, Any]):
        """A helper to log messages to the state"""
        log = f"[{self.name}]: {message}"
        print(log) # Also print to console for real-time tracking
        state['messages'].append(log)

    def mark_step_complete(self, step_name: str, state: Dict[str, Any]):
        """Marks a step as complete in the state"""
        if 'completed_steps' not in state:
            state['completed_steps'] = []
        state['completed_steps'].append(step_name)