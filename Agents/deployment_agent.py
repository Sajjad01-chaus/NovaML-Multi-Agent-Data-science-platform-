# In agents/deployment_agent.py

from agents.base_agent import BaseAgent
from tools import api_tools
from typing import Dict, Any
import os

class DeploymentAgent(BaseAgent):
    """Agent responsible for generating the deployment API."""
    
    def __init__(self, api_dir: str = "api"):
        super().__init__(
            name="DeploymentAgent",
            description="Generates a FastAPI deployment script."
        )
        self.api_dir = api_dir
        os.makedirs(self.api_dir, exist_ok=True)

    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        self.log_message("Generating deployment API...", state)
        
        # Get all the needed info from the state
        model_path = state.get("model_path")
        processed_data = state.get("processed_data")
        target_col = state.get("target_column")
        
        if not all([model_path, processed_data is not None, target_col]):
            self.log_message("Skipping deployment: missing model, data, or target.", state)
            return state

        # 1. Call the API generation tool
        api_file_path = os.path.join(self.api_dir, "main.py")
        saved_path = api_tools.write_api_file(
            model_path=model_path,
            processed_data=processed_data,
            target_col=target_col,
            api_file_path=api_file_path
        )
        
        # 2. Update state with the final artifact
        state["api_endpoint_file"] = saved_path
        state["deployment_status"] = "API file generated."
        
        self.log_message(f"Deployment API successfully generated at {saved_path}", state)
        self.mark_step_complete("deployment")
        return state