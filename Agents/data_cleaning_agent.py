# In agents/data_cleaning_agent.py

from agents.base_agent import BaseAgent
from tools import data_tools
from typing import Dict, Any

class DataCleaningAgent(BaseAgent):
    """Agent for cleaning data."""
    def __init__(self):
        super().__init__(name="DataCleaningAgent", description="Cleans data using data_tools.")

    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        self.log_message("Data is dirty. Running cleaning agent...", state)
        
        df = state['raw_data']
        numeric_cols = state['data_info']['numeric_columns']
        categorical_cols = state['data_info']['categorical_columns']
        
        # 1. Call the clean tool
        cleaned_df = data_tools.clean_data(df, numeric_cols, categorical_cols)
        
        # 2. Update state
        state['processed_data'] = cleaned_df
        
        self.log_message("Data cleaning complete.", state)
        self.mark_step_complete("data_cleaning", state)
        return state