# In agents/data_ingestion_agent.py

from agents.base_agent import BaseAgent
from tools import data_tools
from typing import Dict, Any

class DataIngestionAgent(BaseAgent):
    """Agent for loading, validating, and profiling data."""
    def __init__(self):
        super().__init__(name="DataIngestionAgent", description="Loads and profiles data using data_tools.")

    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        self.log_message("Starting data ingestion...", state)
        
        # 1. Call the load tool
        df = data_tools.load_dataset(state['dataset_path'])
        
        # 2. Call the profile tool
        data_info, data_quality_report = data_tools.profile_data(df)
        
        # 3. Update state
        state['raw_data'] = df
        state['data_info'] = data_info
        state['data_quality_report'] = data_quality_report
        
        self.log_message(f"Data loaded: {data_info['total_rows']} rows.", state)
        if data_quality_report['total_missing_values'] > 0:
            self.log_message(f"Found {data_quality_report['total_missing_values']} missing values.", state)
            
        self.mark_step_complete("data_ingestion", state)
        return state