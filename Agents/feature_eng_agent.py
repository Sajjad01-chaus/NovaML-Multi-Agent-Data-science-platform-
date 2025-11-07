# In agents/feature_engineering_agent.py

from agents.base_agent import BaseAgent
from tools import ml_tools
from typing import Dict, Any

class FeatureEngineeringAgent(BaseAgent):
    """Agent for creating new features."""
    def __init__(self):
        super().__init__(name="FeatureEngineeringAgent", description="Performs FE using ml_tools.")

    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        self.log_message("Starting feature engineering...", state)
        
        # Use cleaned data if it exists, otherwise use raw
        df = state.get('processed_data', state.get('raw_data'))
        if df is None:
            raise ValueError("No data available for Feature Engineering.")
        
        # 1. Call the encoding tool
        encoded_df, new_features = ml_tools.encode_features(
            df, 
            state['data_info']['categorical_columns'], 
            state['target_column']
        )
        
        if new_features:
            self.log_message(f"Created {len(new_features)} new features via one-hot encoding.", state)
        
        # 2. Update state
        # This now becomes the *final* processed data for training
        state['processed_data'] = encoded_df 
        state['engineered_features'] = new_features
        
        self.mark_step_complete("feature_engineering", state)
        return state