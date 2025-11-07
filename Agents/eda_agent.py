# In agents/eda_agent.py

from agents.base_agent import BaseAgent
from tools import data_tools, viz_tools
from typing import Dict, Any
import os

class EDAAgent(BaseAgent):
    """Agent for performing Exploratory Data Analysis."""
    def __init__(self, plot_dir: str = "plots"):
        super().__init__(name="EDAAgent", description="Performs EDA using data and viz tools.")
        self.plot_dir = plot_dir
        os.makedirs(self.plot_dir, exist_ok=True)

    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        self.log_message("Starting EDA...", state)
        
        # Use cleaned data if it exists, otherwise use raw
        df = state.get('processed_data', state.get('raw_data'))
        if df is None:
            raise ValueError("No data available for EDA.")
            
        # 1. Call the insight tool
        insights = data_tools.generate_eda_insights(df, state['target_column'])
        
        # 2. Call the visualization tool
        plot_paths = []
        try:
            corr_plot_path = os.path.join(self.plot_dir, "correlation_matrix.png")
            saved_path = viz_tools.plot_correlation_matrix(
                df, 
                state['data_info']['numeric_columns'], 
                corr_plot_path
            )
            if saved_path:
                plot_paths.append(saved_path)
                self.log_message(f"Correlation matrix saved to {saved_path}", state)
        except Exception as e:
            self.log_message(f"Could not generate plot: {e}", state)
        
        # 3. Update state
        state['insights'] = insights
        state['visualizations'] = plot_paths
        
        self.log_message("EDA complete, insights generated.", state)
        self.mark_step_complete("eda", state)
        return state