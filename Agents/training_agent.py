# In agents/training_agent.py

from agents.base_agent import BaseAgent
from typing import Dict, Any
import os

# --- This is the key change! ---
# We import the "workers" from our tools directory
from tools import ml_tools 

class TrainingAgent(BaseAgent):
    """Agent responsible for training ML models"""
    
    def __init__(self, model_dir: str = "models"):
        super().__init__(
            name="TrainingAgent",
            description="Trains multiple ML models using tools."
        )
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)

    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Manages the model training process."""
        
        self.log_message("Training process starting...", state)
        
        # 1. Get data from state
        df = state.get("processed_data")
        target_col = state.get("target_column")
        problem_type = state.get("problem_type")
        
        # --- HITL FIX ---
        # Use the human-approved models, not the recommended ones
        models_to_train = state.get("user_approved_models") 
        if not models_to_train:
            # Fallback just in case HITL was skipped (e.g., in CLI mode)
            models_to_train = state.get("recommended_models", [])

        if df is None or not target_col:
            raise ValueError("Processed data or target column not available")

        # 2. Call the data splitting tool
        self.log_message("Splitting data into train/test sets.", state)
        X_train, X_test, y_train, y_test = ml_tools.split_data(df, target_col)
        
        trained_models_data = {}
        training_logs_data = {}
        
        # 3. Loop and call the training tool
        for model_name in models_to_train:
            self.log_message(f"Training {model_name}...", state)
            
            # Get a model instance
            model = ml_tools.get_model_instance(model_name, problem_type)
            
            # Call the training tool
            trained_model, log = ml_tools.train_single_model(model, X_train, y_train)
            
            # Store results
            training_logs_data[model_name] = log
            trained_models_data[model_name] = {
                'model': trained_model,
                'X_train': X_train, 'X_test': X_test,
                'y_train': y_train, 'y_test': y_test
            }
            
            self.log_message(f"{model_name} CV score: {log['mean_cv_score']:.4f}", state)

        # 4. Update the state
        state["trained_models"] = trained_models_data
        state["training_logs"] = training_logs_data
        state["current_step"] = "training"
        self.mark_step_complete("training", state)
        
        return state