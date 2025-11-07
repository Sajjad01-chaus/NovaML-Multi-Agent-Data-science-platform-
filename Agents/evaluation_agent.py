# In agents/evaluation_agent.py

from agents.base_agent import BaseAgent
from tools import ml_tools
from typing import Dict, Any
import os

class EvaluationAgent(BaseAgent):
    """Agent responsible for evaluating trained models."""
    
    def __init__(self, model_dir: str = "models"):
        super().__init__(
            name="EvaluationAgent",
            description="Evaluates models using ml_tools and selects the best."
        )
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)

    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        self.log_message("Evaluating models...", state)
        
        trained_models = state.get("trained_models", {})
        problem_type = state.get("problem_type")
        
        if not trained_models:
            raise ValueError("No trained models available to evaluate.")
            
        evaluation_results = {}
        
        # Define the primary metric for comparison
        primary_metric = "accuracy" if problem_type == "classification" else "r2_score"
        # -inf if maximizing (accuracy), +inf if minimizing (e.g., mse)
        best_score = -float('inf') 
        best_model_name = None
        
        for model_name, model_data in trained_models.items():
            model = model_data['model']
            X_test = model_data['X_test']
            y_test = model_data['y_test']
            
            y_pred = model.predict(X_test)
            
            # 1. Call the appropriate evaluation tool
            if problem_type == "classification":
                metrics = ml_tools.evaluate_classification(y_test, y_pred)
            else:
                metrics = ml_tools.evaluate_regression(y_test, y_pred)
                
            evaluation_results[model_name] = metrics
            score = metrics[primary_metric]
            
            # 2. Compare to find the best model
            if score > best_score:
                best_score = score
                best_model_name = model_name
                
            self.log_message(f"{model_name} - {primary_metric}: {score:.4f}", state)

        # 3. Save the best model
        best_model_obj = trained_models[best_model_name]['model']
        model_path = ml_tools.save_model(best_model_obj, best_model_name, self.model_dir)
        
        self.log_message(f"Best model found: {best_model_name}", state)
        
        # 4. Update state
        state["evaluation_results"] = evaluation_results
        state["best_model_name"] = best_model_name
        state["best_model"] = best_model_obj
        state["model_path"] = model_path
        state["metrics"] = evaluation_results[best_model_name]
        
        self.mark_step_complete("evaluation", state)
        return state