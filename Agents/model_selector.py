# In agents/model_selection_agent.py

from agents.base_agent import BaseAgent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_groq import ChatGroq
from typing import Dict, Any, List
import os
import pandas as pd

# --- This is the LLM-powered "brain" ---
# Make sure GROQ_API_KEY is set in your environment variables
llm = ChatGroq(
    temperature=0, 
    model_name="llama3-70b-8192",
    api_key=os.environ.get("GROQ_API_KEY") 
)

class ModelRecommendations(BaseModel):
    """The output format for model recommendations."""
    recommended_models: List[str] = Field(description="A list of 3 model names, e.g., ['RandomForest', 'LinearRegression']")
    reasoning: str = Field(description="A justification for why these models were chosen.")

class ModelSelectionAgent(BaseAgent):
    """Agent that uses an LLM to select models."""
    def __init__(self):
        super().__init__(
            name="ModelSelectionAgent",
            description="Recommends models by reasoning over data profile."
        )
        self.llm_with_tools = llm.with_structured_output(ModelRecommendations)
        
        # This prompt is "constrained" to only recommend models we actually have in ml_tools.py
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", 
             "You are a principal data scientist. Your job is to recommend 3 ML models. "
             "Base your recommendations on the data profile and problem type."
             "Your recommendations MUST be chosen from the following lists:\n"
             "Classification: ['LogisticRegression', 'RandomForest', 'GradientBoosting']\n"
             "Regression: ['LinearRegression', 'RandomForest', 'GradientBoosting']"
            ),
            ("human", 
             "Here is the data profile: \n"
             "Problem Type: {problem_type}\n"
             "Data Info: {data_info}\n"
             "EDA Insights: {insights}\n\n"
             "Please provide your 3 model recommendations and the reasoning."
            )
        ])
        self.chain = self.prompt | self.llm_with_tools

    def _auto_detect_problem_type(self, state: Dict[str, Any]) -> str:
        """Detects problem type if not specified by user."""
        target_col = state['target_column']
        df = state['raw_data']
        
        # Simple heuristic: if target is numeric and has > 20 unique values, it's regression.
        if pd.api.types.is_numeric_dtype(df[target_col]) and df[target_col].nunique() > 20:
            problem_type = "regression"
        else:
            problem_type = "classification"
            
        self.log_message(f"Auto-detected problem type: {problem_type}", state)
        return problem_type

    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        self.log_message("Using LLM to reason about model selection...", state)
        
        problem_type = state.get("problem_type")
        if not problem_type or problem_type == "auto":
            problem_type = self._auto_detect_problem_type(state)
            state['problem_type'] = problem_type

        insights_str = "\n".join(state.get('insights', []))
        
        try:
            # Call the LLM chain
            response = self.chain.invoke({
                "data_info": state['data_info'],
                "insights": insights_str,
                "problem_type": state['problem_type']
            })
            state["recommended_models"] = response.recommended_models
            state["model_reasoning"] = response.reasoning
            self.log_message(f"LLM recommended: {', '.join(response.recommended_models)}", state)
            
        except Exception as e:
            # Fallback logic if LLM fails
            self.log_message(f"LLM call failed ({e}). Falling back to default models.", state)
            if problem_type == "classification":
                models = ["LogisticRegression", "RandomForest", "GradientBoosting"]
            else:
                models = ["LinearRegression", "RandomForest", "GradientBoosting"]
            state["recommended_models"] = models
            state["model_reasoning"] = "Fell back to default models due to an LLM error."
        
        self.mark_step_complete("model_selection", state)
        return state