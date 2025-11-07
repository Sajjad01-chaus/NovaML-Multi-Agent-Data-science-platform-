# In orchestrator/state.py

from typing import TypedDict, Annotated, List, Dict, Any, Optional
import operator
from datetime import datetime

class DataScienceState(TypedDict):
    """State for the multi-agent data science workflow"""
    
    # Input
    dataset_path: str
    problem_type: Optional[str]
    target_column: Optional[str]
    
    # Data Ingestion
    raw_data: Optional[Any]
    data_info: Optional[Dict]
    data_quality_report: Optional[Dict]
    
    # EDA
    insights: Optional[List[str]]
    visualizations: Optional[List[str]]
    
    # Feature Engineering
    processed_data: Optional[Any]
    engineered_features: Optional[List[str]]
    
    # Model Selection
    recommended_models: Optional[List[str]]
    model_reasoning: Optional[str]
    
    # HITL
    user_approved_models: Optional[List[str]] 
    
    # Training
    trained_models: Optional[Dict[str, Any]]
    training_logs: Optional[Dict]
    best_params: Optional[Dict] # For optuna (if added)
    
    # Evaluation
    evaluation_results: Optional[Dict]
    best_model_name: Optional[str]
    best_model: Optional[Any]
    metrics: Optional[Dict]
    
    # Deployment
    model_path: Optional[str]
    api_endpoint_file: Optional[str]
    deployment_status: Optional[str]
    
    # Workflow
    current_step: str
    messages: Annotated[List[str], operator.add]
    errors: Annotated[List[str], operator.add]
    completed_steps: Annotated[List[str], operator.add]
    timestamp: str
    
    # Metadata
    execution_time: Optional[Dict[str, float]]
    thread_id: Optional[str]

def create_initial_state(dataset_path: str,  
                         problem_type: Optional[str] = None,
                         target_column: Optional[str] = None) -> DataScienceState:
    """Create initial state for workflow"""
    return DataScienceState(
        dataset_path=dataset_path,
        problem_type=problem_type,
        target_column=target_column,
        raw_data=None,
        data_info=None,
        data_quality_report=None,
        insights=[],
        visualizations=[],
        processed_data=None,
        engineered_features=[],
        recommended_models=None,
        model_reasoning=None,
        user_approved_models=None,
        trained_models=None,
        training_logs=None,
        best_params=None,
        evaluation_results=None,
        best_model_name=None,
        best_model=None,
        metrics=None,
        model_path=None,
        api_endpoint_file=None,
        deployment_status=None,
        current_step="initialization",
        messages=[],
        errors=[],
        completed_steps=[],
        timestamp=datetime.now().isoformat(),
        execution_time={},
        thread_id=None
    )