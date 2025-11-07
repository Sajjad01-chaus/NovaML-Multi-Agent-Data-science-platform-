# In orchestrator/graph.py

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
from orchestrator.state import DataScienceState, create_initial_state
import time
from typing import Dict, Any, List

# --- CORRECT IMPORTS ---
# Import each agent from its own file
from agents.data_ingestion_agent import DataIngestionAgent
from agents.data_cleaning_agent import DataCleaningAgent
from agents.eda_agent import EDAAgent
from agents.feature_engineering_agent import FeatureEngineeringAgent
from agents.model_selection_agent import ModelSelectionAgent
from agents.training_agent import TrainingAgent
from agents.evaluation_agent import EvaluationAgent
from agents.deployment_agent import DeploymentAgent
# --- END CORRECTIONS ---

class DataScienceOrchestrator:
    def __init__(self):
        self.memory = SqliteSaver.from_conn_string(":memory:")
        self.workflow = StateGraph(DataScienceState)
        self._initialize_graph()
        
        self.app = self.workflow.compile(
            checkpointer=self.memory,
            interrupt_before=["train"] 
        )

    def _initialize_graph(self):
        self.workflow = StateGraph(DataScienceState)
        
        # 1. Instantiate ALL 8 agents
        ingestion_agent = DataIngestionAgent()
        cleaning_agent = DataCleaningAgent()
        eda_agent = EDAAgent()
        feature_eng_agent = FeatureEngineeringAgent()
        model_selection_agent = ModelSelectionAgent()
        training_agent = TrainingAgent()
        evaluation_agent = EvaluationAgent()
        deployment_agent = DeploymentAgent() # <-- This was missing
        
        # 2. Add ALL 8 agents as nodes
        self.workflow.add_node("ingest", self.run_agent(ingestion_agent.execute))
        self.workflow.add_node("clean", self.run_agent(cleaning_agent.execute))
        self.workflow.add_node("eda", self.run_agent(eda_agent.execute))
        self.workflow.add_node("feature_eng", self.run_agent(feature_eng_agent.execute))
        self.workflow.add_node("select_model", self.run_agent(model_selection_agent.execute))
        self.workflow.add_node("train", self.run_agent(training_agent.execute))
        self.workflow.add_node("evaluate", self.run_agent(evaluation_agent.execute))
        self.workflow.add_node("deploy", self.run_agent(deployment_agent.execute)) # <-- This was missing

        # Node for HITL
        def human_review_node(state: DataScienceState):
            # This node is just a placeholder for the interruption
            # The log_message call was removed as this class doesn't have it
            return state
        
        self.workflow.add_node("human_review", human_review_node)

        # 3. Define conditional router
        def data_quality_router(state: DataScienceState):
            if state['data_quality_report']['total_missing_values'] > 0:
                return "clean"
            else:
                return "eda"

        # 4. Build the graph topology
        self.workflow.set_entry_point("ingest")
        self.workflow.add_conditional_edges(
            "ingest",
            data_quality_router,
            {"clean": "clean", "eda": "eda"}
        )
        self.workflow.add_edge("clean", "eda")
        self.workflow.add_edge("eda", "feature_eng")
        self.workflow.add_edge("feature_eng", "select_model")
        
        # HITL Interruption Point
        self.workflow.add_edge("select_model", "human_review")
        self.workflow.add_edge("human_review", "train") # Pauses HERE
        
        self.workflow.add_edge("train", "evaluate")
        self.workflow.add_edge("evaluate", "deploy") # <-- Corrected flow
        self.workflow.add_edge("deploy", END) # <-- Corrected flow

    # --- COMPLETE run_agent WRAPPER ---
    # Your file had this as a comment
    def run_agent(self, agent_function):
        """Wrapper to add execution time tracking."""
        def wrapper(state: DataScienceState):
            start_time = time.time()
            agent_name = agent_function.__self__.name
            state['current_step'] = agent_name
            
            try:
                # Run the actual agent function
                state = agent_function(state)
            except Exception as e:
                error_msg = f"Error in {agent_name}: {str(e)}"
                print(error_msg)
                state['errors'].append(error_msg)
                # You might want to stop the graph here
                # For now, we'll let it continue to END, but errors will be logged
                raise e # Re-raise exception to stop the graph
            
            end_time = time.time()
            if 'execution_time' not in state:
                state['execution_time'] = {}
            state['execution_time'][agent_name] = end_time - start_time
            
            return state
        
        wrapper.__self__ = agent_function.__self__
        return wrapper
    # --- END COMPLETE WRAPPER ---

    def start_workflow(self, dataset_path: str, target_column: str, problem_type: str = None) -> Dict[str, Any]:
        """STARTS a new workflow."""
        initial_state = create_initial_state(
            dataset_path=dataset_path,
            problem_type=problem_type,
            target_column=target_column
        )
        
        thread_id = str(int(time.time()))
        config = {"configurable": {"thread_id": thread_id}}
        
        final_state = {}
        # Use .invoke() to run until the first interruption
        final_state = self.app.invoke(initial_state, config)

        final_state['thread_id'] = thread_id
        return final_state

    def resume_workflow(self, thread_id: str, approved_models: List[str]) -> Dict[str, Any]:
        """RESUMES a paused workflow."""
        config = {"configurable": {"thread_id": thread_id}}
        
        # Get current state and update it
        current_state_dict = self.app.get_state(config).values
        current_state_dict['user_approved_models'] = approved_models
        
        # Resume the graph
        final_state = {}
        # Pass the updated state to continue the run
        for event in self.app.stream(None, config, input=current_state_dict):
             final_state = list(event.values())[0]
            
        return final_state