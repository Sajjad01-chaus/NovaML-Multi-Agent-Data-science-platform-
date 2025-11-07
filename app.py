# In app.py

import streamlit as st
import pandas as pd
import os
import sys
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from orchestrator.graph import DataScienceOrchestrator
from orchestrator.state import DataScienceState

# --- Page Config (MUST be first Streamlit command) ---
st.set_page_config(
    page_title="NovaML: Multi-Agent DS Platform",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- All Helper Functions (You were missing these) ---

def display_header():
    """Display main header"""
    st.markdown("""
        <style>
        .main-header {
            font-size: 2.5rem;
            font-weight: bold;
            color: #1f77b4;
            text-align: center;
            margin-bottom: 2rem;
        }
        </style>
    """, unsafe_allow_html=True)
    st.markdown('<h1 class="main-header">🤖 NovaML: Multi-Agent Data Science Platform</h1>', unsafe_allow_html=True)
    st.markdown("---")

def display_data_overview(state: DataScienceState):
    st.header("📊 Data Overview")
    data_info = state.get("data_info", {})
    quality_report = state.get("data_quality_report", {})
    if data_info and quality_report:
        col1, col2, col3, col4 = st.columns(4)
        with col1: st.metric("Total Rows", f"{data_info.get('total_rows', 'N/A')}")
        with col2: st.metric("Total Columns", data_info.get('total_columns', 'N/A'))
        with col3: st.metric("Numeric Features", len(data_info.get('numeric_columns', [])))
        with col4: st.metric("Categorical Features", len(data_info.get('categorical_columns', [])))
        
        with st.expander("Data Quality Report"):
            st.metric("Total Missing Values", quality_report.get('total_missing_values', 'N/A'))
            st.metric("Duplicate Rows", quality_report.get('duplicate_rows', 'N/A'))

def display_eda_results(state: DataScienceState):
    st.header("🔍 Exploratory Data Analysis")
    insights = state.get("insights", [])
    visualizations = state.get("visualizations", [])
    if insights:
        st.subheader("💡 Key Insights")
        for insight in insights: st.info(f"• {insight}")
    if visualizations:
        st.subheader("📈 Visualizations")
        for viz_path in visualizations:
            if os.path.exists(viz_path):
                try:
                    st.image(Image.open(viz_path), use_container_width=True, caption=os.path.basename(viz_path))
                except Exception as e:
                    st.warning(f"Could not load image {viz_path}: {e}")

def display_model_results(state: DataScienceState):
    st.header("🎯 Model Results")
    evaluation_results = state.get("evaluation_results", {})
    best_model_name = state.get("best_model_name")
    if best_model_name:
        st.success(f"🏆 Best Model: **{best_model_name}**")
    if evaluation_results:
        st.subheader("Model Performance Comparison")
        metrics_df = pd.DataFrame(evaluation_results).T
        st.dataframe(metrics_df.style.highlight_max(axis=0, color="lightgreen"), use_container_width=True)

def display_workflow_logs(state: DataScienceState):
    st.header("📋 Workflow Logs")
    with st.expander("Show Agent Logs"):
        messages = state.get("messages", [])
        if messages:
            log_text = "\n".join(reversed(messages))
            st.code(log_text, language=None)

# --- End Helper Functions ---


def initialize_session_state():
    """Initialize session state variables"""
    if 'orchestrator' not in st.session_state:
        st.session_state.orchestrator = DataScienceOrchestrator()
    if 'graph_state' not in st.session_state:
        st.session_state.graph_state = None
    if 'workflow_status' not in st.session_state:
        st.session_state.workflow_status = 'initial'

def display_sidebar():
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        is_disabled = st.session_state.workflow_status in ['running', 'awaiting_input']
        
        uploaded_file = st.file_uploader(
            "Upload Dataset", 
            type=['csv', 'xlsx'],  # <-- Removed 'json' for consistency
            disabled=is_disabled
        )
        problem_type = st.selectbox(
            "Problem Type", 
            ["auto", "classification", "regression"], # <-- Changed "Auto-detect" to "auto"
            disabled=is_disabled
        )
        target_column = st.text_input(
            "Target Column", 
            placeholder="Enter target column name", 
            disabled=is_disabled
        )
        
        st.markdown("---")
        
        run_button = st.button(
            "🚀 Run Analysis", type="primary", use_container_width=True,
            disabled=(is_disabled or not uploaded_file or not target_column)
        )
        
        if st.button("🔄 Reset", use_container_width=True):
            st.session_state.clear()
            st.rerun()
            
    return uploaded_file, problem_type, target_column, run_button

def display_human_review_ui():
    """Displays the model approval UI when the graph is paused."""
    st.info("🤖 **Human-in-the-Loop:** The agents have recommended the following models.")
    st.write(st.session_state.graph_state.get('model_reasoning', 'No reasoning provided.'))
    
    recommended = st.session_state.graph_state.get('recommended_models', [])
    if not recommended:
        st.error("Model Selection Agent failed to recommend models. Please check logs and reset.")
        return

    with st.form("hitl_form"):
        st.subheader("Please approve the models to be trained:")
        approved_models = []
        for model in recommended:
            if st.checkbox(model, value=True):
                approved_models.append(model)
        
        submit_button = st.form_submit_button("✅ Approve & Resume Training")

    if submit_button:
        if not approved_models:
            st.warning("Please select at least one model to train.")
        else:
            st.session_state.workflow_status = 'running'
            st.session_state.user_approved_models = approved_models
            st.rerun()

def display_results(state: DataScienceState):
    """Displays the final results once the workflow is 'completed'."""
    st.success(f"✅ Workflow Completed! Best model: **{state.get('best_model_name', 'N/A')}**")
    
    tabs = st.tabs(["🎯 Model Results", "📊 Data Overview", "🔍 EDA", "📋 Logs"])
    
    with tabs[0]: display_model_results(state)
    with tabs[1]: display_data_overview(state)
    with tabs[2]: display_eda_results(state)
    with tabs[3]: display_workflow_logs(state)
    
    # Download buttons
    if state.get("model_path") and os.path.exists(state["model_path"]):
        with open(state["model_path"], "rb") as f:
            st.download_button(
                label="⬇️ Download Best Model (.pkl)", data=f,
                file_name=os.path.basename(state["model_path"]),
                mime="application/octet-stream"
            )
    
    if state.get("api_endpoint_file") and os.path.exists(state["api_endpoint_file"]):
        with open(state["api_endpoint_file"], "r") as f:
            st.download_button(
                label="⬇️ Download Generated API (.py)", data=f.read(),
                file_name="main.py", mime="text/x-python"
            )

def main():
    initialize_session_state()
    display_header()
    
    uploaded_file, prob_type, target_col, run_button = display_sidebar()
    
    # Main app logic
    try:
        if run_button:
            st.session_state.workflow_status = 'running'
            os.makedirs("data", exist_ok=True)
            dataset_path = os.path.join("data", uploaded_file.name)
            with open(dataset_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            st.session_state.run_params = {
                "dataset_path": dataset_path,
                "target_column": target_col,
                "problem_type": None if prob_type == "auto" else prob_type
            }
            st.rerun()

        if st.session_state.workflow_status == 'initial':
            st.info("👋 Welcome! Configure your analysis in the sidebar and click 'Run'.")

        elif st.session_state.workflow_status == 'running':
            # This block is for *both* starting and resuming
            if 'thread_id' not in (st.session_state.graph_state or {}):
                with st.spinner("🚀 Starting workflow... Agents are ingesting, cleaning, and running EDA..."):
                    interrupted_state = st.session_state.orchestrator.start_workflow(
                        **st.session_state.run_params
                    )
                    st.session_state.graph_state = interrupted_state
                    st.session_state.workflow_status = 'awaiting_input'
                    st.rerun()
            else:
                with st.spinner("👩‍🔬 Human input received! Resuming... Training, evaluating, and deploying..."):
                    thread_id = st.session_state.graph_state['thread_id']
                    approved_models = st.session_state.user_approved_models
                    
                    final_state = st.session_state.orchestrator.resume_workflow(
                        thread_id, approved_models
                    )
                    
                    st.session_state.graph_state = final_state
                    st.session_state.workflow_status = 'completed'
                    st.rerun()

        elif st.session_state.workflow_status == 'awaiting_input':
            display_human_review_ui()
            
        elif st.session_state.workflow_status == 'completed':
            display_results(st.session_state.graph_state)

    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        st.session_state.workflow_status = 'initial' # Reset on error
        st.button("Reset Workflow")
        if st.session_state.graph_state and st.session_state.graph_state.get('errors'):
            st.subheader("Logged Errors:")
            st.error(st.session_state.graph_state['errors'][-1])

if __name__ == "__main__":
    main()