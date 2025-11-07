# In tools/api_tools.py

from typing import List, Dict, Any
import joblib
import pandas as pd

# This is a template for the FastAPI code.
# We will fill in the blanks ({...})
API_TEMPLATE = """
import uvicorn
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

# Load the trained model
model = joblib.load("{model_path}")

# Define the input data schema using Pydantic
# This is built from the columns in the training data
class ModelInput(BaseModel):
    {model_input_schema}

app = FastAPI(
    title="Nova-Sci Deployed Model",
    description="API for the auto-generated ML model"
)

@app.get("/")
def read_root():
    return {{"message": "Model API is running!"}}

@app.post("/predict")
def predict(data: ModelInput):
    # Convert Pydantic input to a pandas DataFrame
    input_df = pd.DataFrame([data.model_dump()])
    
    # Ensure columns are in the same order as training
    input_df = input_df[{feature_list}]
    
    # Get predictions
    try:
        prediction = model.predict(input_df)
        return {{"prediction": prediction.tolist()[0]}}
    except Exception as e:
        return {{"error": str(e)}}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
"""

def generate_pydantic_schema(df: pd.DataFrame) -> str:
    """Creates Pydantic field definitions from a DataFrame."""
    schema_lines = []
    for col, dtype in df.dtypes.items():
        if pd.api.types.is_numeric_dtype(dtype):
            schema_lines.append(f"    {col}: float")
        else:
            schema_lines.append(f"    {col}: str")
    return "\n".join(schema_lines)

def write_api_file(
    model_path: str, 
    processed_data: pd.DataFrame, 
    target_col: str, 
    api_file_path: str = "api/main.py"
) -> str:
    """
    Generates a complete FastAPI application file.
    """
    
    # Get the training features (X) to define API input
    features_df = processed_data.drop(columns=[target_col])
    
    # Create the Pydantic schema
    schema = generate_pydantic_schema(features_df)
    
    # Get the feature list for the model
    feature_list = features_df.columns.tolist()

    # Fill in the API template
    api_code = API_TEMPLATE.format(
        model_path=model_path,
        model_input_schema=schema,
        feature_list=str(feature_list) # Convert list to string for template
    )
    
    # Write the file
    import os
    os.makedirs(os.path.dirname(api_file_path), exist_ok=True)
    with open(api_file_path, "w") as f:
        f.write(api_code)
        
    return api_file_path