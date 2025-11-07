# In tools/data_tools.py

import pandas as pd
from typing import Dict, Any, List

def load_dataset(path: str) -> pd.DataFrame:
    """Loads a dataset from a file path (supports CSV, Excel)."""
    if path.endswith('.csv'):
        return pd.read_csv(path)
    elif path.endswith(('.xls', '.xlsx')):
        return pd.read_excel(path)
    else:
        raise ValueError(f"Unsupported file type: {path}")

def profile_data(df: pd.DataFrame) -> (Dict[str, Any], Dict[str, Any]):
    """Generates a data info dict and a data quality report dict."""
    
    # Data Info
    total_rows, total_cols = df.shape
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=['number']).columns.tolist()
    
    data_info = {
        "total_rows": total_rows,
        "total_columns": total_cols,
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "sample_data": df.head().to_dict('records')
    }
    
    # Data Quality Report
    missing_values = df.isnull().sum()
    total_missing = int(missing_values.sum())
    missing_percentage = (missing_values / total_rows) * 100
    duplicate_rows = int(df.duplicated().sum())
    
    data_quality_report = {
        "total_missing_values": total_missing,
        "missing_by_column": missing_values.to_dict(),
        "missing_percentage": missing_percentage.to_dict(),
        "duplicate_rows": duplicate_rows
    }
    
    return data_info, data_quality_report

def clean_data(df: pd.DataFrame, numeric_cols: List[str], categorical_cols: List[str]) -> pd.DataFrame:
    """Cleans data by imputing missing values (median/mode)."""
    df_cleaned = df.copy()
    
    # Impute numeric columns with median
    for col in numeric_cols:
        if df_cleaned[col].isnull().any():
            median_val = df_cleaned[col].median()
            df_cleaned[col] = df_cleaned[col].fillna(median_val)
            
    # Impute categorical columns with mode
    for col in categorical_cols:
        if df_cleaned[col].isnull().any():
            mode_val = df_cleaned[col].mode()
            if not mode_val.empty:
                df_cleaned[col] = df_cleaned[col].fillna(mode_val[0])
                
    return df_cleaned

def generate_eda_insights(df: pd.DataFrame, target_col: str) -> List[str]:
    """Generates simple text-based insights from the data."""
    insights = []
    
    if target_col in df.columns:
        insights.append(f"Target column '{target_col}' is present.")
        if target_col in df.select_dtypes(include=['number']).columns:
            insights.append(f"Target '{target_col}' is numeric. Mean: {df[target_col].mean():.2f}, Std: {df[target_col].std():.2f}.")
        else:
             insights.append(f"Target '{target_col}' is categorical. Top value: {df[target_col].mode()[0]}.")
    
    insights.append(f"Data has {df.shape[0]} rows and {df.shape[1]} columns.")
    
    # (You can add more complex insight generation here)
    
    return insights