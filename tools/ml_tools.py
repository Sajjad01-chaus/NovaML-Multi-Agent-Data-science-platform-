# In tools/ml_tools.py

# ... (Keep all the existing functions: 
#      MODEL_MAP, get_model_instance, split_data, train_single_model) ...

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from typing import Dict, Any, Tuple, List
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    mean_squared_error, r2_score, mean_absolute_error)


# A mapping to get model instances easily
MODEL_MAP = {
    "classification": {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
        "GradientBoosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
    },
    "regression": {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
        "GradientBoosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
    }
}

def get_model_instance(model_name: str, problem_type: str):
    """Gets a scikit-learn model instance from our map."""
    try:
        return MODEL_MAP[problem_type][model_name]
    except KeyError:
        raise ValueError(f"Model '{model_name}' not defined for problem type '{problem_type}'")

def split_data(df: pd.DataFrame, target_column: str, test_size: float = 0.2) -> Tuple:
    """Splits data into train and test sets."""
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return train_test_split(X, y, test_size=test_size, random_state=42)

def train_single_model(model: Any, X_train: pd.DataFrame, y_train: pd.DataFrame) -> Dict[str, Any]:
    """Trains a single model and gets its CV score."""
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Get cross-validation scores
    cv_scores = cross_val_score(model, X_train, y_train, cv=5)
    
    training_log = {
        'cv_scores': cv_scores.tolist(),
        'mean_cv_score': float(cv_scores.mean()),
        'std_cv_score': float(cv_scores.std())
    }
    
    return model, training_log

def encode_features(df: pd.DataFrame, categorical_cols: List[str], target_col: str) -> (pd.DataFrame, List[str]):
    """Performs one-hot encoding on categorical features."""
    df_encoded = df.copy()
    
    # Get list of categorical columns to encode (excluding target)
    cols_to_encode = [c for c in categorical_cols if c != target_col and c in df.columns]
    
    if not cols_to_encode:
        return df_encoded, [] # No encoding needed

    df_encoded = pd.get_dummies(df_encoded, columns=cols_to_encode, drop_first=True)
    
    # Get list of new feature names
    original_cols = set(df.columns)
    new_cols = set(df_encoded.columns)
    engineered_features = list(new_cols - original_cols)
    
    return df_encoded, engineered_features

def evaluate_classification(y_true, y_pred) -> Dict[str, float]:
    """Calculates all standard classification metrics."""
    return {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'precision': float(precision_score(y_true, y_pred, average='weighted', zero_division=0)),
        'recall': float(recall_score(y_true, y_pred, average='weighted', zero_division=0)),
        'f1_score': float(f1_score(y_true, y_pred, average='weighted', zero_division=0))
    }

def evaluate_regression(y_true, y_pred) -> Dict[str, float]:
    """Calculates all standard regression metrics."""
    return {
        'r2_score': float(r2_score(y_true, y_pred)),
        'mse': float(mean_squared_error(y_true, y_pred)),
        'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred))),
        'mae': float(mean_absolute_error(y_true, y_pred))
    }

def save_model(model: Any, model_name: str, model_dir: str = "models") -> str:
    """Saves a trained model to a .pkl file."""
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"best_model_{model_name}.pkl")
    joblib.dump(model, model_path)
    return model_path