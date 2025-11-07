# In tools/viz_tools.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from typing import List

def plot_correlation_matrix(df: pd.DataFrame, numeric_cols: List[str], save_path: str) -> str:
    """Plots a correlation matrix heatmap and saves it to a file."""
    if not numeric_cols:
        return "" # No numeric columns to plot

    plt.figure(figsize=(10, 8))
    corr = df[numeric_cols].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Correlation Matrix")
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    plt.savefig(save_path)
    plt.close()
    return save_path

# (You can add more functions here like plot_distributions, plot_scatter, etc.)