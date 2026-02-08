"""
CHANGELOG from preprocess_v1.py:
- Added logic to drop features with Pearson correlation > 0.85.
- Purpose: eliminate redundant predictors that could distort feature importance scores,
  inflate variance, or confuse model learning, especially in tree-based or linear models.
- Output: cleaned DataFrame saved to `data/diamonds_clean.csv`.

"""


import os
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

DATA_PATH = "data/diamonds.csv"
RESULT_PATH = os.path.join("results", "feature_analysis_diamonds_dataset_versions")
CLEANED_DATA_PATH = os.path.join("data", "diamonds_v2.csv")
HEATMAP_PATH = os.path.join(RESULT_PATH, "correlation_matrix_v2.png")


def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    unnamed_cols = [col for col in df.columns if col.startswith("Unnamed")]
    if unnamed_cols:
        df = df.drop(columns=unnamed_cols)
        print(f"🧹 Dropped unnamed index columns: {unnamed_cols}")
    print("✅ Dataset loaded successfully!")
    print(f"Shape: {df.shape}")
    print("Columns:", df.columns.tolist())
    print("\nMissing values:\n", df.isnull().sum())
    return df


def compute_correlation(
    df: pd.DataFrame, save_path: str = HEATMAP_PATH
) -> pd.DataFrame:
    numeric_df = df.select_dtypes(include=["float64", "int64"])
    corr = numeric_df.corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", square=True)
    plt.title("Feature Correlation Matrix (Numerical Features)")
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    print(f"📊 Correlation heatmap saved to: {save_path}")
    return corr


def drop_highly_correlated(
    df: pd.DataFrame, corr_matrix: pd.DataFrame, threshold: float = 0.85
) -> Tuple[pd.DataFrame, List[str]]:
    abs_corr = corr_matrix.abs()
    mask = np.triu(np.ones(abs_corr.shape), k=1).astype(bool)
    upper = abs_corr.where(mask)

    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    if to_drop:
        cleaned_df = df.drop(columns=to_drop)
        print(f"🧹 Dropped columns with |correlation| > {threshold}: {to_drop}")
    else:
        cleaned_df = df.copy()
        print(f"✅ No columns exceeded the correlation threshold ({threshold}).")

    return cleaned_df, to_drop


def save_cleaned_data(df: pd.DataFrame, path: str = CLEANED_DATA_PATH) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"💾 Cleaned dataset saved to: {path}")


if __name__ == "__main__":
    data = load_data()
    correlation_matrix = compute_correlation(data)
    cleaned_data, dropped_columns = drop_highly_correlated(data, correlation_matrix)
    save_cleaned_data(cleaned_data)
