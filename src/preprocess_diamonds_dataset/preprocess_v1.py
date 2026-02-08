import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

DATA_PATH = "data/diamonds.csv"
RESULT_PATH = os.path.join("results", "feature_analysis_diamonds_dataset_versions")

def load_data(path=DATA_PATH):
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

def compute_correlation(df, save_path=os.path.join(RESULT_PATH, "correlation_matrix_v1.png")):
    numeric_df = df.select_dtypes(include=["float64", "int64"])
    corr = numeric_df.corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", square=True)
    plt.title("Feature Correlation Matrix (Numerical Features)")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"📊 Correlation heatmap saved to: {save_path}")

if __name__ == "__main__":
    df = load_data()
    compute_correlation(df)
