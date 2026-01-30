"""
"Download and clean 4 public tabular datasets for varied ML tasks."
- Binary classification: Titanic survival
- Multiclass classification: Dry Bean varieties
- Tabular image (flattened): MNIST digits
- Real-world tabular: Telco customer churn

Saves cleaned CSVs under data/ and logs basic stats (shape, columns, target distribution).
"""

import os
import sys
from pathlib import Path
from typing import Tuple, Callable

import pandas as pd
from sklearn.datasets import load_digits, fetch_openml
from sklearn.preprocessing import LabelEncoder

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)


def log_summary(name: str, df: pd.DataFrame, target: str) -> None:
    print(f"\n=== {name} ===")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    if target not in df.columns:
        print(f"⚠️ Target '{target}' not in columns; skipping target distribution.")
        return
    y = df[target]
    if y.dtype == object or str(y.dtype).startswith("category"):
        counts = y.value_counts(normalize=False)
        print("Target distribution (counts):")
        print(counts)
    else:
        print(f"Target summary: min={y.min()}, max={y.max()}, unique={y.nunique()}")


def encode_target(df: pd.DataFrame, target: str) -> pd.DataFrame:
    if target in df.columns and (df[target].dtype == object or str(df[target].dtype).startswith("category")):
        le = LabelEncoder()
        df[target] = le.fit_transform(df[target].astype(str))
    return df


def save_clean(df: pd.DataFrame, name: str, target: str) -> None:
    df = df.dropna()
    df = encode_target(df, target)
    out_path = DATA_DIR / f"{name}.csv"
    df.to_csv(out_path, index=False)
    log_summary(name, df, target)
    print(f"💾 Saved to {out_path}")


def load_titanic() -> Tuple[pd.DataFrame, str]:
    url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    df = pd.read_csv(url)
    return df, "Survived"


def load_dry_bean() -> Tuple[pd.DataFrame, str]:
    # Try UCI Excel directly, then OpenML fallback (ids are updated occasionally)
    excel_urls = [
        "https://archive.ics.uci.edu/ml/machine-learning-databases/00602/Dry_Bean_Dataset.xlsx",
        "https://archive.ics.uci.edu/ml/machine-learning-databases/00602/DryBeanDataset/Dry_Bean_Dataset.xlsx",
    ]
    for url in excel_urls:
        try:
            df = pd.read_excel(url)
            return df, "Class"
        except Exception:
            continue
    # OpenML fallback
    for data_id in [44160, 44161, 44163]:
        try:
            bean = fetch_openml(data_id=data_id, as_frame=True)
            df = bean.frame
            if "Class" not in df.columns and "class" in df.columns:
                df = df.rename(columns={"class": "Class"})
            return df, "Class"
        except Exception:
            continue
    raise RuntimeError("Unable to download Dry Bean dataset from UCI or OpenML.")


def load_mnist_tabular() -> Tuple[pd.DataFrame, str]:
    digits = load_digits(as_frame=True)
    df = digits.frame.copy()
    df = df.rename(columns={"target": "label"})
    return df, "label"


def load_telco_churn() -> Tuple[pd.DataFrame, str]:
    url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
    df = pd.read_csv(url)
    # Ensure numeric fields are parsed
    for col in ["TotalCharges"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df, "Churn"


def main():
    loaders: Tuple[Tuple[str, Callable[[], Tuple[pd.DataFrame, str]]], ...] = (
        ("titanic_binary_classification", load_titanic),
        ("drybean_multiclass_classification", load_dry_bean),
        ("mnist_tabular_digits", load_mnist_tabular),
        ("telco_churn_classification", load_telco_churn),
    )
    for name, loader in loaders:
        try:
            df, target = loader()
            save_clean(df, name, target)
        except Exception as e:
            print(f"❌ Failed to process {name}: {e}")
            continue


if __name__ == "__main__":
    sys.exit(main())
