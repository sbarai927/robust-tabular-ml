"""
Quantify SHAP stability under covariate shift for tabular classification datasets.

This script reuses saved HPO artifacts (best_model.pkl) and the same preprocessing
as HPO to avoid retraining or new datasets. Outputs are stored under:
results/why_tree_outperforms/<dataset>/<model>/
"""

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

# Reduce OpenMP shared-memory issues in constrained environments.
os.environ.setdefault("KMP_USE_SHM", "0")
os.environ.setdefault("KMP_SHM_DISABLE", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MPLBACKEND", "Agg")
mpl_dir = Path("results/.matplotlib").absolute()
mpl_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(mpl_dir))

# Import torch first to avoid OpenMP SHM init crashes on some setups.
try:
    import torch
except Exception:
    torch = None

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder

try:
    import shap
except Exception:
    shap = None

RANDOM_STATE = 42
RESULTS_DIR = Path("results/why_tree_outperforms")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class DatasetConfig:
    name: str
    path: Path
    target: str
    is_multiclass: bool = False


DATASETS: Dict[str, DatasetConfig] = {
    "titanic": DatasetConfig(
        "titanic", Path("data/titanic_binary_classification.csv"), "Survived", False
    ),
    "telco": DatasetConfig(
        "telco", Path("data/telco_churn_classification.csv"), "Churn", False
    ),
    "drybean": DatasetConfig(
        "drybean", Path("data/drybean_multiclass_classification.csv"), "Class", True
    ),
}

MODELS = ["rf", "lgbm", "catboost", "xgboost", "mlp", "tabnet"]


def load_dataset(cfg: DatasetConfig) -> Tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(cfg.path)
    df = df.dropna().reset_index(drop=True)
    X = df.drop(columns=[cfg.target])
    y = df[cfg.target]
    return X, y


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    cat_cols = [c for c in X.columns if X[c].dtype == object]
    num_cols = [c for c in X.columns if c not in cat_cols]
    transformers = []
    if num_cols:
        transformers.append(("num", StandardScaler(), num_cols))
    if cat_cols:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols))
    return ColumnTransformer(transformers)


def classification_metrics(y_true, y_pred, is_multiclass: bool) -> Dict[str, float]:
    average = "macro" if is_multiclass else "binary"
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average=average)
    return {"Accuracy": float(acc), "F1": float(f1)}


def load_model(cfg: DatasetConfig, model_name: str):
    model_path = Path("results/hpo") / cfg.name / model_name / "best_model.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model at {model_path}")
    return joblib.load(model_path)


def apply_covariate_shift(
    X: pd.DataFrame,
    num_stats: Dict[str, Tuple[float, float]],
    cat_modes: Dict[str, str],
    rng: np.random.Generator,
) -> pd.DataFrame:
    # Add numeric mean shift + noise + structured outliers, and perturb categoricals.
    X_shift = X.copy()
    for col, (mean, std) in num_stats.items():
        if std == 0 or np.isnan(std):
            continue
        noise = rng.normal(0.0, 0.5 * std, size=len(X_shift))
        shift = 0.2 * std
        X_shift[col] = X_shift[col].astype(float) + shift + noise
        outlier_mask = rng.random(len(X_shift)) < 0.02
        X_shift.loc[outlier_mask, col] = X_shift.loc[outlier_mask, col] + 3.0 * std
    for col, mode in cat_modes.items():
        mask = rng.random(len(X_shift)) < 0.1
        X_shift.loc[mask, col] = mode
        mask_new = rng.random(len(X_shift)) < 0.05
        X_shift.loc[mask_new, col] = "ShiftedCategory"
    return X_shift


def _feature_names(preprocessor: ColumnTransformer) -> List[str]:
    names = []
    for name, transformer, cols in preprocessor.transformers_:
        if name == "cat" and hasattr(transformer, "get_feature_names_out"):
            names.extend(transformer.get_feature_names_out(cols))
        else:
            names.extend(cols)
    return [str(n) for n in names]


def compute_shap_importance(
    model,
    model_name: str,
    X_sample: np.ndarray,
    feature_names: List[str],
    is_multiclass: bool,
) -> Dict[str, float]:
    if shap is None:
        raise RuntimeError("shap not installed")
    if model_name in {"rf", "lgbm", "xgboost", "catboost"}:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample, check_additivity=False)
        if isinstance(shap_values, list):
            values = np.mean([np.abs(v) for v in shap_values], axis=0)
        else:
            values = np.abs(shap_values)
        if values.ndim == 3:
            values = values.mean(axis=(0, 2))
        else:
            values = values.mean(axis=0)
    else:
        predict_fn = model.predict_proba if hasattr(model, "predict_proba") else model.predict
        background = shap.utils.sample(X_sample, min(50, X_sample.shape[0]))
        explainer = shap.KernelExplainer(predict_fn, background)
        shap_values = explainer.shap_values(X_sample, nsamples=50)
        if isinstance(shap_values, list):
            values = np.mean([np.abs(v) for v in shap_values], axis=0)
        else:
            values = np.abs(shap_values)
        if values.ndim == 3:
            values = values.mean(axis=(0, 2))
        else:
            values = values.mean(axis=0)
    return {feature_names[i]: float(values[i]) for i in range(len(feature_names))}


def rank_stability(
    clean_imp: Dict[str, float], shift_imp: Dict[str, float], top_k: int
) -> Dict[str, float]:
    clean = pd.Series(clean_imp)
    shift = pd.Series(shift_imp)
    common = clean.index.intersection(shift.index)
    clean = clean.loc[common]
    shift = shift.loc[common]
    clean_rank = clean.rank(ascending=False)
    shift_rank = shift.rank(ascending=False)
    spearman = float(clean_rank.corr(shift_rank))

    clean_top = set(clean.sort_values(ascending=False).head(top_k).index)
    shift_top = set(shift.sort_values(ascending=False).head(top_k).index)
    union = clean_top | shift_top
    jaccard = float(len(clean_top & shift_top) / max(len(union), 1))
    return {
        "spearman": spearman,
        "top_k": top_k,
        "jaccard": jaccard,
        "clean_top": sorted(clean_top),
        "shift_top": sorted(shift_top),
    }


def evaluate_model(
    cfg: DatasetConfig,
    model_name: str,
    max_shap_samples: int,
    run_shift: bool,
) -> Dict[str, float]:
    X, y = load_dataset(cfg)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    preprocessor = build_preprocessor(X_train)
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)
    if hasattr(X_train_proc, "toarray"):
        X_train_proc = X_train_proc.toarray()
        X_test_proc = X_test_proc.toarray()

    model = load_model(cfg, model_name)

    label_encoder = None
    if model_name == "tabnet":
        label_encoder = LabelEncoder()
        label_encoder.fit(y_train)
        y_test_eval = label_encoder.transform(y_test)
        X_test_proc = np.asarray(X_test_proc, dtype=np.float32)
    else:
        y_test_eval = y_test

    preds = model.predict(X_test_proc)
    metrics_clean = classification_metrics(y_test_eval, preds, cfg.is_multiclass)

    rng = np.random.default_rng(RANDOM_STATE)
    num_cols = [c for c in X_train.columns if X_train[c].dtype != object]
    cat_cols = [c for c in X_train.columns if X_train[c].dtype == object]
    num_stats = {c: (float(X_train[c].mean()), float(X_train[c].std())) for c in num_cols}
    cat_modes = {c: str(X_train[c].mode(dropna=True).iloc[0]) for c in cat_cols}

    out_dir = RESULTS_DIR / cfg.name / model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_shift = None
    shift_imp = None
    if run_shift:
        X_shift = apply_covariate_shift(X_test, num_stats, cat_modes, rng)
        X_shift_proc = preprocessor.transform(X_shift)
        if hasattr(X_shift_proc, "toarray"):
            X_shift_proc = X_shift_proc.toarray()
        if model_name == "tabnet":
            X_shift_proc = np.asarray(X_shift_proc, dtype=np.float32)
        preds_shift = model.predict(X_shift_proc)
        metrics_shift = classification_metrics(y_test_eval, preds_shift, cfg.is_multiclass)

    stability = None
    if shap is not None:
        sample_cap = max_shap_samples
        if model_name in {"mlp", "tabnet"}:
            sample_cap = min(sample_cap, 50)
        sample_n = min(sample_cap, X_test_proc.shape[0])
        idx = rng.choice(X_test_proc.shape[0], size=sample_n, replace=False)
        feature_names = _feature_names(preprocessor)
        clean_imp = compute_shap_importance(
            model,
            model_name,
            X_test_proc[idx],
            feature_names,
            cfg.is_multiclass,
        )
        pd.DataFrame(
            {"feature": list(clean_imp.keys()), "importance": list(clean_imp.values())}
        ).to_csv(out_dir / "shap_importance_clean.csv", index=False)
        if run_shift:
            shift_imp = compute_shap_importance(
                model,
                model_name,
                X_shift_proc[idx],
                feature_names,
                cfg.is_multiclass,
            )
            pd.DataFrame(
                {"feature": list(shift_imp.keys()), "importance": list(shift_imp.values())}
            ).to_csv(out_dir / "shap_importance_shift.csv", index=False)
            stability = rank_stability(clean_imp, shift_imp, top_k=10)
            with open(out_dir / "shap_stability.json", "w") as f:
                json.dump(stability, f, indent=2)

    if run_shift and metrics_shift is not None:
        delta_f1 = metrics_shift["F1"] - metrics_clean["F1"]
        rel_drop = delta_f1 / metrics_clean["F1"] if metrics_clean["F1"] else 0.0
        perf = {
            "Accuracy_clean": metrics_clean["Accuracy"],
            "F1_clean": metrics_clean["F1"],
            "Accuracy_shift": metrics_shift["Accuracy"],
            "F1_shift": metrics_shift["F1"],
            "delta_f1": float(delta_f1),
            "relative_drop_f1": float(rel_drop),
        }
        with open(out_dir / "performance_shift.json", "w") as f:
            json.dump(perf, f, indent=2)
    return {
        "dataset": cfg.name,
        "model": model_name,
        "Accuracy_clean": metrics_clean["Accuracy"],
        "F1_clean": metrics_clean["F1"],
        "Accuracy_shift": None if metrics_shift is None else metrics_shift["Accuracy"],
        "F1_shift": None if metrics_shift is None else metrics_shift["F1"],
        "shap_spearman": None if stability is None else stability["spearman"],
        "shap_jaccard": None if stability is None else stability["jaccard"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="SHAP stability under covariate shift.")
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS.keys()))
    parser.add_argument("--models", nargs="+", default=MODELS)
    parser.add_argument("--max-shap-samples", type=int, default=200)
    parser.add_argument("--skip-shift", action="store_true")
    args = parser.parse_args()

    rows: List[Dict[str, float]] = []
    for dataset_name in args.datasets:
        if dataset_name not in DATASETS:
            continue
        cfg = DATASETS[dataset_name]
        for model_name in args.models:
            if model_name not in MODELS:
                continue
            try:
                rows.append(
                    evaluate_model(
                        cfg,
                        model_name,
                        max_shap_samples=args.max_shap_samples,
                        run_shift=not args.skip_shift,
                    )
                )
            except Exception as exc:
                out_dir = RESULTS_DIR / cfg.name / model_name
                out_dir.mkdir(parents=True, exist_ok=True)
                with open(out_dir / "error.json", "w") as f:
                    json.dump({"error": str(exc)}, f, indent=2)
                rows.append(
                    {
                        "dataset": cfg.name,
                        "model": model_name,
                        "Accuracy_clean": None,
                        "F1_clean": None,
                        "Accuracy_shift": None,
                        "F1_shift": None,
                        "shap_spearman": None,
                        "shap_jaccard": None,
                        "error": str(exc),
                    }
                )

    summary_path = RESULTS_DIR / "summary_metrics.csv"
    if summary_path.exists():
        existing = pd.read_csv(summary_path)
        existing_rows = {
            (row["dataset"], row["model"]): row.to_dict()
            for _, row in existing.iterrows()
        }
        for row in rows:
            existing_rows[(row["dataset"], row["model"])] = row
        summary = pd.DataFrame(existing_rows.values())
    else:
        summary = pd.DataFrame(rows)
    summary.to_csv(summary_path, index=False)


if __name__ == "__main__":
    main()
