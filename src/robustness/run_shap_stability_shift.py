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
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
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
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    LabelEncoder,
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
)

try:
    import shap
except Exception:
    shap = None

try:
    from tabpfn import TabPFNClassifier
except Exception:
    TabPFNClassifier = None

try:
    from apt.model import APTClassifier
except Exception:
    APTClassifier = None

RANDOM_STATE = 42
RESULTS_DIR = Path("results/tree_vs_deep_stability_analysis")
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

MODELS = ["rf", "lgbm", "catboost", "xgboost", "mlp", "tabnet", "tabpfn", "apt"]


def load_dataset(cfg: DatasetConfig) -> Tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(cfg.path)
    df = df.dropna().reset_index(drop=True)
    X = df.drop(columns=[cfg.target])
    y = df[cfg.target]
    return X, y


def build_preprocessor(X: pd.DataFrame, model_name: str) -> ColumnTransformer:
    """Build the feature matrix expected by the saved HPO model artifacts."""
    cat_cols = [c for c in X.columns if X[c].dtype == object]
    num_cols = [c for c in X.columns if c not in cat_cols]
    transformers = []
    if num_cols:
        if model_name in {"apt"}:
            transformers.append(
                (
                    "num",
                    Pipeline(
                        [
                            ("imputer", SimpleImputer(strategy="mean")),
                            ("scaler", StandardScaler()),
                        ]
                    ),
                    num_cols,
                )
            )
        else:
            transformers.append(
                (
                    "num",
                    Pipeline(
                        [
                            ("imputer", SimpleImputer(strategy="mean")),
                            ("scaler", StandardScaler()),
                        ]
                    ),
                    num_cols,
                )
            )
    if cat_cols:
        if model_name in {"apt"}:
            transformers.append(
                (
                    "cat",
                    Pipeline(
                        [
                            (
                                "imputer",
                                SimpleImputer(strategy="constant", fill_value="Missing"),
                            ),
                            ("encoder", OneHotEncoder(handle_unknown="ignore")),
                        ]
                    ),
                    cat_cols,
                )
            )
        else:
            transformers.append(
                (
                    "cat",
                    Pipeline(
                        [
                            (
                                "imputer",
                                SimpleImputer(strategy="constant", fill_value="Missing"),
                            ),
                            ("encoder", OneHotEncoder(handle_unknown="ignore")),
                        ]
                    ),
                    cat_cols,
                )
            )
    return ColumnTransformer(transformers)


def build_tabpfn_inputs(
    X_train: pd.DataFrame, X_test: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray, List[int], List[str]]:
    cat_cols = [c for c in X_train.columns if X_train[c].dtype == object]
    num_cols = [c for c in X_train.columns if c not in cat_cols]
    num_means = {c: float(X_train[c].mean()) for c in num_cols}
    categories = {
        col: pd.Categorical(X_train[col].astype(str).fillna("missing")).categories
        for col in cat_cols
    }

    def encode(df: pd.DataFrame) -> np.ndarray:
        num = (
            df[num_cols].astype(float).fillna(pd.Series(num_means)).to_numpy()
            if num_cols
            else np.empty((len(df), 0))
        )
        if cat_cols:
            cat_arrays = []
            for col in cat_cols:
                cat = pd.Categorical(
                    df[col].astype(str).fillna("missing"),
                    categories=categories[col],
                )
                cat_arrays.append(cat.codes.reshape(-1, 1))
            cat_mat = np.hstack(cat_arrays)
        else:
            cat_mat = np.empty((len(df), 0))
        return np.hstack([num, cat_mat]).astype(np.float32)

    X_train_proc = encode(X_train)
    X_test_proc = encode(X_test)
    cat_indices = list(range(len(num_cols), len(num_cols) + len(cat_cols)))
    feature_names = [str(c) for c in num_cols + cat_cols]
    return X_train_proc, X_test_proc, cat_indices, feature_names


def load_apt_artifact(model_path: Path):
    if torch is None:
        raise RuntimeError("torch not available for APT loading")
    if APTClassifier is None:
        raise RuntimeError("APT not installed")
    payload = torch.load(model_path, map_location="cpu", weights_only=False)
    model = APTClassifier(device="cpu")
    state = payload.get("state_dict") if isinstance(payload, dict) else None
    if state:
        model.load_state_dict(state)
    for key in ["x_train", "y_train", "x_encoder", "y_encoder", "feature_perm"]:
        if isinstance(payload, dict) and key in payload:
            setattr(model, key, payload[key])
    model.eval()
    return model


def classification_metrics(y_true, y_pred, is_multiclass: bool) -> Dict[str, float]:
    average = "macro" if is_multiclass else "binary"
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average=average)
    return {"Accuracy": float(acc), "F1": float(f1)}


def load_model(cfg: DatasetConfig, model_name: str):
    model_path = Path("results/hpo") / cfg.name / model_name / "best_model.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model at {model_path}")
    if model_name == "apt":
        return load_apt_artifact(model_path)
    return joblib.load(model_path)


def constrain_prediction_threads(model, model_name: str):
    """Force single-thread prediction for libraries that can hit OpenMP SHM limits."""
    try:
        if model_name in {"rf", "lgbm", "xgboost"} and hasattr(model, "set_params"):
            model.set_params(n_jobs=1)
    except Exception:
        pass
    # Avoid calling LightGBM booster_.reset_parameter after unpickling: on some
    # macOS/ARM environments it segfaults. The process-level thread variables
    # above still keep prediction single-threaded enough for this evaluation.
    try:
        if model_name == "xgboost" and hasattr(model, "get_booster"):
            model.get_booster().set_param({"nthread": 1})
    except Exception:
        pass
    return model


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
        cols = [str(c) for c in cols]
        encoder = None
        if hasattr(transformer, "named_steps"):
            encoder = transformer.named_steps.get("encoder")
        if (
            name == "cat"
            and encoder is not None
            and isinstance(encoder, OneHotEncoder)
            and hasattr(encoder, "get_feature_names_out")
        ):
            names.extend(encoder.get_feature_names_out(cols))
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
        if model_name in {"tabpfn", "apt"}:
            predict_fn = model.predict
            background = shap.utils.sample(X_sample, min(5, X_sample.shape[0]))
            nsamples = 10
        else:
            predict_fn = model.predict_proba if hasattr(model, "predict_proba") else model.predict
            background = shap.utils.sample(X_sample, min(50, X_sample.shape[0]))
            nsamples = 50
        explainer = shap.KernelExplainer(predict_fn, background)
        shap_values = explainer.shap_values(X_sample, nsamples=nsamples)
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
    if model_name == "tabpfn":
        X_train_proc, X_test_proc, _, feature_names = build_tabpfn_inputs(X_train, X_test)
        preprocessor = None
    else:
        preprocessor = build_preprocessor(X_train, model_name)
        X_train_proc = preprocessor.fit_transform(X_train)
        X_test_proc = preprocessor.transform(X_test)
        if hasattr(X_train_proc, "toarray"):
            X_train_proc = X_train_proc.toarray()
            X_test_proc = X_test_proc.toarray()
        feature_names = _feature_names(preprocessor)

    model = constrain_prediction_threads(load_model(cfg, model_name), model_name)

    label_encoder = None
    if model_name in {"tabnet", "tabpfn", "apt", "xgboost"}:
        label_encoder = LabelEncoder()
        label_encoder.fit(y_train)
        y_test_eval = label_encoder.transform(y_test)
        X_test_proc = np.asarray(X_test_proc, dtype=np.float32)
    else:
        y_test_eval = y_test

    rng = np.random.default_rng(RANDOM_STATE)
    eval_idx = None
    if model_name in {"tabpfn", "apt"} and X_test_proc.shape[0] > 20:
        eval_idx = rng.choice(X_test_proc.shape[0], size=20, replace=False)
        X_test_proc = X_test_proc[eval_idx]
        y_test_eval = np.asarray(y_test_eval)[eval_idx]
    preds = model.predict(X_test_proc)
    metrics_clean = classification_metrics(y_test_eval, preds, cfg.is_multiclass)

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
        if model_name == "tabpfn":
            _, X_shift_proc, _, _ = build_tabpfn_inputs(X_train, X_shift)
        else:
            X_shift_proc = preprocessor.transform(X_shift)
            if hasattr(X_shift_proc, "toarray"):
                X_shift_proc = X_shift_proc.toarray()
            if model_name in {"tabnet", "apt"}:
                X_shift_proc = np.asarray(X_shift_proc, dtype=np.float32)
        if eval_idx is not None:
            X_shift_proc = X_shift_proc[eval_idx]
        preds_shift = model.predict(X_shift_proc)
        metrics_shift = classification_metrics(y_test_eval, preds_shift, cfg.is_multiclass)

    stability = None
    if shap is not None and model_name not in {"tabpfn", "apt"}:
        try:
            sample_cap = max_shap_samples
            if model_name in {"mlp", "tabnet"}:
                sample_cap = min(sample_cap, 50)
            sample_n = min(sample_cap, X_test_proc.shape[0])
            idx = rng.choice(X_test_proc.shape[0], size=sample_n, replace=False)
            if model_name != "tabpfn":
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
        except Exception as exc:
            with open(out_dir / "shap_error.json", "w") as f:
                json.dump({"error": str(exc)}, f, indent=2)
            print(f"⚠️ SHAP failed but predictive metrics were kept for {cfg.name}/{model_name}: {exc}")

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
