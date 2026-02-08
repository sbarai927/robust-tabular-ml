"""
Objective:
Study how tree-based models and neural networks handle missing data and high-cardinality categorical features under controlled stress scenarios.

Dataset Selection:
Use the following 4 tabular datasets for a diverse mix of tasks and properties:
1. diamonds_v2.csv (regression, mixed features, moderate size)
2. telco_churn_classification.csv (binary classification, high-cardinality categorical, missing values)
3. drybean_multiclass_classification.csv (multiclass classification, categorical imbalance)
4. mnist_tabular_digits.csv (multiclass classification, clean numeric input)

Models to Use:
- Random Forest
- LightGBM
- CatBoost
- MLP (Multilayer Perceptron)

Tasks to Perform:
1. Load each dataset and identify the target column (same as in prior work).
2. For each model and dataset:
   - Run evaluation under three scenarios:
     a. **Original (clean) data**
     b. **Injected missingness**: Randomly set 10–20% of values to NaN in non-target features.
     c. **High-cardinality encoding stress**: Introduce synthetic categorical columns with 100+ unique values or duplicate existing high-cardinality ones.

3. Preprocess data appropriately:
   - Impute missing values (mean for numeric, mode or 'Missing' for categorical).
   - Encode categorical features using suitable methods for each model (LabelEncoder for trees, OneHot or Embedding for MLP).

4. Train models with previously tuned/best hyperparameters (no additional HPO in this step).
5. Evaluate using:
   - Accuracy / MAE / RMSE / R² depending on task
   - Training time
   - Model size on disk (after saving)
   - SHAP value delta (compare SHAP importance rankings between clean vs stressed versions)

6. Save the following outputs under `results/robustness_challenges/`:
   - CSV of performance metrics for each stress type and model: `metrics_{dataset}.csv`
   - JSON files with model size and training time
   - SHAP summary difference metrics (e.g., top-k feature shift) in `shap_delta_{dataset}_{model}.json`

Notes:
- Reuse utilities and preprocessing functions if already implemented in the codebase.
- Avoid retraining models unnecessarily. Cache models and predictions where applicable.
- Use tqdm or similar for progress tracking if long loops are involved.


Explicitly save the SHAP top-k features for both clean and perturbed versions, not just the deltas.

Log which features were randomly selected for missingness and high-cardinality injection, to ensure reproducibility and later debugging.

"""

import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

os.environ.setdefault("KMP_USE_SHM", "0")
os.environ.setdefault("KMP_SHM_DISABLE", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

try:
    import torch
except Exception:
    torch = None

try:
    from pytorch_tabnet.tab_model import TabNetClassifier, TabNetRegressor
except Exception:
    TabNetClassifier = TabNetRegressor = None

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

try:
    import shap
except Exception:
    shap = None

LGBMClassifier = LGBMRegressor = None

try:
    from catboost import CatBoostClassifier, CatBoostRegressor
    import catboost as cb
except Exception:
    CatBoostClassifier = CatBoostRegressor = None
    cb = None

xgb = None

try:
    from tqdm import tqdm
except Exception:
    tqdm = None

RANDOM_STATE = 42
ROBUST_DIR = Path("results/robustness_challenges")
ROBUST_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class DatasetConfig:
    name: str
    path: Path
    target: str
    task: str  # "regression" or "classification"
    is_multiclass: bool = False


DATASETS: Dict[str, DatasetConfig] = {
    "diamonds_v2": DatasetConfig(
        "diamonds_v2", Path("data/diamonds_v2.csv"), "total_sales_price", "regression"
    ),
    "telco": DatasetConfig(
        "telco", Path("data/telco_churn_classification.csv"), "Churn", "classification"
    ),
    "drybean": DatasetConfig(
        "drybean",
        Path("data/drybean_multiclass_classification.csv"),
        "Class",
        "classification",
        True,
    ),
    "mnist": DatasetConfig(
        "mnist",
        Path("data/mnist_tabular_digits.csv"),
        "label",
        "classification",
        True,
    ),
}

MODELS = {"rf", "lgbm", "catboost", "xgboost", "mlp", "tabnet"}
SCENARIOS = ("clean", "missingness", "high_cardinality")


def load_dataset(cfg: DatasetConfig) -> Tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(cfg.path)
    df = df.dropna(subset=[cfg.target]).reset_index(drop=True)
    X = df.drop(columns=[cfg.target])
    y = df[cfg.target]
    return X, y


def split_dataset(
    X: pd.DataFrame, y: pd.Series, cfg: DatasetConfig
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    stratify = y if cfg.task == "classification" else None
    return train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=stratify,
    )


def inject_missingness(
    X: pd.DataFrame, rate: float, columns: List[str], rng: np.random.Generator
) -> pd.DataFrame:
    # Randomly mask values with NaN for a selected subset of columns.
    X_masked = X.copy()
    for col in columns:
        col_mask = rng.random(X_masked[col].shape) < rate
        X_masked.loc[col_mask, col] = np.nan
    return X_masked


def add_high_cardinality_features(
    X: pd.DataFrame,
    base_columns: List[str],
    num_cols: int,
    n_unique: int,
    rng: np.random.Generator,
) -> Tuple[pd.DataFrame, List[str]]:
    # Add synthetic categorical columns derived from selected base columns.
    X_aug = X.copy()
    new_cols = []
    for idx in range(num_cols):
        base = base_columns[idx % len(base_columns)]
        values = [f"hc_{idx}_{i}" for i in range(n_unique)]
        suffix = rng.choice(values, size=len(X_aug), replace=True)
        X_aug[f"high_card_{idx}"] = X_aug[base].astype(str) + "_" + suffix
        new_cols.append(f"high_card_{idx}")
    return X_aug, new_cols


def build_preprocessor(X: pd.DataFrame, model_name: str) -> ColumnTransformer:
    # Separate numeric vs categorical columns with simple imputers.
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = [c for c in X.columns if c not in cat_cols]

    if model_name == "mlp":
        num_pipeline = [
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()),
        ]
        cat_pipeline = [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    else:
        num_pipeline = [
            ("imputer", SimpleImputer(strategy="mean")),
        ]
        cat_pipeline = [
            ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
            ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
        ]

    transformers = []
    if num_cols:
        transformers.append(("num", Pipeline(num_pipeline), num_cols))
    if cat_cols:
        transformers.append(("cat", Pipeline(cat_pipeline), cat_cols))
    return ColumnTransformer(transformers)


def classification_metrics(y_true, y_pred, multiclass=False) -> Dict[str, float]:
    average = "macro" if multiclass else "binary"
    if multiclass:
        return {
            "Accuracy": float(accuracy_score(y_true, y_pred)),
            "F1": float(f1_score(y_true, y_pred, average=average)),
        }
    le = LabelEncoder()
    y_true_enc = le.fit_transform(np.asarray(y_true))
    y_pred_enc = le.transform(np.asarray(y_pred))
    return {
        "Accuracy": float(accuracy_score(y_true_enc, y_pred_enc)),
        "F1": float(f1_score(y_true_enc, y_pred_enc, average=average, pos_label=1)),
    }


def regression_metrics(y_true, y_pred) -> Dict[str, float]:
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
    }


def load_best_params(cfg: DatasetConfig, model_name: str) -> Dict[str, object]:
    params_path = Path("results/hpo") / cfg.name / model_name / "best_params.json"
    if not params_path.exists():
        raise FileNotFoundError(f"Missing best_params.json at {params_path}")
    with open(params_path, "r") as f:
        return json.load(f)


def split_tabnet_params(params: Dict[str, object]) -> Tuple[Dict[str, object], Dict[str, object]]:
    params = params.copy()
    batch_size = int(params.pop("batch_size", 1024))
    virtual_batch_size = int(params.pop("virtual_batch_size", 128))
    if virtual_batch_size > batch_size:
        virtual_batch_size = batch_size
    fit_params = {
        "batch_size": batch_size,
        "virtual_batch_size": virtual_batch_size,
    }
    lr = float(params.pop("learning_rate", 1e-3))
    params["optimizer_params"] = {"lr": lr}
    return params, fit_params


def tabnet_regression_target(y: pd.Series) -> np.ndarray:
    return np.asarray(y).reshape(-1, 1)


def build_model(cfg: DatasetConfig, model_name: str, params: Dict[str, object]):
    is_reg = cfg.task == "regression"
    if model_name == "rf":
        base = RandomForestRegressor if is_reg else RandomForestClassifier
        return base(random_state=RANDOM_STATE, n_jobs=1, **params)
    if model_name == "lgbm":
        global LGBMClassifier, LGBMRegressor
        if LGBMClassifier is None:
            try:
                from lightgbm import LGBMClassifier as _LGBMC, LGBMRegressor as _LGBMR
                LGBMClassifier, LGBMRegressor = _LGBMC, _LGBMR
            except Exception:
                raise RuntimeError("lightgbm not installed")
        base = LGBMRegressor if is_reg else LGBMClassifier
        return base(random_state=RANDOM_STATE, n_jobs=1, **params)
    if model_name == "catboost":
        if CatBoostClassifier is None:
            raise RuntimeError("catboost not installed")
        base = CatBoostRegressor if is_reg else CatBoostClassifier
        return base(random_seed=RANDOM_STATE, verbose=False, **params)
    if model_name == "xgboost":
        global xgb
        if xgb is None:
            try:
                import xgboost as _xgb
                xgb = _xgb
            except Exception:
                raise RuntimeError("xgboost not installed")
        if is_reg:
            return xgb.XGBRegressor(
                random_state=RANDOM_STATE,
                n_jobs=1,
                eval_metric="rmse",
                **params,
            )
        metric = "mlogloss" if cfg.is_multiclass else "logloss"
        return xgb.XGBClassifier(
            random_state=RANDOM_STATE,
            n_jobs=1,
            eval_metric=metric,
            **params,
        )
    if model_name == "mlp":
        if "hidden_layer_sizes" in params:
            hls = params["hidden_layer_sizes"]
            if isinstance(hls, list):
                params["hidden_layer_sizes"] = tuple(hls)
        else:
            h1 = params.pop("h1")
            h2 = params.pop("h2")
            params["hidden_layer_sizes"] = tuple(sorted([h1, h2], reverse=True))
        base = MLPRegressor if is_reg else MLPClassifier
        return base(
            max_iter=200,
            random_state=RANDOM_STATE,
            early_stopping=True,
            **params,
        )
    if model_name == "tabnet":
        if TabNetClassifier is None:
            raise RuntimeError("pytorch-tabnet not installed")
        device_name = "cuda" if torch is not None and torch.cuda.is_available() else "cpu"
        params = params.copy()
        params.update({"seed": RANDOM_STATE, "device_name": device_name})
        params.setdefault("mask_type", "entmax")
        base = TabNetRegressor if is_reg else TabNetClassifier
        return base(**params)
    raise ValueError(f"Unsupported model {model_name}")


def fit_and_evaluate(
    cfg: DatasetConfig,
    model_name: str,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    params: Dict[str, object],
    scenario: str,
    out_dir: Path,
) -> Dict[str, object]:
    # Preprocess data for the chosen model.
    preprocessor = build_preprocessor(X_train, model_name)
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)
    if hasattr(X_train_proc, "toarray"):
        X_train_proc = X_train_proc.toarray()
        X_test_proc = X_test_proc.toarray()
    if model_name == "tabnet":
        X_train_proc = np.asarray(X_train_proc, dtype=np.float32)
        X_test_proc = np.asarray(X_test_proc, dtype=np.float32)

    # Train model with best parameters and track time.
    start = time.time()
    if model_name == "tabnet":
        model_params, fit_params = split_tabnet_params(params)
        fit_params.setdefault("max_epochs", 5)
        fit_params.setdefault("patience", 3)
        model = build_model(cfg, model_name, model_params)
        if cfg.task == "classification":
            le = LabelEncoder()
            y_train_enc = le.fit_transform(np.asarray(y_train))
            y_test_enc = le.transform(np.asarray(y_test))
            X_train_inner, X_valid, y_train_inner, y_valid = train_test_split(
                X_train_proc,
                y_train_enc,
                test_size=0.1,
                random_state=RANDOM_STATE,
                stratify=y_train_enc,
            )
            model.fit(
                X_train_inner,
                y_train_inner,
                eval_set=[(X_valid, y_valid)],
                **fit_params,
            )
            preds = model.predict(X_test_proc)
            y_eval = y_test_enc
        else:
            y_train_arr = np.asarray(y_train)
            X_train_inner, X_valid, y_train_inner, y_valid = train_test_split(
                X_train_proc,
                y_train_arr,
                test_size=0.1,
                random_state=RANDOM_STATE,
            )
            model.fit(
                X_train_inner,
                tabnet_regression_target(pd.Series(y_train_inner)),
                eval_set=[(X_valid, tabnet_regression_target(pd.Series(y_valid)))],
                **fit_params,
            )
            preds = model.predict(X_test_proc)
            y_eval = y_test
    elif model_name == "xgboost" and cfg.task == "classification":
        model = build_model(cfg, model_name, params.copy())
        le = LabelEncoder()
        y_train_enc = le.fit_transform(np.asarray(y_train))
        y_test_enc = le.transform(np.asarray(y_test))
        model.fit(X_train_proc, y_train_enc)
        preds = model.predict(X_test_proc)
        y_eval = y_test_enc
    else:
        model = build_model(cfg, model_name, params.copy())
        model.fit(X_train_proc, y_train)
        preds = model.predict(X_test_proc)
        y_eval = y_test
    elapsed = time.time() - start

    # Evaluate and gather metrics.
    if cfg.task == "regression":
        metrics = regression_metrics(y_eval, preds)
    else:
        metrics = classification_metrics(y_eval, preds, multiclass=cfg.is_multiclass)

    # Save model and compute size.
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "best_model.pkl"
    joblib.dump(model, model_path)
    model_size_mb = model_path.stat().st_size / (1024 * 1024)

    # Save timing and size metadata.
    meta = {
        "train_time_sec": float(elapsed),
        "model_size_mb": float(model_size_mb),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "params": params,
        "scenario": scenario,
    }
    with open(out_dir / f"meta_{scenario}.json", "w") as f:
        json.dump(meta, f, indent=2)

    metrics.update(meta)
    metrics.update({"scenario": scenario, "model": model_name})
    return metrics, model, preprocessor, X_test_proc


def _feature_names(preprocessor: ColumnTransformer) -> List[str]:
    names = []
    for name, transformer, cols in preprocessor.transformers_:
        if name == "cat" and hasattr(transformer.named_steps["encoder"], "get_feature_names_out"):
            enc = transformer.named_steps["encoder"]
            names.extend(enc.get_feature_names_out(cols))
        elif name == "num":
            names.extend(cols)
        else:
            names.extend(cols)
    return [str(n) for n in names]


def compute_shap_importance(
    model,
    model_name: str,
    X_sample: np.ndarray,
    feature_names: List[str],
    max_kernel_samples: int = 100,
) -> Dict[str, float]:
    # Compute mean absolute SHAP values for a small sample.
    if shap is None:
        raise RuntimeError("shap not installed")
    if model_name in {"rf", "lgbm", "catboost", "xgboost"}:
        if model.__class__.__name__.startswith("CatBoost") and cb is not None:
            pool = cb.Pool(X_sample, feature_names=feature_names)
            values = model.get_feature_importance(pool, type="ShapValues")
            values = np.asarray(values)
            if values.ndim == 3:
                values = np.mean(np.abs(values[:, :, :-1]), axis=0)
                if values.ndim == 2:
                    values = values.mean(axis=1)
            else:
                values = np.abs(values[:, :-1]).mean(axis=0)
            limit = min(len(feature_names), len(values))
            return {feature_names[i]: float(values[i]) for i in range(limit)}
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
        limit = min(len(feature_names), len(values))
        return {feature_names[i]: float(values[i]) for i in range(limit)}

    # MLP/TabNet: kernel explainer on a small sample.
    if model_name == "tabnet":
        predict_fn = model.predict_proba if hasattr(model, "predict_proba") else model.predict
        X_sample = np.asarray(X_sample, dtype=np.float32)
    else:
        predict_fn = model.predict_proba if hasattr(model, "predict_proba") else model.predict
    sample_n = min(max_kernel_samples, X_sample.shape[0])
    background = shap.utils.sample(X_sample, min(50, sample_n))
    explainer = shap.KernelExplainer(predict_fn, background)
    shap_values = explainer.shap_values(
        X_sample[:sample_n], nsamples=30, l1_reg="num_features(10)"
    )
    if isinstance(shap_values, list):
        values = np.mean([np.abs(v) for v in shap_values], axis=0)
    else:
        values = np.abs(shap_values)
    if values.ndim == 3:
        values = values.mean(axis=(0, 2))
    else:
        values = values.mean(axis=0)
    limit = min(len(feature_names), len(values))
    return {feature_names[i]: float(values[i]) for i in range(limit)}


def top_k_features(importance: Dict[str, float], k: int = 10) -> List[str]:
    return [f for f, _ in sorted(importance.items(), key=lambda x: x[1], reverse=True)[:k]]


def save_shap_delta(
    cfg: DatasetConfig,
    model_name: str,
    clean_top: List[str],
    stressed_top: Dict[str, List[str]],
) -> None:
    # Compare clean vs stressed top-k SHAP features.
    output = {"clean_top_k": clean_top, "comparisons": {}}
    clean_set = set(clean_top)
    for scenario, top_list in stressed_top.items():
        stressed_set = set(top_list)
        overlap = clean_set & stressed_set
        union = clean_set | stressed_set
        output["comparisons"][scenario] = {
            "top_k": top_list,
            "overlap_count": len(overlap),
            "jaccard": float(len(overlap) / max(len(union), 1)),
        }
    out_path = ROBUST_DIR / f"shap_delta_{cfg.name}_{model_name}.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)


def save_shap_top_k(
    cfg: DatasetConfig,
    model_name: str,
    top_k: Dict[str, List[str]],
) -> None:
    out_path = ROBUST_DIR / f"shap_top_k_{cfg.name}_{model_name}.json"
    with open(out_path, "w") as f:
        json.dump(top_k, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run robustness challenges for missingness/high-cardinality.")
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS.keys()))
    parser.add_argument("--models", nargs="+", default=sorted(MODELS))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    datasets = [d for d in args.datasets if d in DATASETS]
    models = [m for m in args.models if m in MODELS]
    iterator = tqdm(datasets, desc="Datasets") if tqdm else datasets

    for dataset_name in iterator:
        cfg = DATASETS[dataset_name]
        metrics_rows: List[Dict[str, object]] = []
        X, y = load_dataset(cfg)
        X_train, X_test, y_train, y_test = split_dataset(X, y, cfg)
        rng = np.random.default_rng(RANDOM_STATE)

        for model_name in models:
            try:
                params = load_best_params(cfg, model_name)
            except Exception as exc:
                print(f"⚠️ Missing params for {cfg.name}/{model_name}: {exc}")
                continue
            model_output_dir = ROBUST_DIR / cfg.name / model_name
            model_output_dir.mkdir(parents=True, exist_ok=True)

            shap_clean_top = None
            shap_comparisons: Dict[str, List[str]] = {}
            shap_top_k: Dict[str, List[str]] = {}

            for scenario in SCENARIOS:
                scenario_dir = model_output_dir / scenario
                metrics_path = scenario_dir / "metrics.json"
                if metrics_path.exists() and not args.overwrite:
                    with open(metrics_path, "r") as f:
                        metrics_rows.append(json.load(f))
                    continue

                if scenario == "clean":
                    X_train_s, X_test_s = X_train.copy(), X_test.copy()
                elif scenario == "missingness":
                    scenario_dir.mkdir(parents=True, exist_ok=True)
                    cols = X_train.columns.tolist()
                    selected = rng.choice(cols, size=max(1, int(0.2 * len(cols))), replace=False)
                    X_train_s = inject_missingness(X_train, rate=0.15, columns=selected, rng=rng)
                    X_test_s = inject_missingness(X_test, rate=0.15, columns=selected, rng=rng)
                    with open(scenario_dir / "injection_log.json", "w") as f:
                        json.dump(
                            {
                                "missingness_rate": 0.15,
                                "missingness_columns": [str(c) for c in selected],
                            },
                            f,
                            indent=2,
                        )
                else:
                    scenario_dir.mkdir(parents=True, exist_ok=True)
                    cat_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
                    base_cols = cat_cols if cat_cols else X_train.columns.tolist()
                    selected = rng.choice(base_cols, size=min(2, len(base_cols)), replace=False)
                    X_train_s, new_cols = add_high_cardinality_features(
                        X_train, base_columns=selected, num_cols=2, n_unique=150, rng=rng
                    )
                    X_test_s, _ = add_high_cardinality_features(
                        X_test, base_columns=selected, num_cols=2, n_unique=150, rng=rng
                    )
                    with open(scenario_dir / "injection_log.json", "w") as f:
                        json.dump(
                            {
                                "high_cardinality_base_columns": [str(c) for c in selected],
                                "high_cardinality_new_columns": new_cols,
                                "high_cardinality_unique_values": 150,
                            },
                            f,
                            indent=2,
                        )

                try:
                    metrics, model, preprocessor, X_test_proc = fit_and_evaluate(
                        cfg,
                        model_name,
                        X_train_s,
                        X_test_s,
                        y_train,
                        y_test,
                        params,
                        scenario,
                        scenario_dir,
                    )
                    metrics["dataset"] = cfg.name
                    with open(metrics_path, "w") as f:
                        json.dump(metrics, f, indent=2)
                    metrics_rows.append(metrics)
                except Exception as exc:
                    warn_path = scenario_dir / "error.json"
                    warn_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(warn_path, "w") as f:
                        json.dump({"error": str(exc)}, f, indent=2)
                    print(f"⚠️ Failed {cfg.name}/{model_name}/{scenario}: {exc}")
                    continue

                # Compute SHAP delta on a small sample.
                if shap is not None:
                    sample_cap = 200 if model_name in {"rf", "lgbm", "catboost", "xgboost"} else 50
                    sample_idx = np.random.default_rng(RANDOM_STATE).choice(
                        X_test_proc.shape[0], size=min(sample_cap, X_test_proc.shape[0]), replace=False
                    )
                    X_sample = X_test_proc[sample_idx]
                    feature_names = _feature_names(preprocessor)
                    if X_sample.shape[1] != len(feature_names):
                        feature_names = [f"f{i}" for i in range(X_sample.shape[1])]
                    try:
                        importance = compute_shap_importance(
                            model, model_name, X_sample, feature_names
                        )
                        top_list = top_k_features(importance, k=10)
                        shap_top_k[scenario] = top_list
                        if scenario == "clean":
                            shap_clean_top = top_list
                        else:
                            shap_comparisons[scenario] = top_list
                    except Exception as exc:
                        print(f"⚠️ SHAP failed {cfg.name}/{model_name}/{scenario}: {exc}")
                        continue

            if shap_top_k:
                save_shap_top_k(cfg, model_name, shap_top_k)
            if shap_top_k and "clean" in shap_top_k:
                comparisons = {
                    k: v for k, v in shap_top_k.items() if k != "clean"
                }
                if comparisons:
                    save_shap_delta(cfg, model_name, shap_top_k["clean"], comparisons)

        # Rebuild aggregate metrics from disk to avoid partial overwrites.
        aggregate_rows: List[Dict[str, object]] = []
        dataset_dir = ROBUST_DIR / cfg.name
        for metrics_path in sorted(dataset_dir.glob("*/*/metrics.json")):
            try:
                with open(metrics_path, "r") as f:
                    row = json.load(f)
                aggregate_rows.append(row)
            except Exception:
                continue
        metrics_df = pd.DataFrame(aggregate_rows if aggregate_rows else metrics_rows)
        if not metrics_df.empty:
            cols = ["dataset", "model", "scenario"] + [
                c for c in metrics_df.columns if c not in {"dataset", "model", "scenario"}
            ]
            metrics_df = metrics_df[[c for c in cols if c in metrics_df.columns]]
            metrics_df.to_csv(ROBUST_DIR / f"metrics_{cfg.name}.csv", index=False)


if __name__ == "__main__":
    main()
