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
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

try:
    import torch
except Exception:
    torch = None

try:
    from pytorch_tabnet.tab_model import TabNetClassifier, TabNetRegressor
except Exception:
    TabNetClassifier = TabNetRegressor = None

try:
    from tabpfn import TabPFNClassifier
except Exception:
    TabPFNClassifier = None

try:
    from apt.model import APTClassifier
except Exception:
    APTClassifier = None

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

MODELS = {"rf", "lgbm", "catboost", "xgboost", "mlp", "tabnet", "tabpfn", "apt"}
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

    if model_name in {"mlp", "apt"}:
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


def build_tabpfn_inputs(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    num_means: Dict[str, float] | None = None,
    categories: Dict[str, pd.Index] | None = None,
) -> Tuple[np.ndarray, np.ndarray, List[int], Dict[str, float], Dict[str, pd.Index]]:
    cat_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = [c for c in X_train.columns if c not in cat_cols]
    if num_means is None:
        num_means = {c: float(X_train[c].mean()) for c in num_cols}
    if categories is None:
        categories = {
            col: pd.Categorical(X_train[col].astype(str).fillna("missing")).categories
            for col in cat_cols
        }
    effective_num_means = {c: float(num_means.get(c, X_train[c].mean())) for c in num_cols}
    effective_categories = {
        col: categories.get(
            col,
            pd.Categorical(X_train[col].astype(str).fillna("missing")).categories,
        )
        for col in cat_cols
    }

    def encode(df: pd.DataFrame) -> np.ndarray:
        num = (
            df[num_cols].astype(float).fillna(pd.Series(effective_num_means)).to_numpy()
            if num_cols
            else np.empty((len(df), 0))
        )
        if cat_cols:
            cat_arrays = []
            for col in cat_cols:
                cat = pd.Categorical(
                    df[col].astype(str).fillna("missing"),
                    categories=effective_categories[col],
                )
                cat_arrays.append(cat.codes.reshape(-1, 1))
            cat_mat = np.hstack(cat_arrays)
        else:
            cat_mat = np.empty((len(df), 0))
        return np.hstack([num, cat_mat]).astype(np.float32)

    X_train_proc = encode(X_train)
    X_test_proc = encode(X_test)
    cat_indices = list(range(len(num_cols), len(num_cols) + len(cat_cols)))
    return X_train_proc, X_test_proc, cat_indices, num_means, categories


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
    if model_name == "tabpfn":
        if TabPFNClassifier is None:
            raise RuntimeError("tabpfn not installed")
        if is_reg:
            raise RuntimeError("tabpfn supports classification only")
        device = "cuda" if torch is not None and torch.cuda.is_available() else "cpu"
        return TabPFNClassifier(
            device=device,
            random_state=RANDOM_STATE,
            ignore_pretraining_limits=True,
        )
    if model_name == "apt":
        if APTClassifier is None:
            raise RuntimeError("APT not installed")
        if is_reg:
            raise RuntimeError("APT regression is not supported")
        device = "cuda" if torch is not None and torch.cuda.is_available() else "cpu"
        return APTClassifier(device=device)
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
    tabpfn_cat_indices: List[int] = []
    if model_name == "tabpfn":
        X_train_proc, X_test_proc, tabpfn_cat_indices, _, _ = build_tabpfn_inputs(
            X_train, X_test
        )
        preprocessor = None
    else:
        preprocessor = build_preprocessor(X_train, model_name)
        X_train_proc = preprocessor.fit_transform(X_train)
        X_test_proc = preprocessor.transform(X_test)
        if hasattr(X_train_proc, "toarray"):
            X_train_proc = X_train_proc.toarray()
            X_test_proc = X_test_proc.toarray()
        if model_name in {"tabnet", "apt"}:
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
    elif model_name == "tabpfn":
        if cfg.task == "regression":
            raise RuntimeError("tabpfn supports classification only")
        device = "cuda" if torch is not None and torch.cuda.is_available() else "cpu"
        model = TabPFNClassifier(
            device=device,
            random_state=RANDOM_STATE,
            ignore_pretraining_limits=True,
            categorical_features_indices=tabpfn_cat_indices or None,
            fit_mode="low_memory",
        )
        le = LabelEncoder()
        y_train_enc = le.fit_transform(np.asarray(y_train))
        y_test_enc = le.transform(np.asarray(y_test))
        model.fit(X_train_proc, y_train_enc)
        preds = model.predict(X_test_proc)
        y_eval = y_test_enc
    elif model_name == "apt":
        if cfg.task == "regression":
            raise RuntimeError("APT regression is not supported")
        model = build_model(cfg, model_name, params.copy())
        le = LabelEncoder()
        y_train_enc = le.fit_transform(np.asarray(y_train))
        y_test_enc = le.transform(np.asarray(y_test))
        model.fit(X_train_proc, y_train_enc, tune=False, process_data=True)
        preds = model.predict(X_test_proc)
        y_eval = y_test_enc
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
    if model_name == "apt":
        if torch is None:
            raise RuntimeError("torch required to save APT model")
        torch.save(
            {
                "state_dict": model.state_dict(),
                "x_train": getattr(model, "x_train", None),
                "y_train": getattr(model, "y_train", None),
                "x_encoder": getattr(model, "x_encoder", None),
                "y_encoder": getattr(model, "y_encoder", None),
                "feature_perm": getattr(model, "feature_perm", None),
            },
            model_path,
        )
    else:
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


def _collapse_one_hot_importance(
    importance: Dict[str, float], base_cols: List[str]
) -> Dict[str, float]:
    aggregated: Dict[str, float] = {}
    for name, value in importance.items():
        base_match = None
        for base in base_cols:
            prefix = f"{base}_"
            if name.startswith(prefix):
                base_match = base
                break
        key = base_match if base_match is not None else name
        aggregated[key] = aggregated.get(key, 0.0) + float(value)
    return aggregated


def compute_shap_importance(
    model,
    model_name: str,
    X_sample: np.ndarray,
    feature_names: List[str],
    one_hot_base_cols: List[str] | None = None,
    max_kernel_samples: int = 100,
    kernel_nsamples: int | None = None,
    background_size: int | None = None,
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
        importance = {feature_names[i]: float(values[i]) for i in range(limit)}
        if model_name == "mlp" and one_hot_base_cols:
            return _collapse_one_hot_importance(importance, one_hot_base_cols)
        return importance

    # MLP/TabNet: kernel explainer on a small sample.
    if model_name == "tabnet":
        predict_fn = model.predict_proba if hasattr(model, "predict_proba") else model.predict
        X_sample = np.asarray(X_sample, dtype=np.float32)
    else:
        if model_name in {"tabpfn", "apt"}:
            predict_fn = model.predict
        else:
            predict_fn = model.predict_proba if hasattr(model, "predict_proba") else model.predict
    sample_n = min(max_kernel_samples, X_sample.shape[0])
    bg_size = background_size if background_size is not None else (1 if model_name in {"tabpfn", "apt"} else 50)
    background = shap.utils.sample(X_sample, min(bg_size, sample_n))
    explainer = shap.KernelExplainer(predict_fn, background)
    nsamples = kernel_nsamples if kernel_nsamples is not None else (1 if model_name in {"tabpfn", "apt"} else 30)
    shap_values = explainer.shap_values(
        X_sample[:sample_n], nsamples=nsamples, l1_reg="num_features(10)"
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
    importance = {feature_names[i]: float(values[i]) for i in range(limit)}
    if model_name == "mlp" and one_hot_base_cols:
        return _collapse_one_hot_importance(importance, one_hot_base_cols)
    return importance


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


def tabpfn_feature_names(X: pd.DataFrame) -> List[str]:
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = [c for c in X.columns if c not in cat_cols]
    return [str(c) for c in num_cols + cat_cols]


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
    parser.add_argument(
        "--shap-only",
        action="store_true",
        help="Recompute SHAP top-k/delta using existing models without retraining.",
    )
    parser.add_argument(
        "--allow-tabpfn-shap",
        action="store_true",
        help="Enable SHAP for TabPFN/APT (slow).",
    )
    parser.add_argument("--tabpfn-shap-samples", type=int, default=50)
    parser.add_argument("--tabpfn-shap-nsamples", type=int, default=5)
    parser.add_argument("--tabpfn-shap-background", type=int, default=10)
    parser.add_argument(
        "--tabpfn-importance",
        choices=["none", "perm", "shap"],
        default="none",
        help="Importance method for TabPFN/APT (perm is faster than SHAP).",
    )
    parser.add_argument("--tabpfn-perm-samples", type=int, default=50)
    parser.add_argument("--tabpfn-perm-repeats", type=int, default=2)
    parser.add_argument("--tabpfn-perm-features", type=int, default=50)
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
            if cfg.task == "regression" and model_name in {"tabpfn", "apt"}:
                print(f"⚠️ Skipping {cfg.name}/{model_name}: regression not supported.")
                continue
            if model_name == "tabpfn" and TabPFNClassifier is None:
                print(f"⚠️ Skipping {cfg.name}/{model_name}: tabpfn not installed.")
                continue
            if model_name == "apt" and APTClassifier is None:
                print(f"⚠️ Skipping {cfg.name}/{model_name}: apt not installed.")
                continue
            params = None
            if not args.shap_only:
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
            cached_model = None
            cached_model_path = None
            cached_preprocessor = None
            tabpfn_num_means = None
            tabpfn_categories = None
            if not args.shap_only and model_name in {"tabpfn", "apt"}:
                cached_model_path = Path("results/hpo") / cfg.name / model_name / "best_model.pkl"
                if not cached_model_path.exists():
                    print(f"⚠️ Missing HPO model for {cfg.name}/{model_name}: {cached_model_path}")
                    continue
                try:
                    cached_model = (
                        load_apt_artifact(cached_model_path)
                        if model_name == "apt"
                        else joblib.load(cached_model_path)
                    )
                    if model_name == "apt":
                        cached_preprocessor = build_preprocessor(X_train, model_name)
                        cached_preprocessor.fit(X_train)
                    if model_name == "tabpfn":
                        _, _, _, tabpfn_num_means, tabpfn_categories = build_tabpfn_inputs(
                            X_train, X_train
                        )
                except Exception as exc:
                    print(f"⚠️ Failed to load HPO model for {cfg.name}/{model_name}: {exc}")
                    continue

            for scenario in SCENARIOS:
                scenario_dir = model_output_dir / scenario
                metrics_path = scenario_dir / "metrics.json"
                if not args.shap_only and metrics_path.exists() and not args.overwrite:
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
                    if model_name in {"tabpfn", "apt"}:
                        X_train_s = X_train.copy()
                        X_test_s = X_test.copy()
                        if not cat_cols and model_name == "apt":
                            selected = []
                        for idx, col in enumerate(selected):
                            values = [f"hc_{idx}_{i}" for i in range(150)]
                            X_train_s[col] = X_train_s[col].astype(str) + "_" + rng.choice(
                                values, size=len(X_train_s), replace=True
                            )
                            X_test_s[col] = X_test_s[col].astype(str) + "_" + rng.choice(
                                values, size=len(X_test_s), replace=True
                            )
                        with open(scenario_dir / "injection_log.json", "w") as f:
                            json.dump(
                                {
                                    "high_cardinality_base_columns": [str(c) for c in selected],
                                    "high_cardinality_new_columns": [],
                                    "high_cardinality_unique_values": 150,
                                    "mode": "overwrite",
                                    "skipped_no_categorical": not cat_cols and model_name == "apt",
                                },
                                f,
                                indent=2,
                            )
                    else:
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

                if args.shap_only:
                    model_path = scenario_dir / "best_model.pkl"
                    if not model_path.exists():
                        print(f"⚠️ Missing model for SHAP: {model_path}")
                        continue
                    try:
                        if model_name == "tabpfn":
                            X_train_proc, X_test_proc, _, _, _ = build_tabpfn_inputs(
                                X_train_s, X_test_s
                            )
                            preprocessor = None
                            model = joblib.load(model_path)
                        else:
                            preprocessor = build_preprocessor(X_train_s, model_name)
                            X_train_proc = preprocessor.fit_transform(X_train_s)
                            X_test_proc = preprocessor.transform(X_test_s)
                            if hasattr(X_train_proc, "toarray"):
                                X_test_proc = X_test_proc.toarray()
                            if model_name in {"tabnet", "apt"}:
                                X_test_proc = np.asarray(X_test_proc, dtype=np.float32)
                            if model_name == "apt":
                                model = load_apt_artifact(model_path)
                            else:
                                model = joblib.load(model_path)
                        # Align MLP input shape with expected model input if schema drifted.
                        if model_name == "mlp" and hasattr(model, "coefs_"):
                            expected = model.coefs_[0].shape[0]
                            current = X_test_proc.shape[1]
                            if current < expected:
                                pad = np.zeros((X_test_proc.shape[0], expected - current))
                                X_test_proc = np.hstack([X_test_proc, pad])
                            elif current > expected:
                                X_test_proc = X_test_proc[:, :expected]
                    except Exception as exc:
                        print(f"⚠️ Failed SHAP prep {cfg.name}/{model_name}/{scenario}: {exc}")
                        continue
                else:
                    try:
                        if model_name in {"tabpfn", "apt"} and cached_model is not None:
                            model = cached_model
                            if model_name == "tabpfn":
                                X_train_proc, X_test_proc, _, _, _ = build_tabpfn_inputs(
                                    X_train_s,
                                    X_test_s,
                                    num_means=tabpfn_num_means,
                                    categories=tabpfn_categories,
                                )
                                preprocessor = None
                            else:
                                preprocessor = cached_preprocessor
                                X_test_proc = preprocessor.transform(X_test_s)
                                if hasattr(X_test_proc, "toarray"):
                                    X_test_proc = X_test_proc.toarray()
                                if model_name == "apt":
                                    X_test_proc = np.asarray(X_test_proc, dtype=np.float32)
                            eval_idx = None
                            if model_name in {"tabpfn", "apt"} and len(X_test_proc) > 50:
                                eval_idx = rng.choice(len(X_test_proc), size=50, replace=False)
                                X_eval = X_test_proc[eval_idx]
                            else:
                                X_eval = X_test_proc
                            preds = cached_model.predict(X_eval)
                            y_eval = y_test
                            if cfg.task == "classification":
                                le = LabelEncoder()
                                y_eval = le.fit_transform(np.asarray(y_test))
                            if eval_idx is not None:
                                y_eval = np.asarray(y_eval)[eval_idx]
                            if cfg.task == "regression":
                                metrics = regression_metrics(y_eval, preds)
                            else:
                                metrics = classification_metrics(
                                    y_eval, preds, multiclass=cfg.is_multiclass
                                )
                            scenario_dir.mkdir(parents=True, exist_ok=True)
                            model_path = scenario_dir / "best_model.pkl"
                            if not model_path.exists():
                                if model_name == "apt":
                                    torch.save(
                                        {
                                            "state_dict": cached_model.state_dict(),
                                            "x_train": getattr(cached_model, "x_train", None),
                                            "y_train": getattr(cached_model, "y_train", None),
                                            "x_encoder": getattr(cached_model, "x_encoder", None),
                                            "y_encoder": getattr(cached_model, "y_encoder", None),
                                            "feature_perm": getattr(cached_model, "feature_perm", None),
                                        },
                                        model_path,
                                    )
                                else:
                                    joblib.dump(cached_model, model_path)
                            model_size_mb = model_path.stat().st_size / (1024 * 1024)
                            meta = {
                                "train_time_sec": 0.0,
                                "model_size_mb": float(model_size_mb),
                                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                                "params": params,
                                "scenario": scenario,
                                "reuse_hpo_model": True,
                                "eval_subset_n": None if eval_idx is None else int(len(eval_idx)),
                                "eval_total_n": int(len(X_test_proc)),
                            }
                            with open(scenario_dir / f"meta_{scenario}.json", "w") as f:
                                json.dump(meta, f, indent=2)
                            metrics.update(meta)
                            metrics.update({"scenario": scenario, "model": model_name})
                        else:
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
                            if scenario == "clean" and model_name in {"tabpfn", "apt"}:
                                cached_model = model
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
                if shap is not None or (model_name in {"tabpfn", "apt"} and args.tabpfn_importance == "perm"):
                    if model_name in {"tabpfn", "apt"} and args.tabpfn_importance == "none":
                        shap_top_k.setdefault(scenario, [])
                        if scenario == "clean":
                            shap_clean_top = []
                        else:
                            shap_comparisons[scenario] = []
                        continue
                    sample_cap = 200 if model_name in {"rf", "lgbm", "catboost", "xgboost"} else 50
                    if model_name in {"tabpfn", "apt"}:
                        sample_cap = (
                            args.tabpfn_shap_samples
                            if args.tabpfn_importance == "shap"
                            else args.tabpfn_perm_samples
                        )
                    sample_idx = np.random.default_rng(RANDOM_STATE).choice(
                        X_test_proc.shape[0], size=min(sample_cap, X_test_proc.shape[0]), replace=False
                    )
                    X_sample = X_test_proc[sample_idx]
                    if model_name == "tabpfn":
                        feature_names = tabpfn_feature_names(X_train_s)
                    else:
                        feature_names = _feature_names(preprocessor)
                    if len(feature_names) < X_sample.shape[1]:
                        extra = X_sample.shape[1] - len(feature_names)
                        feature_names = feature_names + [f"pad_{i}" for i in range(extra)]
                    elif len(feature_names) > X_sample.shape[1]:
                        feature_names = feature_names[: X_sample.shape[1]]
                    one_hot_base_cols = None
                    if model_name == "mlp":
                        for name, transformer, cols in preprocessor.transformers_:
                            if name == "cat":
                                one_hot_base_cols = [str(c) for c in cols]
                                break
                    try:
                        if model_name in {"tabpfn", "apt"} and args.tabpfn_importance == "perm":
                            le = None
                            y_eval = y_test
                            if cfg.task == "classification":
                                le = LabelEncoder()
                                y_eval = le.fit_transform(np.asarray(y_test))
                                y_eval = np.asarray(y_eval)[sample_idx]
                            else:
                                y_eval = np.asarray(y_eval)[sample_idx]
                            baseline = accuracy_score(y_eval, model.predict(X_sample))
                            importances = []
                            feature_indices = list(range(X_sample.shape[1]))
                            if args.tabpfn_perm_features and args.tabpfn_perm_features < len(feature_indices):
                                variances = np.var(X_sample, axis=0)
                                feature_indices = list(
                                    np.argsort(variances)[-args.tabpfn_perm_features :]
                                )
                            rng_local = np.random.default_rng(RANDOM_STATE)
                            for col_idx in feature_indices:
                                scores = []
                                for _ in range(args.tabpfn_perm_repeats):
                                    X_perm = X_sample.copy()
                                    rng_local.shuffle(X_perm[:, col_idx])
                                    preds = model.predict(X_perm)
                                    scores.append(accuracy_score(y_eval, preds))
                                importances.append(baseline - float(np.mean(scores)))
                            importance = {
                                feature_names[feature_indices[i]]: float(importances[i])
                                for i in range(len(importances))
                            }
                        else:
                            importance = compute_shap_importance(
                                model,
                                model_name,
                                X_sample,
                                feature_names,
                                one_hot_base_cols=one_hot_base_cols,
                                kernel_nsamples=args.tabpfn_shap_nsamples if model_name in {"tabpfn", "apt"} else None,
                                background_size=args.tabpfn_shap_background if model_name in {"tabpfn", "apt"} else None,
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
