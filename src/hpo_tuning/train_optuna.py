"""Optuna-based HPO for multiple models across tabular datasets."""

import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, Any, Callable, Optional

# Import torch before sklearn to avoid OpenMP SHM init crashes on some setups.
try:
    import torch as _torch
    torch = _torch
    nn = _torch.nn
    from torch.utils.data import DataLoader, TensorDataset
except Exception:
    torch = None
    nn = None
    DataLoader = TensorDataset = None

import joblib
import numpy as np
import optuna
import pandas as pd
import warnings
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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.exceptions import ConvergenceWarning


def _has_torch() -> bool:
    """Check (and lazily import) torch for transformer models."""
    global torch, nn, DataLoader, TensorDataset
    if torch is not None:
        return True
    try:
        import importlib

        torch = importlib.import_module("torch")
        nn = importlib.import_module("torch.nn")
        utils_data = importlib.import_module("torch.utils.data")
        DataLoader = getattr(utils_data, "DataLoader", None)
        TensorDataset = getattr(utils_data, "TensorDataset", None)
        return True
    except Exception:
        return False

# Optional models
try:
    from lightgbm import LGBMClassifier, LGBMRegressor
    import lightgbm as lgb
except Exception:
    LGBMClassifier = LGBMRegressor = None
    lgb = None

try:
    from catboost import CatBoostClassifier, CatBoostRegressor
except Exception:
    CatBoostClassifier = CatBoostRegressor = None

try:
    import xgboost as xgb
except Exception:
    xgb = None

try:
    from pytorch_tabnet.tab_model import TabNetClassifier, TabNetRegressor
except Exception:
    TabNetClassifier = TabNetRegressor = None

RANDOM_STATE = 42
DEFAULT_TRIALS = int(os.getenv("HPO_TRIALS", "30"))
DEFAULT_TIMEOUT = int(os.getenv("HPO_TIMEOUT", "0"))  # seconds per study; 0 disables
DATASET_FILTER = set(os.getenv("HPO_DATASETS", "").split(",")) if os.getenv("HPO_DATASETS") else None
MODEL_FILTER = set(os.getenv("HPO_MODELS", "").split(",")) if os.getenv("HPO_MODELS") else None
OVERWRITE = os.getenv("HPO_OVERWRITE", "false").lower() == "true"

# FTTransformer tuning guards for memory-constrained environments.
FTT_MAX_DMODEL = int(os.getenv("FTT_MAX_DMODEL", "256"))
FTT_MAX_LAYERS = int(os.getenv("FTT_MAX_LAYERS", "6"))
FTT_BATCH_CHOICES = [
    int(x) for x in os.getenv("FTT_BATCH_CHOICES", "256,512,1024").split(",") if x.strip()
]

# Ensure Matplotlib has a writable cache to avoid warnings and slowdowns
os.environ.setdefault("MPLBACKEND", "Agg")
mpl_dir = Path("results/.matplotlib").absolute()
mpl_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(mpl_dir))


@dataclass
class DatasetConfig:
    name: str
    path: Path
    target: str
    task: str  # "regression" or "classification"
    is_multiclass: bool = False


def load_dataset(cfg: DatasetConfig) -> Tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(cfg.path)
    df = df.dropna().reset_index(drop=True)
    if cfg.task == "classification" and not np.issubdtype(df[cfg.target].dtype, np.number):
        df[cfg.target] = df[cfg.target].astype(str)
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


def regression_metrics(y_true, y_pred) -> Dict[str, float]:
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
    }


def classification_metrics(y_true, y_pred, multiclass=False) -> Dict[str, float]:
    average = "macro" if multiclass else "binary"
    if not multiclass:
        le = LabelEncoder()
        y_true_enc = le.fit_transform(np.asarray(y_true))
        y_pred_enc = le.transform(np.asarray(y_pred))
        acc = accuracy_score(y_true_enc, y_pred_enc)
        f1 = f1_score(y_true_enc, y_pred_enc, average=average, pos_label=1)
    else:
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average=average)
    return {"Accuracy": float(acc), "F1": float(f1)}


def split_tabnet_params(params: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    params = params.copy()
    batch_size = params.pop("batch_size", 1024)
    virtual_batch_size = params.pop("virtual_batch_size", 128)
    if virtual_batch_size > batch_size:
        virtual_batch_size = batch_size
    fit_params = {
        "batch_size": batch_size,
        "virtual_batch_size": virtual_batch_size,
    }
    lr = params.pop("learning_rate", 1e-3)
    params["optimizer_params"] = {"lr": lr}
    return params, fit_params


def tabnet_regression_target(y: pd.Series) -> np.ndarray:
    return np.asarray(y).reshape(-1, 1)


class TorchTransformerModel:
    """Minimal Transformer-style tabular model supporting FTTransformer/SAINT variants."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        task: str,
        is_multiclass: bool,
        architecture: str,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 3,
        dropout: float = 0.1,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        batch_size: int = 512,
        max_epochs: int = 50,
        patience: int = 8,
    ):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.task = task
        self.is_multiclass = is_multiclass
        self.architecture = architecture
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self._ensure_torch()
        self.device = torch.device("cuda" if torch and torch.cuda.is_available() else "cpu")
        self.model: Optional[Any] = None

    @staticmethod
    def _ensure_torch():
        global torch, nn, DataLoader, TensorDataset
        if torch is None:
            import importlib
            try:
                torch = importlib.import_module("torch")
                nn = importlib.import_module("torch.nn")
                DataLoader = importlib.import_module("torch.utils.data").DataLoader
                TensorDataset = importlib.import_module("torch.utils.data").TensorDataset
            except Exception as exc:
                raise RuntimeError(f"torch not available: {exc}")

    def _build_net(self) -> Any:
        # Embed each feature independently then add learnable column embeddings
        embed = nn.Linear(1, self.d_model)
        col_embed = nn.Parameter(torch.randn(self.input_dim, self.d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.n_heads,
            dim_feedforward=max(4 * self.d_model, 256),
            dropout=self.dropout,
            batch_first=True,
        )
        encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.n_layers)
        proj = nn.Linear(self.d_model, self.output_dim)
        dropout = nn.Dropout(self.dropout)

        class Net(nn.Module):
            def __init__(self, task: str, is_multiclass: bool):
                super().__init__()
                self.task = task
                self.is_multiclass = is_multiclass
                self.embed = embed
                self.col_embed = col_embed
                self.encoder = encoder
                self.dropout = dropout
                self.proj = proj

            def forward(self, x):
                # x: [B, F]
                tokens = self.embed(x.unsqueeze(-1)) + self.col_embed  # [B, F, d_model]
                tokens = self.encoder(tokens)
                pooled = tokens.mean(dim=1)
                pooled = self.dropout(pooled)
                logits = self.proj(pooled)
                return logits

        return Net(self.task, self.is_multiclass)

    def fit(
        self,
        X_train: np.ndarray,
        y_train: pd.Series,
        X_valid: Optional[np.ndarray] = None,
        y_valid: Optional[pd.Series] = None,
    ):
        self.model = self._build_net().to(self.device)
        # If no explicit validation set is provided, create a small split
        if X_valid is None or y_valid is None:
            X_train, X_valid, y_train, y_valid = train_test_split(
                X_train, y_train, test_size=0.2, random_state=RANDOM_STATE, stratify=None
            )

        X_tr = torch.tensor(X_train, dtype=torch.float32)
        X_va = torch.tensor(X_valid, dtype=torch.float32)
        if self.task == "regression":
            y_tr = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(-1)
            y_va = torch.tensor(y_valid.values, dtype=torch.float32).unsqueeze(-1)
            criterion = nn.MSELoss()
        else:
            # Ensure labels start at 0
            le = LabelEncoder()
            y_tr_np = le.fit_transform(np.asarray(y_train))
            y_va_np = le.transform(np.asarray(y_valid))
            self.classes_ = le.classes_
            y_tr = torch.tensor(y_tr_np, dtype=torch.long)
            y_va = torch.tensor(y_va_np, dtype=torch.long)
            criterion = nn.CrossEntropyLoss()

        train_ds = TensorDataset(X_tr, y_tr)
        valid_ds = TensorDataset(X_va, y_va)
        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
        valid_loader = DataLoader(valid_ds, batch_size=self.batch_size, shuffle=False)

        optim = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        best_loss = float("inf")
        patience_left = self.patience
        best_state = None

        for epoch in range(self.max_epochs):
            self.model.train()
            for xb, yb in train_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optim.zero_grad()
                logits = self.model(xb)
                if self.task == "regression":
                    loss = criterion(logits, yb)
                else:
                    loss = criterion(logits, yb)
                loss.backward()
                optim.step()

            # Validation
            self.model.eval()
            val_losses = []
            with torch.no_grad():
                for xb, yb in valid_loader:
                    xb, yb = xb.to(self.device), yb.to(self.device)
                    logits = self.model(xb)
                    if self.task == "regression":
                        loss = criterion(logits, yb)
                    else:
                        loss = criterion(logits, yb)
                    val_losses.append(loss.item())
            val_loss = float(np.mean(val_losses)) if val_losses else float("inf")
            if val_loss + 1e-6 < best_loss:
                best_loss = val_loss
                best_state = {k: v.cpu() for k, v in self.model.state_dict().items()}
                patience_left = self.patience
            else:
                patience_left -= 1
                if patience_left <= 0:
                    break

        if best_state:
            self.model.load_state_dict(best_state)

    def predict(self, X: np.ndarray) -> np.ndarray:
        assert self.model is not None
        self.model.eval()
        X_t = torch.tensor(X, dtype=torch.float32)
        loader = DataLoader(TensorDataset(X_t), batch_size=self.batch_size, shuffle=False)
        preds = []
        with torch.no_grad():
            for (xb,) in loader:
                xb = xb.to(self.device)
                logits = self.model(xb)
                if self.task == "regression":
                    preds.append(logits.cpu().numpy())
                else:
                    probs = torch.softmax(logits, dim=1)
                    labels = torch.argmax(probs, dim=1)
                    preds.append(labels.cpu().numpy())
        pred_arr = np.concatenate(preds, axis=0)
        if self.task == "regression":
            return pred_arr.ravel()
        return pred_arr


def objective_factory(
    model_name: str,
    cfg: DatasetConfig,
    X_train: Any,
    X_valid: Any,
    y_train: pd.Series,
    y_valid: pd.Series,
) -> Tuple[
    Callable[[optuna.trial.Trial], float],
    Callable[[Dict[str, Any]], Any],
    Callable[[optuna.trial.Trial], Dict[str, Any]],
]:
    is_reg = cfg.task == "regression"

    def suggest_params(trial: optuna.trial.Trial) -> Dict[str, Any]:
        if model_name == "rf":
            return {
                "n_estimators": trial.suggest_int("n_estimators", 120, 400),
                "max_depth": trial.suggest_int("max_depth", 4, 18),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 4),
            }
        if model_name == "lgbm":
            return {
                "n_estimators": trial.suggest_int("n_estimators", 150, 400),
                "num_leaves": trial.suggest_int("num_leaves", 15, 255),
                "max_depth": trial.suggest_int("max_depth", -1, 16),
                "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.2, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            }
        if model_name == "catboost":
            return {
                "iterations": trial.suggest_int("iterations", 200, 500),
                "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.2, log=True),
                "depth": trial.suggest_int("depth", 4, 10),
                "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-2, 10.0, log=True),
            }
        if model_name == "xgboost":
            return {
                "n_estimators": trial.suggest_int("n_estimators", 150, 400),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.2, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            }
        if model_name == "mlp":
            h1 = trial.suggest_int("h1", 64, 256)
            h2 = trial.suggest_int("h2", 32, 128)
            return {
                "h1": h1,
                "h2": h2,
                "alpha": trial.suggest_float("alpha", 1e-5, 1e-2, log=True),
                "learning_rate_init": trial.suggest_float(
                    "learning_rate_init", 1e-4, 1e-2, log=True
                ),
            }
        if model_name in {"fttransformer", "saint"}:
            d_model_high = min(FTT_MAX_DMODEL, 256)
            d_model_low = min(64, d_model_high)
            d_model = trial.suggest_int("d_model", d_model_low, d_model_high, step=32)
            n_heads = trial.suggest_int("n_heads", 2, 8)
            if d_model % n_heads != 0:
                d_model = (d_model // n_heads) * n_heads
            return {
                "d_model": d_model,
                "n_heads": n_heads,
                "n_layers": trial.suggest_int("n_layers", 2, min(FTT_MAX_LAYERS, 6)),
                "dropout": trial.suggest_float("dropout", 0.05, 0.4),
                "lr": trial.suggest_float("lr", 1e-4, 5e-3, log=True),
                "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
                "batch_size": trial.suggest_categorical(
                    "batch_size", FTT_BATCH_CHOICES or [256, 512, 1024]
                ),
                "max_epochs": trial.suggest_int("max_epochs", 20, 60),
                "patience": trial.suggest_int("patience", 5, 12),
            }
        if model_name == "tabnet":
            batch_size = trial.suggest_categorical("batch_size", [256, 512, 1024])
            virtual_batch_size = trial.suggest_categorical(
                "virtual_batch_size", [128, 256, 512]
            )
            return {
                "n_d": trial.suggest_int("n_d", 8, 64, step=8),
                "n_a": trial.suggest_int("n_a", 8, 64, step=8),
                "n_steps": trial.suggest_int("n_steps", 3, 8),
                "gamma": trial.suggest_float("gamma", 1.0, 2.0),
                "lambda_sparse": trial.suggest_float("lambda_sparse", 1e-6, 1e-2, log=True),
                "momentum": trial.suggest_float("momentum", 0.01, 0.4),
                "learning_rate": trial.suggest_float("learning_rate", 1e-3, 5e-2, log=True),
                "batch_size": batch_size,
                "virtual_batch_size": virtual_batch_size,
            }
        raise optuna.TrialPruned(f"Unsupported model {model_name}")

    def build_model(params: Dict[str, Any]):
        if model_name == "rf":
            base = RandomForestRegressor if is_reg else RandomForestClassifier
            return base(random_state=RANDOM_STATE, n_jobs=-1, **params)
        if model_name == "lgbm":
            if LGBMClassifier is None:
                raise optuna.TrialPruned("lightgbm not installed")
            if is_reg:
                return LGBMRegressor(random_state=RANDOM_STATE, **params)
            return LGBMClassifier(random_state=RANDOM_STATE, **params)
        if model_name == "catboost":
            if CatBoostClassifier is None:
                raise optuna.TrialPruned("catboost not installed")
            if is_reg:
                return CatBoostRegressor(random_seed=RANDOM_STATE, verbose=False, **params)
            return CatBoostClassifier(random_seed=RANDOM_STATE, verbose=False, **params)
        if model_name == "xgboost":
            if xgb is None:
                raise optuna.TrialPruned("xgboost not installed")
            if is_reg:
                return xgb.XGBRegressor(
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                    eval_metric="rmse",
                    **params,
                )
            metric = "mlogloss" if cfg.is_multiclass else "logloss"
            return xgb.XGBClassifier(
                random_state=RANDOM_STATE,
                n_jobs=-1,
                eval_metric=metric,
                **params,
            )
        if model_name == "mlp":
            # Convert helper h1/h2 to hidden_layer_sizes
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
        if model_name in {"fttransformer", "saint"}:
            if torch is None:
                raise optuna.TrialPruned("torch not installed")
            input_dim = X_train.shape[1]
            if cfg.task == "classification":
                classes = np.unique(y_train)
                output_dim = len(classes)
            else:
                output_dim = 1
            return TorchTransformerModel(
                input_dim=input_dim,
                output_dim=output_dim,
                task=cfg.task,
                is_multiclass=cfg.is_multiclass,
                architecture=model_name,
                **params,
            )
        if model_name == "tabnet":
            if TabNetClassifier is None:
                raise optuna.TrialPruned("pytorch-tabnet not installed")
            device_name = "cuda" if _has_torch() and torch.cuda.is_available() else "cpu"
            params = params.copy()
            params.update({"seed": RANDOM_STATE, "device_name": device_name})
            params.setdefault("mask_type", "entmax")
            if is_reg:
                return TabNetRegressor(**params)
            return TabNetClassifier(**params)
        raise optuna.TrialPruned(f"Unsupported model {model_name}")

    def objective(trial: optuna.trial.Trial) -> float:
        params = suggest_params(trial)
        model = None
        if model_name != "tabnet":
            model = build_model(params)
        fit_kwargs = {}
        # Early stopping for tree/boosted models
        if model_name == "lgbm":
            callbacks = []
            if lgb is not None:
                callbacks = [lgb.early_stopping(50, verbose=False)]
            fit_kwargs = {
                "eval_set": [(X_valid, y_valid)],
                "eval_metric": "rmse" if is_reg else "logloss",
                "callbacks": callbacks,
            }
        elif model_name == "xgboost":
            fit_kwargs = {}
        elif model_name == "catboost":
            fit_kwargs = {"eval_set": (X_valid, y_valid), "use_best_model": True}
        elif model_name in {"fttransformer", "saint"}:
            fit_kwargs = {"X_valid": X_valid, "y_valid": y_valid}
        elif model_name == "tabnet":
            fit_kwargs = {"eval_set": [(X_valid, y_valid)], "patience": 20}

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            if model_name == "tabnet" and cfg.task == "classification":
                model_params, fit_params = split_tabnet_params(params)
                model = build_model(model_params)
                X_train_arr = np.asarray(X_train, dtype=np.float32)
                X_valid_arr = np.asarray(X_valid, dtype=np.float32)
                le = LabelEncoder()
                y_train_enc = le.fit_transform(np.asarray(y_train))
                y_valid_enc = le.transform(np.asarray(y_valid))
                try:
                    model.fit(
                        X_train_arr,
                        y_train_enc,
                        eval_set=[(X_valid_arr, y_valid_enc)],
                        patience=20,
                        **fit_params,
                    )
                except RuntimeError as exc:
                    raise optuna.TrialPruned(str(exc)) from exc
                preds = model.predict(X_valid_arr)
                f1 = f1_score(
                    y_valid_enc,
                    preds,
                    average="macro" if cfg.is_multiclass else "binary",
                )
                return float(f1)
            if model_name == "tabnet":
                model_params, fit_params = split_tabnet_params(params)
                model = build_model(model_params)
                y_train_arr = tabnet_regression_target(y_train)
                y_valid_arr = tabnet_regression_target(y_valid)
                X_train_arr = np.asarray(X_train, dtype=np.float32)
                X_valid_arr = np.asarray(X_valid, dtype=np.float32)
                try:
                    model.fit(
                        X_train_arr,
                        y_train_arr,
                        eval_set=[(X_valid_arr, y_valid_arr)],
                        patience=20,
                        **fit_params,
                    )
                except RuntimeError as exc:
                    raise optuna.TrialPruned(str(exc)) from exc
                preds = model.predict(X_valid_arr)
                metrics = regression_metrics(y_valid, preds) if is_reg else classification_metrics(
                    y_valid, preds, multiclass=cfg.is_multiclass
                )
                return metrics["RMSE"] if is_reg else metrics["F1"]
            model.fit(X_train, y_train, **fit_kwargs)
        preds = model.predict(X_valid)
        if is_reg:
            metrics = regression_metrics(y_valid, preds)
            return metrics["RMSE"]
        else:
            metrics = classification_metrics(
                y_valid, preds, multiclass=cfg.is_multiclass
            )
            # Maximize F1 for comparability
            return metrics["F1"]

    return objective, build_model, suggest_params


def train_model(
    cfg: DatasetConfig,
    model_name: str,
    X_train: pd.DataFrame,
    X_valid: pd.DataFrame,
    y_train: pd.Series,
    y_valid: pd.Series,
    preprocessor: ColumnTransformer,
    out_dir: Path,
    n_trials: int = DEFAULT_TRIALS,
    timeout_seconds: int = DEFAULT_TIMEOUT,
    overwrite: bool = False,
) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    preprocessor = build_preprocessor(X_train)
    X_train_proc = preprocessor.fit_transform(X_train)
    X_valid_proc = preprocessor.transform(X_valid)
    feature_names = getattr(preprocessor, "get_feature_names_out", lambda: None)()
    if feature_names is None:
        feature_names = [f"f{i}" for i in range(X_train_proc.shape[1])]
    else:
        feature_names = list(feature_names)
    # Ensure dense arrays for estimators that dislike sparse
    if hasattr(X_train_proc, "toarray"):
        X_train_proc = X_train_proc.toarray()
        X_valid_proc = X_valid_proc.toarray()

    sampler = optuna.samplers.TPESampler(seed=RANDOM_STATE)
    objective, build_model, suggest_params = objective_factory(
        model_name, cfg, X_train_proc, X_valid_proc, y_train, y_valid
    )
    start = time.time()
    direction = "minimize" if cfg.task == "regression" else "maximize"
    study_name = f"{cfg.name}_{model_name}"
    study_db = out_dir / "study.db"
    if study_db.exists() and overwrite:
        study_db.unlink()
    storage = f"sqlite:///{study_db}"
    study = optuna.create_study(
        direction=direction,
        sampler=sampler,
        study_name=study_name,
        storage=storage,
        load_if_exists=not overwrite,
    )
    timeout = None if timeout_seconds <= 0 else timeout_seconds
    completed_trials = len(study.trials)
    remaining_trials = max(0, n_trials - completed_trials)
    if remaining_trials > 0:
        study.optimize(objective, n_trials=remaining_trials, timeout=timeout)
    else:
        print(
            f"⏩ Study already has {completed_trials} trials; skipping optimization."
        )
    elapsed = time.time() - start
    # Save study
    joblib.dump(study, out_dir / "study.pkl")
    pd.DataFrame([t.params for t in study.trials]).to_csv(out_dir / "trials.csv", index=False)
    # Refit best model
    best_params = study.best_trial.params
    fit_params = {}
    if model_name == "tabnet":
        model_params, fit_params = split_tabnet_params(best_params)
        model = build_model(model_params)
    else:
        model = build_model(best_params)
    fit_kwargs = {}
    if model_name == "lgbm":
        callbacks = []
        if lgb is not None:
            callbacks = [lgb.early_stopping(50, verbose=False)]
        fit_kwargs = {
            "eval_set": [(X_valid_proc, y_valid)],
            "eval_metric": "rmse" if cfg.task == "regression" else "logloss",
            "callbacks": callbacks,
        }
    elif model_name == "xgboost":
        fit_kwargs = {}
    elif model_name == "catboost":
        fit_kwargs = {"eval_set": (X_valid_proc, y_valid), "use_best_model": True}
    elif model_name in {"fttransformer", "saint"}:
        fit_kwargs = {"X_valid": X_valid_proc, "y_valid": y_valid}
    elif model_name == "tabnet":
        fit_kwargs = {"eval_set": [(X_valid_proc, y_valid)], "patience": 20}

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        X_full = np.vstack([X_train_proc, X_valid_proc])
        y_full = pd.concat([y_train, y_valid])
        if model_name == "tabnet" and cfg.task == "classification":
            le = LabelEncoder()
            y_full_enc = le.fit_transform(np.asarray(y_full))
            y_valid_enc = le.transform(np.asarray(y_valid))
            X_full_arr = np.asarray(X_full, dtype=np.float32)
            X_valid_arr = np.asarray(X_valid_proc, dtype=np.float32)
            model.fit(
                X_full_arr,
                y_full_enc,
                eval_set=[(X_valid_arr, y_valid_enc)],
                patience=20,
                **fit_params,
            )
            preds = model.predict(X_valid_arr)
            metrics = {
                "Accuracy": float(accuracy_score(y_valid_enc, preds)),
                "F1": float(
                    f1_score(y_valid_enc, preds, average="macro" if cfg.is_multiclass else "binary")
                ),
            }
        else:
            if model_name == "tabnet":
                y_full_arr = tabnet_regression_target(y_full)
                y_valid_arr = tabnet_regression_target(y_valid)
                X_full_arr = np.asarray(X_full, dtype=np.float32)
                X_valid_arr = np.asarray(X_valid_proc, dtype=np.float32)
                model.fit(
                    X_full_arr,
                    y_full_arr,
                    eval_set=[(X_valid_arr, y_valid_arr)],
                    patience=20,
                    **fit_params,
                )
                preds = model.predict(X_valid_arr)
                metrics = regression_metrics(y_valid, preds)
            else:
                model.fit(X_full, y_full, **fit_kwargs, **fit_params)
                preds = model.predict(X_valid_proc)
                if cfg.task == "regression":
                    metrics = regression_metrics(y_valid, preds)
                else:
                    metrics = classification_metrics(y_valid, preds, multiclass=cfg.is_multiclass)
    metrics["best_value"] = study.best_value
    metrics["elapsed_sec"] = elapsed
    # Save model
    model_path = out_dir / "best_model.pkl"
    if isinstance(model, TorchTransformerModel):
        torch.save({"state_dict": model.model.state_dict(), "config": model.__dict__}, model_path)
    else:
        joblib.dump(model, model_path)
    model_size_mb = model_path.stat().st_size / (1024 * 1024)
    metrics["model_size_mb"] = model_size_mb
    # Save params and metrics
    with open(out_dir / "best_params.json", "w") as f:
        json.dump(best_params, f, indent=2)
    pd.DataFrame([metrics]).to_csv(out_dir / "metrics.csv", index=False)
    print(f"✔ Saved results for {cfg.name} / {model_name} to {out_dir}")
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Optuna HPO trainer")
    parser.add_argument("--dataset", nargs="*", default=None)
    parser.add_argument("--model", nargs="*", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS)
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    args = parser.parse_args()

    dataset_filter = set(args.dataset) if args.dataset else DATASET_FILTER
    model_filter = set(args.model) if args.model else MODEL_FILTER
    overwrite = args.overwrite or OVERWRITE
    n_trials = args.trials
    timeout_override = args.timeout

    datasets = [
        DatasetConfig("diamonds_v2", Path("data/diamonds_v2.csv"), "total_sales_price", "regression"),
        DatasetConfig("titanic", Path("data/titanic_binary_classification.csv"), "Survived", "classification"),
        DatasetConfig("drybean", Path("data/drybean_multiclass_classification.csv"), "Class", "classification", True),
        DatasetConfig("telco", Path("data/telco_churn_classification.csv"), "Churn", "classification"),
        DatasetConfig("diamonds_v3", Path("data/diamonds_v3.csv"), "total_sales_price", "regression"),
        DatasetConfig("mnist", Path("data/mnist_tabular_digits.csv"), "label", "classification", True),
    ]
    models = ["rf", "lgbm", "catboost", "xgboost", "mlp", "tabnet", "fttransformer"]
    results_root = Path("results/hpo")
    for cfg in datasets:
        if dataset_filter and cfg.name not in dataset_filter:
            continue
        X, y = load_dataset(cfg)
        X_train, X_valid, y_train, y_valid = train_test_split(
            X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y if cfg.task == "classification" else None
        )
        for model_name in models:
            if model_filter and model_name not in model_filter:
                continue
            deps_missing = (
                (model_name == "lgbm" and LGBMClassifier is None) or
                (model_name == "catboost" and CatBoostClassifier is None) or
                (model_name == "xgboost" and xgb is None) or
                (model_name == "tabnet" and TabNetClassifier is None) or
                (model_name in {"fttransformer", "saint"} and not _has_torch())
            )
            if deps_missing:
                print(f"⚠️ Skipping {model_name} for {cfg.name}: dependency not installed.")
                continue
            out_dir = results_root / cfg.name / model_name
            metrics_path = out_dir / "metrics.csv"
            if metrics_path.exists() and not overwrite:
                study_db = out_dir / "study.db"
                if study_db.exists():
                    try:
                        storage = f"sqlite:///{study_db}"
                        study = optuna.load_study(
                            study_name=f"{cfg.name}_{model_name}",
                            storage=storage,
                        )
                        if len(study.trials) < n_trials:
                            print(
                                f"↩️ Resuming {cfg.name}/{model_name} from {len(study.trials)} to {n_trials} trials."
                            )
                        else:
                            print(
                                f"⏩ Skipping {cfg.name}/{model_name} (metrics.csv exists and trials >= {n_trials})."
                            )
                            continue
                    except Exception:
                        print(
                            f"⏩ Skipping {cfg.name}/{model_name} (metrics.csv exists; set HPO_OVERWRITE=true to rerun)."
                        )
                        continue
                else:
                    print(
                        f"⏩ Skipping {cfg.name}/{model_name} (metrics.csv exists; set HPO_OVERWRITE=true to rerun)."
                    )
                    continue
            train_model(
                cfg,
                model_name,
                X_train,
                X_valid,
                y_train,
                y_valid,
                None,
                out_dir,
                n_trials=n_trials,
                timeout_seconds=timeout_override,
                overwrite=overwrite,
            )


if __name__ == "__main__":
    main()
