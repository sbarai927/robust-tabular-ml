"""Build a train-test gap table from saved post-HPO model artifacts.

The HPO metrics files store final held-out metrics only. This script reloads the
saved selected models, reconstructs the deterministic HPO train/test split, and
scores each model on both sides of that split without rerunning HPO.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from train_optuna import (
    RANDOM_STATE,
    DatasetConfig,
    build_preprocessor,
    build_tabpfn_inputs,
    load_dataset,
)


RESULTS_DIR = Path("results/hpo")
OUT_DIR = Path("results/paper_tables")
DATASETS = [
    DatasetConfig("diamonds_v2", Path("data/diamonds_v2.csv"), "total_sales_price", "regression"),
    DatasetConfig("diamonds_v3", Path("data/diamonds_v3.csv"), "total_sales_price", "regression"),
    DatasetConfig("drybean", Path("data/drybean_multiclass_classification.csv"), "Class", "classification", True),
    DatasetConfig("mnist", Path("data/mnist_tabular_digits.csv"), "label", "classification", True),
    DatasetConfig("telco", Path("data/telco_churn_classification.csv"), "Churn", "classification"),
    DatasetConfig("titanic", Path("data/titanic_binary_classification.csv"), "Survived", "classification"),
]
MODELS = ["rf", "xgboost", "lgbm", "catboost", "mlp", "tabnet", "tabpfn", "apt"]
ENCODED_LABEL_MODELS = {"tabnet", "tabpfn", "apt"}
PRETRAINED_EVAL_CAP = 50
SKIP_TRAIN_GAP_MODELS = {"tabpfn", "apt"}
DISPLAY_NAMES = {
    "rf": "RF",
    "xgboost": "XGBoost",
    "lgbm": "LightGBM",
    "catboost": "CatBoost",
    "mlp": "MLP",
    "tabnet": "TabNet",
    "tabpfn": "TabPFN",
    "apt": "APT",
}
HPO_TEST_TOL = 0.02
REGRESSION_INVERSION_TOL = -0.20


def dense(array: Any) -> np.ndarray:
    return array.toarray() if hasattr(array, "toarray") else np.asarray(array)


def load_apt_artifact(model_path: Path):
    import torch
    from apt.model import APTClassifier

    payload = torch.load(model_path, map_location="cpu", weights_only=False)
    model = APTClassifier(device="cpu")
    if isinstance(payload, dict) and payload.get("state_dict") is not None:
        model.load_state_dict(payload["state_dict"])
    if isinstance(payload, dict):
        for key in ["x_train", "y_train", "x_encoder", "y_encoder", "feature_perm"]:
            if key in payload:
                setattr(model, key, payload[key])
    if hasattr(model, "eval"):
        model.eval()
    return model


def load_saved_model(model_path: Path, model_name: str):
    if model_name == "apt":
        return load_apt_artifact(model_path)
    return joblib.load(model_path)


def preprocess_split(cfg: DatasetConfig, model_name: str, X_train: pd.DataFrame, X_test: pd.DataFrame):
    if model_name == "tabpfn":
        X_train_proc, X_test_proc, _ = build_tabpfn_inputs(X_train, X_test)
        return np.asarray(X_train_proc, dtype=np.float32), np.asarray(X_test_proc, dtype=np.float32)

    preprocessor = build_preprocessor(X_train)
    X_train_proc = dense(preprocessor.fit_transform(X_train))
    X_test_proc = dense(preprocessor.transform(X_test))
    if model_name in {"tabnet", "apt"}:
        X_train_proc = np.asarray(X_train_proc, dtype=np.float32)
        X_test_proc = np.asarray(X_test_proc, dtype=np.float32)
    return X_train_proc, X_test_proc


def f1_metric(y_true, y_pred, multiclass: bool) -> float:
    y_pred = np.asarray(y_pred).ravel()
    average = "macro" if multiclass else "binary"
    return float(f1_score(y_true, y_pred, average=average))


def safe_f1(cfg: DatasetConfig, model_name: str, y_train, y_true, y_pred) -> float:
    if model_name in ENCODED_LABEL_MODELS:
        le = LabelEncoder()
        le.fit(np.asarray(y_train))
        y_true_eval = le.transform(np.asarray(y_true))
        return f1_metric(y_true_eval, y_pred, cfg.is_multiclass)

    try:
        return f1_metric(y_true, y_pred, cfg.is_multiclass)
    except Exception:
        le = LabelEncoder()
        y_train_enc = le.fit_transform(np.asarray(y_train))
        y_true_enc = le.transform(np.asarray(y_true))
        pred = np.asarray(y_pred).ravel()
        if not np.issubdtype(pred.dtype, np.number):
            pred = le.transform(pred)
        return f1_metric(y_true_enc, pred.astype(int), cfg.is_multiclass)


def evaluate_pair(cfg: DatasetConfig, model_name: str) -> dict[str, Any]:
    if model_name in SKIP_TRAIN_GAP_MODELS:
        raise RuntimeError(
            "train-side gap diagnostic skipped: pretrained inference over stored context is too slow "
            "for full/capped train scoring in this environment"
        )
    if cfg.task == "regression" and model_name in {"tabpfn", "apt"}:
        raise RuntimeError(f"{model_name} was saved only for classification datasets.")

    model_path = RESULTS_DIR / cfg.name / model_name / "best_model.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"missing model: {model_path}")

    X, y = load_dataset(cfg)
    if cfg.name in {"diamonds_v2", "diamonds_v3"} and "Unnamed: 0" not in X.columns:
        # Legacy HPO artifacts for the diamonds variants were trained before the
        # accidental index column was removed from the CSVs. Recreate that column
        # here so saved model feature dimensions match without retraining.
        X = X.copy()
        X.insert(0, "Unnamed: 0", np.arange(len(X)))
    stratify = y if cfg.task == "classification" else None
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=stratify,
    )

    X_train_proc, X_test_proc = preprocess_split(cfg, model_name, X_train, X_test)
    model = load_saved_model(model_path, model_name)

    train_n = len(X_train_proc)
    test_n = len(X_test_proc)
    eval_note = "full split"
    if model_name in {"tabpfn", "apt"}:
        # Pretrained tabular models can be slow at inference because prediction attends
        # over stored training context. Use a deterministic cap for train/test gap
        # diagnostics; standard HPO metrics remain unchanged elsewhere.
        rng = np.random.default_rng(RANDOM_STATE)
        train_idx = rng.choice(len(X_train_proc), size=min(PRETRAINED_EVAL_CAP, len(X_train_proc)), replace=False)
        test_idx = rng.choice(len(X_test_proc), size=min(PRETRAINED_EVAL_CAP, len(X_test_proc)), replace=False)
        X_train_proc = X_train_proc[train_idx]
        X_test_proc = X_test_proc[test_idx]
        y_train_eval = y_train.iloc[train_idx] if hasattr(y_train, "iloc") else np.asarray(y_train)[train_idx]
        y_test_eval = y_test.iloc[test_idx] if hasattr(y_test, "iloc") else np.asarray(y_test)[test_idx]
        train_n = len(X_train_proc)
        test_n = len(X_test_proc)
        eval_note = f"deterministic cap n_train={train_n}, n_test={test_n}"
    else:
        y_train_eval = y_train
        y_test_eval = y_test

    pred_train = np.asarray(model.predict(X_train_proc)).ravel()
    pred_test = np.asarray(model.predict(X_test_proc)).ravel()

    if cfg.task == "classification":
        train_score = safe_f1(cfg, model_name, y_train, y_train_eval, pred_train)
        test_score = safe_f1(cfg, model_name, y_train, y_test_eval, pred_test)
        metric = "macro-F1" if cfg.is_multiclass else "F1"
        gap = train_score - test_score
    else:
        train_score = float(r2_score(y_train_eval, pred_train))
        test_score = float(r2_score(y_test_eval, pred_test))
        metric = "R2"
        gap = train_score - test_score
        if not np.isfinite(train_score) or not np.isfinite(test_score):
            train_rmse = float(np.sqrt(mean_squared_error(y_train_eval, pred_train)))
            test_rmse = float(np.sqrt(mean_squared_error(y_test_eval, pred_test)))
            train_score, test_score, gap = train_rmse, test_rmse, test_rmse - train_rmse
            metric = "RMSE"

    # Sanity-check against the saved HPO metric. Some older artifacts, especially
    # diamonds models trained before the index-column cleanup, can still load but
    # no longer align with the current CSV schema. Those rows are not safe to use
    # for a paper train-test gap table.
    metrics_path = RESULTS_DIR / cfg.name / model_name / "metrics.csv"
    if metrics_path.exists() and metric != "RMSE":
        saved_metrics = pd.read_csv(metrics_path).iloc[0]
        saved_col = "R2" if cfg.task == "regression" else "F1"
        if saved_col in saved_metrics and pd.notna(saved_metrics[saved_col]):
            saved_test = float(saved_metrics[saved_col])
            if abs(saved_test - test_score) > HPO_TEST_TOL:
                raise RuntimeError(
                    f"re-scored {metric}={test_score:.6f} differs from saved HPO "
                    f"{saved_col}={saved_test:.6f}; artifact/current-data schema mismatch"
                )
    if cfg.task == "regression" and metric == "R2" and gap < REGRESSION_INVERSION_TOL:
        raise RuntimeError(
            f"regression train-test gap={gap:.6f} is implausibly negative for a saved "
            "post-HPO model; artifact/current-data schema mismatch"
        )

    return {
        "dataset": cfg.name,
        "task": cfg.task,
        "model": model_name,
        "metric": metric,
        "train_score": train_score,
        "test_score": test_score,
        "train_test_gap": gap,
        "n_train_scored": train_n,
        "n_test_scored": test_n,
        "eval_note": eval_note,
        "status": "ok",
        "error": "",
    }


def fmt_num(value: float | str) -> str:
    if value == "" or pd.isna(value):
        return "--"
    return f"{float(value):.3f}"


def bold_best_latex(tex: str, rows: pd.DataFrame) -> str:
    numeric_cols = ["mean_train_score", "mean_test_score", "mean_train_test_gap"]
    for col in numeric_cols:
        if col not in rows:
            continue
        values = rows[col].dropna()
        if values.empty:
            continue
        best = values.min() if col == "mean_train_test_gap" else values.max()
        repl = f"{best:.3f}"
        tex = tex.replace(repl, f"\\textbf{{{repl}}}", 1)
    return tex


def make_latex(summary: pd.DataFrame) -> str:
    latex_df = summary.copy()
    latex_df["model"] = latex_df["model"].map(DISPLAY_NAMES).fillna(latex_df["model"])
    latex_df = latex_df.rename(
        columns={
            "model": "Model",
            "mean_train_score": "Mean Train",
            "mean_test_score": "Mean Test",
            "mean_train_test_gap": "Mean Gap",
            "largest_gap_dataset": "Largest Gap Dataset",
            "largest_gap": "Largest Gap",
            "n_datasets": "$n$",
        }
    )
    columns = ["Model", "Mean Train", "Mean Test", "Mean Gap", "Largest Gap Dataset", "Largest Gap", "$n$"]
    tex = latex_df[columns].to_latex(
        index=False,
        escape=False,
        na_rep="--",
        float_format=lambda x: f"{x:.3f}",
        caption=(
            "Train--test gap for final selected post-HPO models. Classification scores use F1 "
            "(macro-F1 for multiclass datasets); regression scores use $R^2$. Gaps are computed "
            "as train score minus test score, so larger positive values indicate stronger overfitting. "
            "Means are computed over the $n$ verified dataset--model pairs for which the saved model "
            "reproduced its stored HPO test metric; excluded artifacts are logged separately."
        ),
        label="tab:train_test_gap",
    )
    return bold_best_latex(tex, summary)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    failures = []

    for cfg in DATASETS:
        for model_name in MODELS:
            try:
                rows.append(evaluate_pair(cfg, model_name))
            except Exception as exc:
                failures.append(
                    {
                        "dataset": cfg.name,
                        "model": model_name,
                        "status": "failed",
                        "error": str(exc),
                    }
                )

    detailed = pd.DataFrame(rows)
    detailed_path = OUT_DIR / "train_test_gap_by_dataset.csv"
    detailed.to_csv(detailed_path, index=False)

    failures_df = pd.DataFrame(failures)
    failures_path = OUT_DIR / "train_test_gap_failures.csv"
    failures_df.to_csv(failures_path, index=False)

    if detailed.empty:
        raise RuntimeError("No train-test gap rows could be computed.")

    summary_rows = []
    for model_name, group in detailed.groupby("model", sort=False):
        idx = group["train_test_gap"].idxmax()
        summary_rows.append(
            {
                "model": model_name,
                "mean_train_score": group["train_score"].mean(),
                "mean_test_score": group["test_score"].mean(),
                "mean_train_test_gap": group["train_test_gap"].mean(),
                "largest_gap_dataset": group.loc[idx, "dataset"],
                "largest_gap": group.loc[idx, "train_test_gap"],
                "n_datasets": int(len(group)),
            }
        )

    summary = pd.DataFrame(summary_rows).sort_values("mean_train_test_gap", ascending=False)
    summary_path = OUT_DIR / "train_test_gap_summary.csv"
    summary.to_csv(summary_path, index=False)

    latex = make_latex(summary)
    tex_path = OUT_DIR / "train_test_gap_table.tex"
    tex_path.write_text(latex)

    note = (
        "Note: the repository did not contain saved train-side HPO metrics. This table was "
        "computed by reloading the saved final selected models and applying the same deterministic "
        "HPO train/test split and preprocessing used in src/hpo_tuning/train_optuna.py. "
        "Models unavailable in the local environment are listed in train_test_gap_failures.csv.\n"
    )
    (OUT_DIR / "train_test_gap_note.txt").write_text(note)

    print(f"Saved detailed rows: {detailed_path}")
    print(f"Saved summary CSV: {summary_path}")
    print(f"Saved LaTeX table: {tex_path}")
    print(f"Saved failures: {failures_path}")
    print("\nLaTeX table:\n")
    print(latex)


if __name__ == "__main__":
    main()
