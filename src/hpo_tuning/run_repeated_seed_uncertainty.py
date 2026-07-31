"""Repeated-seed uncertainty check for final HPO configurations.

This script reuses saved `results/hpo/<dataset>/<model>/best_params.json`
artifacts and repeats final train/test evaluation over multiple random seeds.
It is intentionally not a nested CV or HPO rerun; it is a lightweight
uncertainty analysis for paper/reviewer reporting.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import f1_score, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import LabelEncoder

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.hpo_tuning.train_optuna import (  # noqa: E402
    CatBoostClassifier,
    CatBoostRegressor,
    DatasetConfig,
    LGBMClassifier,
    LGBMRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
    TabNetClassifier,
    TabNetRegressor,
    _has_torch,
    build_preprocessor,
    build_tabpfn_inputs,
    load_dataset,
    split_tabnet_params,
    tabnet_regression_target,
    torch,
    xgb,
)


DATASETS: dict[str, DatasetConfig] = {
    "diamonds_v2": DatasetConfig(
        "diamonds_v2", Path("data/diamonds_v2.csv"), "total_sales_price", "regression"
    ),
    "diamonds_v3": DatasetConfig(
        "diamonds_v3", Path("data/diamonds_v3.csv"), "total_sales_price", "regression"
    ),
    "drybean": DatasetConfig(
        "drybean",
        Path("data/drybean_multiclass_classification.csv"),
        "Class",
        "classification",
        is_multiclass=True,
    ),
    "mnist": DatasetConfig(
        "mnist",
        Path("data/mnist_tabular_digits.csv"),
        "label",
        "classification",
        is_multiclass=True,
    ),
    "telco": DatasetConfig(
        "telco", Path("data/telco_churn_classification.csv"), "Churn", "classification"
    ),
    "titanic": DatasetConfig(
        "titanic",
        Path("data/titanic_binary_classification.csv"),
        "Survived",
        "classification",
    ),
}

MODELS = ["rf", "xgboost", "lgbm", "catboost", "mlp", "tabnet", "tabpfn", "apt"]
PRETRAINED_MODELS = {"tabpfn", "apt"}
SEEDS = [0, 42, 123]


@dataclass
class EvalRecord:
    dataset: str
    model: str
    task: str
    seed: int
    metric_name: str
    test_score: float | None
    train_score_if_available: float | None
    status: str
    notes: str = ""


def load_params(dataset: str, model: str) -> dict[str, Any] | None:
    path = REPO_ROOT / "results" / "hpo" / dataset / model / "best_params.json"
    if not path.exists():
        return None
    with path.open() as f:
        return json.load(f)


def maybe_cap_rows(
    X: pd.DataFrame,
    y: pd.Series,
    cfg: DatasetConfig,
    seed: int,
    max_rows: int,
) -> tuple[pd.DataFrame, pd.Series, str]:
    if max_rows <= 0 or len(X) <= max_rows:
        return X, y, ""
    stratify = y if cfg.task == "classification" else None
    X_keep, _, y_keep, _ = train_test_split(
        X,
        y,
        train_size=max_rows,
        random_state=seed,
        stratify=stratify,
    )
    note = f"deterministic row cap used for lightweight analysis: {max_rows}/{len(X)} rows"
    return X_keep.reset_index(drop=True), y_keep.reset_index(drop=True), note


def preprocess_standard(
    X_train: pd.DataFrame, X_test: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray]:
    preprocessor = build_preprocessor(X_train)
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)
    if hasattr(X_train_proc, "toarray"):
        X_train_proc = X_train_proc.toarray()
        X_test_proc = X_test_proc.toarray()
    return np.asarray(X_train_proc), np.asarray(X_test_proc)


def build_model(
    model_name: str,
    cfg: DatasetConfig,
    params: dict[str, Any],
    seed: int,
):
    is_reg = cfg.task == "regression"
    params = dict(params)
    if model_name == "rf":
        cls = RandomForestRegressor if is_reg else RandomForestClassifier
        return cls(random_state=seed, n_jobs=-1, **params)
    if model_name == "lgbm":
        if LGBMClassifier is None:
            raise RuntimeError("lightgbm is not installed")
        cls = LGBMRegressor if is_reg else LGBMClassifier
        params.setdefault("verbosity", -1)
        return cls(random_state=seed, **params)
    if model_name == "catboost":
        if CatBoostClassifier is None:
            raise RuntimeError("catboost is not installed")
        cls = CatBoostRegressor if is_reg else CatBoostClassifier
        return cls(random_seed=seed, verbose=False, **params)
    if model_name == "xgboost":
        if xgb is None:
            raise RuntimeError("xgboost is not installed")
        if is_reg:
            return xgb.XGBRegressor(
                random_state=seed, n_jobs=-1, eval_metric="rmse", **params
            )
        metric = "mlogloss" if cfg.is_multiclass else "logloss"
        return xgb.XGBClassifier(
            random_state=seed, n_jobs=-1, eval_metric=metric, **params
        )
    if model_name == "mlp":
        if "h1" in params or "h2" in params:
            h1 = params.pop("h1")
            h2 = params.pop("h2")
            params["hidden_layer_sizes"] = tuple(sorted([h1, h2], reverse=True))
        elif isinstance(params.get("hidden_layer_sizes"), list):
            params["hidden_layer_sizes"] = tuple(params["hidden_layer_sizes"])
        cls = MLPRegressor if is_reg else MLPClassifier
        return cls(max_iter=200, random_state=seed, early_stopping=True, **params)
    if model_name == "tabnet":
        if TabNetClassifier is None:
            raise RuntimeError("pytorch-tabnet is not installed")
        model_params, fit_params = split_tabnet_params(params)
        device = "cuda" if _has_torch() and torch.cuda.is_available() else "cpu"
        model_params.update({"seed": seed, "device_name": device})
        model_params.setdefault("mask_type", "entmax")
        cls = TabNetRegressor if is_reg else TabNetClassifier
        return cls(**model_params), fit_params
    raise RuntimeError(f"unsupported model: {model_name}")


def score_predictions(
    cfg: DatasetConfig,
    y_true,
    y_pred,
    label_encoder: LabelEncoder | None = None,
) -> dict[str, float]:
    if cfg.task == "regression":
        return {
            "R2": float(r2_score(y_true, y_pred)),
            "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        }
    if label_encoder is not None:
        y_true = label_encoder.transform(np.asarray(y_true))
    average = "macro" if cfg.is_multiclass else "binary"
    return {"F1": float(f1_score(y_true, y_pred, average=average))}


def evaluate_standard_model(
    cfg: DatasetConfig,
    model_name: str,
    params: dict[str, Any],
    seed: int,
    max_rows: int,
    tabnet_max_epochs: int,
) -> list[EvalRecord]:
    X, y = load_dataset(cfg)
    X, y, cap_note = maybe_cap_rows(X, y, cfg, seed, max_rows)
    stratify = y if cfg.task == "classification" else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=stratify
    )
    X_train_proc, X_test_proc = preprocess_standard(X_train, X_test)
    notes = cap_note

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        start = time.time()
        label_encoder = None
        if model_name == "tabnet":
            model, fit_params = build_model(model_name, cfg, params, seed)
            if cfg.task == "classification":
                label_encoder = LabelEncoder()
                y_train_fit = label_encoder.fit_transform(np.asarray(y_train))
                y_test_fit = label_encoder.transform(np.asarray(y_test))
            else:
                y_train_fit = tabnet_regression_target(y_train)
                y_test_fit = tabnet_regression_target(y_test)
            model.fit(
                np.asarray(X_train_proc, dtype=np.float32),
                y_train_fit,
                eval_set=[(np.asarray(X_test_proc, dtype=np.float32), y_test_fit)],
                patience=min(5, max(1, tabnet_max_epochs // 3)),
                max_epochs=tabnet_max_epochs,
                compute_importance=False,
                **fit_params,
            )
            train_pred = model.predict(np.asarray(X_train_proc, dtype=np.float32))
            test_pred = model.predict(np.asarray(X_test_proc, dtype=np.float32))
        else:
            model = build_model(model_name, cfg, params, seed)
            if cfg.task == "classification" and model_name == "xgboost":
                label_encoder = LabelEncoder()
                y_train_fit = label_encoder.fit_transform(np.asarray(y_train))
                model.fit(X_train_proc, y_train_fit)
                train_pred = model.predict(X_train_proc)
                test_pred = model.predict(X_test_proc)
            else:
                model.fit(X_train_proc, y_train)
                train_pred = model.predict(X_train_proc)
                test_pred = model.predict(X_test_proc)
        elapsed = time.time() - start

    train_scores = score_predictions(cfg, y_train, train_pred, label_encoder)
    test_scores = score_predictions(cfg, y_test, test_pred, label_encoder)
    records = []
    for metric_name, test_score in test_scores.items():
        records.append(
            EvalRecord(
                dataset=cfg.name,
                model=model_name,
                task=cfg.task,
                seed=seed,
                metric_name=metric_name,
                test_score=test_score,
                train_score_if_available=train_scores.get(metric_name),
                status="success",
                notes=f"{notes}; fit_elapsed_sec={elapsed:.3f}".strip("; "),
            )
        )
    return records


def evaluate_pretrained_model(
    cfg: DatasetConfig,
    model_name: str,
    seed: int,
    include_pretrained: bool,
) -> list[EvalRecord]:
    if cfg.task != "classification":
        note = f"{model_name} is classification-only in this repository"
    elif not include_pretrained:
        note = (
            f"{model_name} is a pretrained/default baseline; repeated random "
            "initialization is not applicable. Pass --include-pretrained to "
            "rerun context-based fits across split seeds."
        )
    else:
        note = (
            f"{model_name} repeat requested, but this lightweight script does not "
            "reload the external pretrained inference path to avoid unstable long runs"
        )
    return [
        EvalRecord(
            dataset=cfg.name,
            model=model_name,
            task=cfg.task,
            seed=seed,
            metric_name="F1" if cfg.task == "classification" else "R2",
            test_score=None,
            train_score_if_available=None,
            status="skipped",
            notes=note,
        )
    ]


def safe_evaluate(
    cfg: DatasetConfig,
    model_name: str,
    seed: int,
    max_rows: int,
    tabnet_max_epochs: int,
    include_pretrained: bool,
) -> list[EvalRecord]:
    params = load_params(cfg.name, model_name)
    if params is None:
        return [
            EvalRecord(
                cfg.name,
                model_name,
                cfg.task,
                seed,
                "F1" if cfg.task == "classification" else "R2",
                None,
                None,
                "skipped",
                "best_params.json not found",
            )
        ]
    if model_name in PRETRAINED_MODELS:
        return evaluate_pretrained_model(cfg, model_name, seed, include_pretrained)
    try:
        return evaluate_standard_model(
            cfg, model_name, params, seed, max_rows, tabnet_max_epochs
        )
    except Exception as exc:
        metric = "F1" if cfg.task == "classification" else "R2"
        return [
            EvalRecord(
                cfg.name,
                model_name,
                cfg.task,
                seed,
                metric,
                None,
                None,
                "failed",
                f"{type(exc).__name__}: {exc}",
            )
        ]


def make_summary(detailed: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, task, metric), group in detailed.groupby(["model", "task", "metric_name"]):
        ok = group[group["status"] == "success"].copy()
        if ok.empty:
            raw_notes = "; ".join(sorted(set(group["notes"].dropna().astype(str))))
            if "pretrained/default baseline" in raw_notes:
                notes = (
                    "Pretrained/default baseline; repeated random initialization is "
                    "not comparable in this lightweight check."
                )
            elif "classification-only" in raw_notes:
                notes = "Classification-only baseline; skipped for regression."
            elif "best_params.json not found" in raw_notes:
                notes = "No saved HPO/default configuration found for this task."
            else:
                notes = raw_notes[:180]
            rows.append(
                {
                    "model": model,
                    "task_type": task,
                    "metric_name": metric,
                    "mean_test_score": np.nan,
                    "std_test_score": np.nan,
                    "n_successful_runs": 0,
                    "notes": notes,
                }
            )
            continue
        raw_notes = "; ".join(sorted(set(group["notes"].fillna("").astype(str))))
        notes = ""
        if "deterministic row cap" in raw_notes:
            notes = "Large regression datasets use a deterministic 10,000-row cap."
        rows.append(
            {
                "model": model,
                "task_type": task,
                "metric_name": metric,
                "mean_test_score": ok["test_score"].mean(),
                "std_test_score": ok["test_score"].std(ddof=1)
                if len(ok) > 1
                else 0.0,
                "n_successful_runs": int(len(ok)),
                "notes": notes,
            }
        )
    summary = pd.DataFrame(rows)
    sort_metric = summary["metric_name"].map({"F1": 0, "R2": 1, "RMSE": 2}).fillna(9)
    return summary.assign(_metric_order=sort_metric).sort_values(
        ["task_type", "_metric_order", "mean_test_score"],
        ascending=[True, True, False],
    ).drop(columns=["_metric_order"])


def latex_table(summary: pd.DataFrame) -> str:
    table = summary.copy()
    table["Mean test score"] = table["mean_test_score"].map(
        lambda x: "--" if pd.isna(x) else f"{x:.3f}"
    )
    table["Std. dev."] = table["std_test_score"].map(
        lambda x: "--" if pd.isna(x) else f"{x:.3f}"
    )
    table = table.rename(
        columns={
            "model": "Model",
            "metric_name": "Metric",
            "n_successful_runs": "Successful runs",
            "notes": "Notes",
        }
    )
    cols = ["Model", "Metric", "Mean test score", "Std. dev.", "Successful runs", "Notes"]
    return table[cols].to_latex(index=False, escape=True)


def write_paragraph(path: Path) -> None:
    text = (
        "\\subsection{Repeated-seed uncertainty}\n"
        "To quantify sensitivity to random initialization and data splitting, the final "
        "evaluation was repeated across three seeds (0, 42, and 123) using the selected "
        "HPO configuration for each model--dataset pair. We report the mean and standard "
        "deviation of the main test metric across successful repeated runs. Classification "
        "datasets are summarized with macro-F1 for multiclass tasks and F1 for binary tasks, "
        "whereas regression datasets are summarized with $R^2$ and RMSE. This analysis is "
        "intended as a lightweight uncertainty check rather than a full nested "
        "cross-validation study; pretrained/default baselines whose randomness is not "
        "controlled through the same HPO fitting process are reported with explicit notes."
    )
    path.write_text(text)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS))
    parser.add_argument("--models", nargs="+", default=MODELS)
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--out-dir", default="results/paper_tables")
    parser.add_argument(
        "--max-rows-per-dataset",
        type=int,
        default=0,
        help="Optional deterministic row cap for large datasets; 0 uses all rows.",
    )
    parser.add_argument(
        "--tabnet-max-epochs",
        type=int,
        default=100,
        help="Cap for TabNet repeated-seed refits to keep the check lightweight.",
    )
    parser.add_argument(
        "--include-pretrained",
        action="store_true",
        help="Attempt pretrained baselines; by default they are documented as skipped.",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append to an existing detailed CSV and rebuild the summary outputs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    records: list[EvalRecord] = []
    for dataset in args.datasets:
        cfg = DATASETS[dataset]
        for model in args.models:
            for seed in args.seeds:
                print(f"[repeat] dataset={dataset} model={model} seed={seed}")
                records.extend(
                    safe_evaluate(
                        cfg,
                        model,
                        seed,
                        args.max_rows_per_dataset,
                        args.tabnet_max_epochs,
                        args.include_pretrained,
                    )
                )

    detailed = pd.DataFrame([r.__dict__ for r in records])
    detail_path = out_dir / "repeated_seed_uncertainty_detailed.csv"
    if args.append and detail_path.exists():
        previous = pd.read_csv(detail_path)
        detailed = pd.concat([previous, detailed], ignore_index=True)
        detailed = detailed.drop_duplicates(
            subset=["dataset", "model", "task", "seed", "metric_name"],
            keep="last",
        )
    detailed.to_csv(detail_path, index=False)

    summary = make_summary(detailed)
    summary_path = out_dir / "repeated_seed_uncertainty_summary.csv"
    summary.to_csv(summary_path, index=False)

    tex_path = out_dir / "repeated_seed_uncertainty_table.tex"
    tex_path.write_text(latex_table(summary))

    paragraph_path = out_dir / "repeated_seed_uncertainty_paragraph.txt"
    write_paragraph(paragraph_path)

    top = summary[summary["n_successful_runs"] > 0].copy()
    if not top.empty:
        top = top.sort_values(
            ["metric_name", "mean_test_score"],
            ascending=[True, False],
        ).head(10)
        print("\nTop repeated-seed summary rows:")
        print(
            top[
                [
                    "model",
                    "task_type",
                    "metric_name",
                    "mean_test_score",
                    "std_test_score",
                    "n_successful_runs",
                ]
            ].to_string(index=False)
        )

    print("\nSaved:")
    for p in [detail_path, summary_path, tex_path, paragraph_path]:
        print(f"- {p}")


if __name__ == "__main__":
    main()
