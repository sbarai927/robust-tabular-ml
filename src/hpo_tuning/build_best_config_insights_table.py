#!/usr/bin/env python3
"""Build compact best-config insights tables for Fig.3/Fig.4 parallel coordinates.

Inputs:
- results/hpo/best_params_aggregate.csv
- results/hpo/metrics_aggregate.csv

Outputs:
- results/paper_figure_delta_summary/figure3_4_best_config_insights_all_models.csv
- results/paper_figure_delta_summary/figure3_best_config_insights_tree_models.csv
- results/paper_figure_delta_summary/figure4_best_config_insights_neural_models.csv
"""

from __future__ import annotations

import argparse
import ast
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


CLASSIFICATION_DATASETS = ["drybean", "mnist", "telco", "titanic"]
REGRESSION_DATASETS = ["diamonds_v2", "diamonds_v3"]
KEEP_MODELS = ["catboost", "lgbm", "rf", "xgboost", "mlp", "tabnet"]
DATASET_ORDER = CLASSIFICATION_DATASETS + REGRESSION_DATASETS

MODEL_PARAM_MAP: Dict[str, List[str]] = {
    "catboost": ["iterations", "learning_rate", "depth", "l2_leaf_reg"],  # 4
    "lgbm": [  # 6
        "n_estimators",
        "num_leaves",
        "max_depth",
        "learning_rate",
        "subsample",
        "colsample_bytree",
    ],
    "rf": ["n_estimators", "max_depth", "min_samples_split", "min_samples_leaf"],  # 4
    "xgboost": [  # 5
        "n_estimators",
        "max_depth",
        "learning_rate",
        "subsample",
        "colsample_bytree",
    ],
    "mlp": ["mlp_h1", "mlp_h2", "alpha", "learning_rate_init"],  # 4
    "tabnet": [  # 6
        "n_d",
        "n_a",
        "n_steps",
        "gamma",
        "lambda_sparse",
        "learning_rate",
    ],
}


def parse_hidden_sizes(v) -> tuple[float, float]:
    if pd.isna(v):
        return (np.nan, np.nan)
    try:
        arr = ast.literal_eval(str(v))
        if isinstance(arr, (list, tuple)) and len(arr) >= 2:
            return (float(arr[0]), float(arr[1]))
        if isinstance(arr, (list, tuple)) and len(arr) == 1:
            return (float(arr[0]), np.nan)
    except Exception:
        pass
    return (np.nan, np.nan)


def build_table(best_params_path: Path, metrics_path: Path) -> pd.DataFrame:
    bp = pd.read_csv(best_params_path)
    mt = pd.read_csv(metrics_path)

    bp = bp[bp["model"].isin(KEEP_MODELS)].copy()
    mt = mt[mt["model"].isin(KEEP_MODELS)].copy()
    bp = bp[bp["dataset"].isin(DATASET_ORDER)].copy()
    mt = mt[mt["dataset"].isin(DATASET_ORDER)].copy()

    # Parse MLP hidden sizes into explicit numeric columns for cleaner axes.
    bp["mlp_h1"] = np.nan
    bp["mlp_h2"] = np.nan
    mlp_mask = bp["model"].eq("mlp")
    if mlp_mask.any():
        parsed = bp.loc[mlp_mask, "hidden_layer_sizes"].apply(parse_hidden_sizes)
        bp.loc[mlp_mask, "mlp_h1"] = parsed.apply(lambda x: x[0]).values
        bp.loc[mlp_mask, "mlp_h2"] = parsed.apply(lambda x: x[1]).values

    merged = bp.merge(
        mt[
            [
                "dataset",
                "model",
                "F1",
                "R2",
                "Accuracy",
            ]
        ],
        on=["dataset", "model"],
        how="left",
    )

    merged["task"] = np.where(
        merged["dataset"].isin(CLASSIFICATION_DATASETS),
        "classification",
        "regression",
    )
    merged["primary_score"] = np.where(
        merged["task"].eq("classification"), merged["F1"], merged["R2"]
    )
    merged["dataset_order"] = merged["dataset"].map(
        {d: i for i, d in enumerate(DATASET_ORDER)}
    )
    merged["model_family"] = np.where(
        merged["model"].isin(["catboost", "lgbm", "rf", "xgboost"]),
        "tree",
        "neural",
    )

    # Rank and gap within each dataset based on primary_score (higher is better).
    merged["dataset_rank"] = merged.groupby("dataset")["primary_score"].rank(
        ascending=False, method="min"
    )
    merged["win_flag"] = (merged["dataset_rank"] == 1).astype(int)
    merged["top2_flag"] = (merged["dataset_rank"] <= 2).astype(int)

    merged["dataset_best_score"] = merged.groupby("dataset")["primary_score"].transform("max")
    denom = merged["dataset_best_score"].abs().clip(lower=1e-12)
    merged["gap_to_best_pct"] = 100.0 * (
        merged["dataset_best_score"] - merged["primary_score"]
    ) / denom

    # Normalized score per dataset (0..1, higher better).
    ds_min = merged.groupby("dataset")["primary_score"].transform("min")
    ds_max = merged.groupby("dataset")["primary_score"].transform("max")
    spread = (ds_max - ds_min).clip(lower=1e-12)
    merged["score_norm_within_dataset"] = (merged["primary_score"] - ds_min) / spread

    # Model-level context helpers for insight.
    merged["model_mean_rank"] = merged.groupby("model")["dataset_rank"].transform("mean")
    merged["model_mean_gap_pct"] = merged.groupby("model")["gap_to_best_pct"].transform("mean")

    # Keep only selected (max 6) params per model + informative axes.
    base_cols = [
        "dataset",
        "dataset_order",
        "task",
        "model",
        "model_family",
        "primary_score",
        "dataset_rank",
        "gap_to_best_pct",
        "score_norm_within_dataset",
        "win_flag",
        "top2_flag",
        "model_mean_rank",
        "model_mean_gap_pct",
    ]
    param_cols = sorted({p for ps in MODEL_PARAM_MAP.values() for p in ps})

    keep_cols = base_cols + [c for c in param_cols if c in merged.columns]
    out = merged[keep_cols].copy()

    # Make params inactive for unrelated models to avoid axis noise.
    for model, pcols in MODEL_PARAM_MAP.items():
        inactive_cols = [c for c in param_cols if c not in pcols and c in out.columns]
        mask = out["model"].eq(model)
        if inactive_cols:
            out.loc[mask, inactive_cols] = np.nan

    # Stable ordering for easy panel creation.
    out = out.sort_values(
        by=["model_family", "model", "dataset_order"],
        ascending=[True, True, True],
    ).reset_index(drop=True)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Build best-config insight tables for parallel coordinates.")
    parser.add_argument(
        "--best-params",
        type=Path,
        default=Path("results/hpo/best_params_aggregate.csv"),
    )
    parser.add_argument(
        "--metrics",
        type=Path,
        default=Path("results/hpo/metrics_aggregate.csv"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/paper_figure_delta_summary"),
    )
    args = parser.parse_args()

    out = build_table(args.best_params, args.metrics)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    p_all = args.out_dir / "figure3_4_best_config_insights_all_models.csv"
    p_tree = args.out_dir / "figure3_best_config_insights_tree_models.csv"
    p_neural = args.out_dir / "figure4_best_config_insights_neural_models.csv"

    out.to_csv(p_all, index=False)
    out[out["model_family"].eq("tree")].to_csv(p_tree, index=False)
    out[out["model_family"].eq("neural")].to_csv(p_neural, index=False)

    print("Saved:")
    print(f"- {p_all}")
    print(f"- {p_tree}")
    print(f"- {p_neural}")
    print(f"Rows: all={len(out)}, tree={len(out[out['model_family'].eq('tree')])}, neural={len(out[out['model_family'].eq('neural')])}")


if __name__ == "__main__":
    main()

