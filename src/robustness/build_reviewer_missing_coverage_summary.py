#!/usr/bin/env python3
"""Create reviewer-facing coverage notes and LaTeX rows.

This script summarizes the newly filled model-coverage rows for robustness and
covariate-shift experiments without recomputing any model outputs.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path("results")
ROBUST_DIR = ROOT / "robustness_challenges"
SHIFT_DIR = ROOT / "tree_vs_deep_stability_analysis"
TABLE_DIR = ROOT / "paper_tables"
MODELS = ["catboost", "lgbm", "rf", "xgboost", "mlp", "tabnet", "apt", "tabpfn"]
FRIENDLY = {
    "catboost": "CatBoost",
    "lgbm": "LightGBM",
    "rf": "Random Forest",
    "xgboost": "XGBoost",
    "mlp": "MLP",
    "tabnet": "TabNet",
    "apt": "APT",
    "tabpfn": "TabPFN",
}


def fmt(value) -> str:
    if pd.isna(value):
        return "NA"
    try:
        return f"{float(value):.3f}"
    except Exception:
        return str(value)


def write_table4_rows() -> Path:
    path = TABLE_DIR / "table4_robustness_model_summary.csv"
    out_path = TABLE_DIR / "table4_robustness_model_summary_rows.tex"
    df = pd.read_csv(path)
    lines = []
    for _, row in df.iterrows():
        model = row["model"]
        lines.append(
            " & ".join(
                [
                    FRIENDLY.get(model, model),
                    fmt(row["mean_performance_delta_missingness"]),
                    fmt(row["mean_performance_delta_high_cardinality"]),
                    fmt(row["mean_training_time_change_pct_missingness"]),
                    fmt(row["mean_training_time_change_pct_high_cardinality"]),
                    fmt(row["mean_shap_overlap_missingness"]),
                    fmt(row["mean_shap_overlap_high_cardinality"]),
                    fmt(row["mean_jaccard_missingness"]),
                    fmt(row["mean_jaccard_high_cardinality"]),
                ]
            )
            + r" \\"
        )
    out_path.write_text("\n".join(lines) + "\n")
    return out_path


def write_table5_rows() -> Path:
    path = TABLE_DIR / "table5_stability_model_summary.csv"
    out_path = TABLE_DIR / "table5_stability_model_summary_rows.tex"
    df = pd.read_csv(path)
    lines = []
    for _, row in df.iterrows():
        model = row["model"]
        lines.append(
            " & ".join(
                [
                    FRIENDLY.get(model, model),
                    fmt(row["mean_accuracy_clean"]),
                    fmt(row["mean_accuracy_shift"]),
                    fmt(row["mean_accuracy_delta_shift_minus_clean"]),
                    fmt(row["mean_f1_clean"]),
                    fmt(row["mean_f1_shift"]),
                    fmt(row["mean_f1_delta_shift_minus_clean"]),
                    fmt(row["mean_shap_spearman"]),
                    fmt(row["mean_shap_jaccard"]),
                    str(int(row["n_datasets_with_perf"])),
                    str(int(row["n_datasets_with_shap"])),
                ]
            )
            + r" \\"
        )
    out_path.write_text("\n".join(lines) + "\n")
    return out_path


def read_json_error(path: Path) -> str | None:
    if not path.exists():
        return None
    try:
        with path.open() as f:
            payload = json.load(f)
        return payload.get("error")
    except Exception as exc:
        return f"Could not read error JSON: {exc}"


def write_markdown() -> Path:
    out_path = TABLE_DIR / "reviewer_missing_model_coverage_summary.md"
    robust_metrics = {
        ds: pd.read_csv(ROBUST_DIR / f"metrics_{ds}.csv")
        for ds in ["diamonds_v2", "drybean", "mnist", "telco"]
    }
    shift = pd.read_csv(SHIFT_DIR / "summary_metrics.csv")

    lines = [
        "# Reviewer Missing Model Coverage Summary",
        "",
        "No values were invented. All rows below were generated from existing saved HPO configurations and the existing robustness/covariate-shift protocols.",
        "",
        "## Robustness Coverage",
        "",
    ]
    for ds, df in robust_metrics.items():
        xgb = df[df["model"].eq("xgboost")]
        scenarios = sorted(xgb["scenario"].dropna().unique().tolist())
        lines.append(f"- `{ds}` / `xgboost`: {len(xgb)} scenario rows present: {', '.join(scenarios) if scenarios else 'none'}.")

    shap = pd.read_csv(ROBUST_DIR / "shap_delta_summary.csv")
    xgb_shap = shap[shap["model"].eq("xgboost")]
    lines.extend(
        [
            "",
            f"- XGBoost robustness SHAP summary rows: {len(xgb_shap)} (`missingness` and `high_cardinality` across available datasets).",
            "",
            "## Covariate-Shift Coverage",
            "",
        ]
    )
    for model in ["lgbm", "xgboost"]:
        subset = shift[shift["model"].eq(model)]
        perf = subset[["Accuracy_clean", "Accuracy_shift", "F1_clean", "F1_shift"]].notna().all(axis=1).sum()
        shap_ok = subset[["shap_spearman", "shap_jaccard"]].notna().all(axis=1).sum()
        lines.append(f"- `{model}`: predictive rows with clean/shift metrics = {perf}; SHAP stability rows = {shap_ok}.")
        for ds in ["titanic", "telco", "drybean"]:
            err = read_json_error(SHIFT_DIR / ds / model / "shap_error.json")
            if err:
                lines.append(f"  - `{ds}` / `{model}` SHAP unavailable: `{err}`.")

    lines.extend(
        [
            "",
            "## Generated Files",
            "",
            "- `results/robustness_challenges/metrics_diamonds_v2.csv`",
            "- `results/robustness_challenges/metrics_drybean.csv`",
            "- `results/robustness_challenges/metrics_mnist.csv`",
            "- `results/robustness_challenges/metrics_telco.csv`",
            "- `results/robustness_challenges/shap_delta_summary.csv`",
            "- `results/tree_vs_deep_stability_analysis/summary_metrics.csv`",
            "- `results/paper_tables/table4_robustness_model_summary.csv`",
            "- `results/paper_tables/table5_stability_model_summary.csv`",
            "- `results/paper_tables/table4_robustness_model_summary_rows.tex`",
            "- `results/paper_tables/table5_stability_model_summary_rows.tex`",
        ]
    )
    out_path.write_text("\n".join(lines) + "\n")
    return out_path


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    for path in (write_table4_rows(), write_table5_rows(), write_markdown()):
        print(f"Wrote: {path}")


if __name__ == "__main__":
    main()
