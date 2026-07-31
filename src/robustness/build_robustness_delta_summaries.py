#!/usr/bin/env python3
"""Rebuild robustness delta summaries from saved per-scenario outputs.

This script does not retrain models. It reads:
- results/robustness_challenges/metrics_<dataset>.csv
- results/robustness_challenges/shap_delta_<dataset>_<model>.json

and writes the aggregate files used by the paper figures/tables.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


DATASETS = ["diamonds_v2", "drybean", "mnist", "telco"]
MODELS = ["catboost", "lgbm", "rf", "xgboost", "mlp", "tabnet", "apt", "tabpfn"]
SCENARIOS = ["missingness", "high_cardinality"]


def primary_metric(row: pd.Series) -> float | None:
    """Use R2 for regression rows and F1 for classification rows."""
    for key in ("R2", "F1"):
        if key in row and pd.notna(row[key]):
            return float(row[key])
    return None


def load_metrics(robust_dir: Path) -> pd.DataFrame:
    frames = []
    for dataset in DATASETS:
        path = robust_dir / f"metrics_{dataset}.csv"
        if path.exists():
            df = pd.read_csv(path)
            if "dataset" not in df.columns:
                df["dataset"] = dataset
            else:
                df["dataset"] = df["dataset"].fillna(dataset)
                df.loc[df["dataset"].astype(str).str.strip().eq(""), "dataset"] = dataset
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def rebuild_shap_delta_summary(robust_dir: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(robust_dir.glob("shap_delta_*_*.json")):
        stem = path.stem.removeprefix("shap_delta_")
        dataset = next((ds for ds in DATASETS if stem.startswith(f"{ds}_")), None)
        if dataset is None:
            continue
        model = stem.removeprefix(f"{dataset}_")
        if model not in MODELS:
            continue
        with path.open() as f:
            payload = json.load(f)
        clean_top = payload.get("clean_top_k", [])
        for scenario, comp in payload.get("comparisons", {}).items():
            if scenario not in SCENARIOS:
                continue
            rows.append(
                {
                    "dataset": dataset,
                    "model": model,
                    "scenario": scenario,
                    "overlap_count": comp.get("overlap_count"),
                    "jaccard": comp.get("jaccard"),
                    "clean_top_k": json.dumps(clean_top),
                    "perturbed_top_k": json.dumps(comp.get("top_k", [])),
                    "source_file": str(path),
                }
            )
    out = pd.DataFrame(rows)
    out_path = robust_dir / "shap_delta_summary.csv"
    out.to_csv(out_path, index=False)
    return out


def build_scenario_summary(metrics: pd.DataFrame, shap_summary: pd.DataFrame, scenario: str) -> pd.DataFrame:
    rows = []
    for model in MODELS:
        perf_deltas = []
        time_pct = []
        size_pct = []
        for dataset in DATASETS:
            subset = metrics[(metrics["dataset"] == dataset) & (metrics["model"] == model)]
            clean = subset[subset["scenario"] == "clean"]
            stressed = subset[subset["scenario"] == scenario]
            if clean.empty or stressed.empty:
                continue
            clean_row = clean.iloc[0]
            stressed_row = stressed.iloc[0]
            clean_perf = primary_metric(clean_row)
            stressed_perf = primary_metric(stressed_row)
            if clean_perf is not None and stressed_perf is not None:
                perf_deltas.append(stressed_perf - clean_perf)
            for col, arr in (("train_time_sec", time_pct), ("model_size_mb", size_pct)):
                if col in clean_row.index and col in stressed_row.index:
                    base = clean_row[col]
                    shifted = stressed_row[col]
                    if pd.notna(base) and pd.notna(shifted) and abs(float(base)) > 1e-12:
                        arr.append(100.0 * (float(shifted) - float(base)) / abs(float(base)))

        for panel, values in (
            ("performance", perf_deltas),
            ("train_time_pct", time_pct),
            ("model_size_pct", size_pct),
        ):
            s = pd.Series(values, dtype="float64")
            rows.append(
                {
                    "model": model,
                    "mean_delta": None if s.empty else s.mean(),
                    "std_delta": None if s.empty else s.std(),
                    "panel": panel,
                    "scenario": scenario,
                }
            )

        shap_vals = shap_summary[
            (shap_summary["model"] == model) & (shap_summary["scenario"] == scenario)
        ]["overlap_count"]
        rows.append(
            {
                "model": model,
                "mean_delta": None if shap_vals.empty else shap_vals.mean(),
                "std_delta": None if shap_vals.empty else shap_vals.std(),
                "panel": "shap_overlap",
                "scenario": scenario,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build robustness delta summaries.")
    parser.add_argument(
        "--robust-dir",
        type=Path,
        default=Path("results/robustness_challenges"),
    )
    parser.add_argument(
        "--figure-dir",
        type=Path,
        default=Path("results/paper_figure_delta_summary"),
    )
    args = parser.parse_args()
    args.figure_dir.mkdir(parents=True, exist_ok=True)

    metrics = load_metrics(args.robust_dir)
    shap_summary = rebuild_shap_delta_summary(args.robust_dir)
    outputs = {
        "missingness": args.figure_dir / "figure5_missingness_robustness_summary.csv",
        "high_cardinality": args.figure_dir / "figure6_high_cardinality_robustness_summary.csv",
    }
    for scenario, path in outputs.items():
        df = build_scenario_summary(metrics, shap_summary, scenario)
        df.to_csv(path, index=False)
        print(f"Wrote: {path}")
    print(f"Wrote: {args.robust_dir / 'shap_delta_summary.csv'}")


if __name__ == "__main__":
    main()
