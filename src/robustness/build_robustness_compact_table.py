#!/usr/bin/env python3
"""Build a compact robustness table for paper use.

Output layout:
- Rows: dataset x scenario x metric
- Columns: model scores side-by-side (fixed order)

Sources:
- results/robustness_challenges/metrics_<dataset>.csv
- results/robustness_challenges/shap_delta_summary.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Tuple


DATASETS = ["diamonds_v2", "drybean", "mnist", "telco"]
SCENARIOS = ["clean", "missingness", "high_cardinality"]
MODELS = ["catboost", "lgbm", "rf", "mlp", "tabnet", "apt", "tabpfn"]


MetricKey = Tuple[str, str, str]  # (dataset, model, scenario)


def _to_float_or_none(v: str | None) -> float | None:
    if v is None:
        return None
    s = str(v).strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _fmt(v: float | None) -> str:
    if v is None:
        return "NA"
    return f"{v:.6f}"


def load_metrics(metrics_dir: Path) -> Dict[MetricKey, dict]:
    by_key: Dict[MetricKey, dict] = {}
    for dataset in DATASETS:
        p = metrics_dir / f"metrics_{dataset}.csv"
        if not p.exists():
            continue
        with p.open(newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                ds = row.get("dataset", "").strip()
                model = row.get("model", "").strip()
                scenario = row.get("scenario", "").strip()
                if ds not in DATASETS or model not in MODELS or scenario not in SCENARIOS:
                    continue
                primary = _to_float_or_none(row.get("F1"))
                if primary is None:
                    primary = _to_float_or_none(row.get("R2"))
                by_key[(ds, model, scenario)] = {
                    "primary": primary,
                    "train_time_sec": _to_float_or_none(row.get("train_time_sec")),
                    "model_size_mb": _to_float_or_none(row.get("model_size_mb")),
                }
    return by_key


def load_shap(shap_csv: Path) -> Dict[MetricKey, dict]:
    by_key: Dict[MetricKey, dict] = {}
    if not shap_csv.exists():
        return by_key
    with shap_csv.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ds = row.get("dataset", "").strip()
            model = row.get("model", "").strip()
            scenario = row.get("scenario", "").strip()
            if ds not in DATASETS or model not in MODELS or scenario not in SCENARIOS:
                continue
            by_key[(ds, model, scenario)] = {
                "overlap_count": _to_float_or_none(row.get("overlap_count")),
                "jaccard": _to_float_or_none(row.get("jaccard")),
            }
    return by_key


def build_rows(metrics_map: Dict[MetricKey, dict], shap_map: Dict[MetricKey, dict]) -> list[dict]:
    rows: list[dict] = []
    for dataset in DATASETS:
        is_regression = dataset == "diamonds_v2"
        primary_metric_name = "R2" if is_regression else "F1"

        for scenario in SCENARIOS:
            metric_names = [primary_metric_name, "train_time_sec", "model_size_mb"]
            if scenario != "clean":
                metric_names.extend(["overlap_count", "jaccard"])

            for metric_name in metric_names:
                out = {
                    "dataset": dataset,
                    "task_type": "regression" if is_regression else "classification",
                    "scenario": scenario,
                    "metric": metric_name,
                }

                for model in MODELS:
                    key = (dataset, model, scenario)
                    v = None
                    if metric_name in {"F1", "R2"}:
                        v = metrics_map.get(key, {}).get("primary")
                    elif metric_name in {"train_time_sec", "model_size_mb"}:
                        v = metrics_map.get(key, {}).get(metric_name)
                    else:
                        v = shap_map.get(key, {}).get(metric_name)
                    out[model] = _fmt(v)
                rows.append(out)
    return rows


def write_outputs(rows: list[dict], out_csv: Path, out_note: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["dataset", "task_type", "scenario", "metric", *MODELS]
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    note = (
        "Caption note: Table reports robustness metrics by dataset, scenario, and model. "
        "Primary metric is F1 for classification datasets (drybean, mnist, telco) and R2 for "
        "the regression dataset (diamonds_v2). SHAP overlap_count and jaccard are shown for "
        "perturbed scenarios only (missingness, high_cardinality). 'NA' indicates unavailable "
        "values; notably SHAP overlap/jaccard for APT and TabPFN were not computed."
    )
    out_note.write_text(note + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build compact robustness table for the paper.")
    parser.add_argument(
        "--metrics-dir",
        type=Path,
        default=Path("results/robustness_challenges"),
    )
    parser.add_argument(
        "--shap-csv",
        type=Path,
        default=Path("results/robustness_challenges/shap_delta_summary.csv"),
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=Path("results/paper_tables/table4_robustness_compact.csv"),
    )
    parser.add_argument(
        "--out-note",
        type=Path,
        default=Path("results/paper_tables/table4_robustness_compact_caption_note.txt"),
    )
    args = parser.parse_args()

    metrics_map = load_metrics(args.metrics_dir)
    shap_map = load_shap(args.shap_csv)
    rows = build_rows(metrics_map, shap_map)
    write_outputs(rows, args.out_csv, args.out_note)

    print(f"Wrote: {args.out_csv}")
    print(f"Wrote: {args.out_note}")
    print(f"Rows: {len(rows)}")


if __name__ == "__main__":
    main()
