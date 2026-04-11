#!/usr/bin/env python3
"""Build compact model-wise robustness summary table for paper main text.

Columns (per model):
- mean performance delta under missingness
- mean performance delta under high-cardinality
- mean training-time change under missingness
- mean training-time change under high-cardinality
- mean SHAP overlap under missingness
- mean SHAP overlap under high-cardinality
- mean Jaccard under missingness
- mean Jaccard under high-cardinality
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


MODELS = ["catboost", "lgbm", "rf", "mlp", "tabnet", "apt", "tabpfn"]


def _to_float(v: str | None) -> float | None:
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


def _read_delta_summary(path: Path) -> dict[str, dict[str, float | None]]:
    out: dict[str, dict[str, float | None]] = {m: {} for m in MODELS}
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            model = row.get("model", "").strip()
            panel = row.get("panel", "").strip()
            mean_delta = _to_float(row.get("mean_delta"))
            if model not in out:
                continue
            out[model][panel] = mean_delta
    return out


def _read_jaccard_means(path: Path) -> dict[str, dict[str, float | None]]:
    vals: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            model = row.get("model", "").strip()
            scenario = row.get("scenario", "").strip()
            if model not in MODELS or scenario not in {"missingness", "high_cardinality"}:
                continue
            v = _to_float(row.get("jaccard"))
            if v is not None:
                vals[model][scenario].append(v)

    out: dict[str, dict[str, float | None]] = {m: {"missingness": None, "high_cardinality": None} for m in MODELS}
    for m in MODELS:
        for sc in ("missingness", "high_cardinality"):
            arr = vals[m][sc]
            if arr:
                out[m][sc] = sum(arr) / len(arr)
    return out


def build_rows(
    miss: dict[str, dict[str, float | None]],
    high: dict[str, dict[str, float | None]],
    jaccard: dict[str, dict[str, float | None]],
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for m in MODELS:
        rows.append(
            {
                "model": m,
                "mean_performance_delta_missingness": _fmt(miss[m].get("performance")),
                "mean_performance_delta_high_cardinality": _fmt(high[m].get("performance")),
                "mean_training_time_change_pct_missingness": _fmt(miss[m].get("train_time_pct")),
                "mean_training_time_change_pct_high_cardinality": _fmt(high[m].get("train_time_pct")),
                "mean_shap_overlap_missingness": _fmt(miss[m].get("shap_overlap")),
                "mean_shap_overlap_high_cardinality": _fmt(high[m].get("shap_overlap")),
                "mean_jaccard_missingness": _fmt(jaccard[m].get("missingness")),
                "mean_jaccard_high_cardinality": _fmt(jaccard[m].get("high_cardinality")),
            }
        )
    return rows


def write_csv(rows: list[dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = [
        "model",
        "mean_performance_delta_missingness",
        "mean_performance_delta_high_cardinality",
        "mean_training_time_change_pct_missingness",
        "mean_training_time_change_pct_high_cardinality",
        "mean_shap_overlap_missingness",
        "mean_shap_overlap_high_cardinality",
        "mean_jaccard_missingness",
        "mean_jaccard_high_cardinality",
    ]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def write_note(path: Path) -> None:
    txt = (
        "Notes: Mean deltas and time-change values are taken from figure-level robustness "
        "summaries (missingness/high_cardinality). Mean Jaccard values are recomputed from "
        "results/robustness_challenges/shap_delta_summary.csv by averaging available entries "
        "per model and scenario. 'NA' indicates unavailable values (e.g., SHAP-based values "
        "for APT/TabPFN were not computed)."
    )
    path.write_text(txt + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build compact model-wise robustness summary table.")
    parser.add_argument(
        "--missingness-summary",
        type=Path,
        default=Path("results/paper_figure_delta_summary/figure5_missingness_robustness_summary.csv"),
    )
    parser.add_argument(
        "--highcard-summary",
        type=Path,
        default=Path("results/paper_figure_delta_summary/figure6_high_cardinality_robustness_summary.csv"),
    )
    parser.add_argument(
        "--shap-delta-summary",
        type=Path,
        default=Path("results/robustness_challenges/shap_delta_summary.csv"),
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=Path("results/paper_tables/table4_robustness_model_summary.csv"),
    )
    parser.add_argument(
        "--out-note",
        type=Path,
        default=Path("results/paper_tables/table4_robustness_model_summary_note.txt"),
    )
    args = parser.parse_args()

    miss = _read_delta_summary(args.missingness_summary)
    high = _read_delta_summary(args.highcard_summary)
    jac = _read_jaccard_means(args.shap_delta_summary)
    rows = build_rows(miss, high, jac)
    write_csv(rows, args.out_csv)
    write_note(args.out_note)

    print(f"Wrote: {args.out_csv}")
    print(f"Wrote: {args.out_note}")
    print(f"Rows: {len(rows)}")


if __name__ == "__main__":
    main()
