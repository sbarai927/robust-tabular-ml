#!/usr/bin/env python3
"""Build publication-ready HPO summary table from aggregated benchmark outputs.

This script uses existing aggregate metrics only (no retraining) and computes:
- Per-dataset model ranks (task-aware metric)
- Per-model summary statistics across datasets
- CSV + LaTeX outputs for paper tables
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

CLASSIFICATION_DATASETS = ["drybean", "mnist", "telco", "titanic"]
REGRESSION_DATASETS = ["diamonds_v2", "diamonds_v3"]

COLUMN_ORDER = [
    "Model",
    "Mean Rank (Cls.)",
    "Mean Rank (Reg.)",
    "Mean Rank (Overall)",
    "Wins",
    "Top-2",
    "Avg. Gap to Best (%)",
    "Time Rank",
    "Size Rank",
]


def _rank_within_dataset(
    df: pd.DataFrame,
    value_col: str,
    ascending: bool,
    rank_col: str,
) -> pd.DataFrame:
    out = df.copy()
    out[rank_col] = np.nan
    for ds, g in out.groupby("dataset", dropna=False):
        valid_idx = g[g[value_col].notna()].index
        if len(valid_idx) == 0:
            continue
        out.loc[valid_idx, rank_col] = out.loc[valid_idx, value_col].rank(
            ascending=ascending, method="min"
        )
    return out


def _safe_mean(series: pd.Series) -> float:
    clean = series.dropna()
    if clean.empty:
        return np.nan
    return float(clean.mean())


def _gap_to_best_per_dataset(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []
    for ds, g in df.groupby("dataset", dropna=False):
        valid = g[g["score_for_ranking"].notna()].copy()
        if valid.empty:
            continue
        best_score = float(valid["score_for_ranking"].max())
        denom = max(abs(best_score), 1e-12)
        valid["gap_percent"] = 100.0 * (best_score - valid["score_for_ranking"]) / denom
        rows.append(valid[["dataset", "model", "gap_percent"]])
    if not rows:
        return pd.DataFrame(columns=["dataset", "model", "gap_percent"])
    return pd.concat(rows, ignore_index=True)


def _fmt_num(x: float, ndigits: int = 2) -> str:
    if pd.isna(x):
        return "--"
    return f"{x:.{ndigits}f}"


def _build_latex_table(df: pd.DataFrame) -> str:
    # Identify best values (ties allowed), excluding NaN.
    best_low_cols = [
        "Mean Rank (Cls.)",
        "Mean Rank (Reg.)",
        "Mean Rank (Overall)",
        "Avg. Gap to Best (%)",
        "Time Rank",
        "Size Rank",
    ]
    best_high_cols = ["Wins", "Top-2"]

    best_masks: Dict[str, pd.Series] = {}
    for col in best_low_cols:
        s = df[col]
        min_val = s.dropna().min() if s.notna().any() else np.nan
        best_masks[col] = s.eq(min_val) if pd.notna(min_val) else pd.Series(False, index=df.index)
    for col in best_high_cols:
        s = df[col]
        max_val = s.dropna().max() if s.notna().any() else np.nan
        best_masks[col] = s.eq(max_val) if pd.notna(max_val) else pd.Series(False, index=df.index)

    def maybe_bold(row_idx: int, col: str, value: str) -> str:
        if best_masks[col].iloc[row_idx]:
            return f"\\textbf{{{value}}}"
        return value

    lines: List[str] = []
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\scriptsize")
    lines.append("\\setlength{\\tabcolsep}{3.5pt}")
    lines.append("\\resizebox{\\columnwidth}{!}{%")
    lines.append("\\begin{tabular}{lrrrrrrrr}")
    lines.append("\\toprule")
    lines.append(
        "Model & Mean Rank (Cls.) & Mean Rank (Reg.) & Mean Rank (Overall) & Wins & Top-2 & Avg. Gap to Best (\\%) & Time Rank & Size Rank \\\\"
    )
    lines.append("\\midrule")

    for i, (_, r) in enumerate(df.iterrows()):
        vals = {
            "Model": str(r["Model"]),
            "Mean Rank (Cls.)": _fmt_num(r["Mean Rank (Cls.)"], 2),
            "Mean Rank (Reg.)": _fmt_num(r["Mean Rank (Reg.)"], 2),
            "Mean Rank (Overall)": _fmt_num(r["Mean Rank (Overall)"], 2),
            "Wins": str(int(r["Wins"])),
            "Top-2": str(int(r["Top-2"])),
            "Avg. Gap to Best (%)": _fmt_num(r["Avg. Gap to Best (%)"], 2),
            "Time Rank": _fmt_num(r["Time Rank"], 2),
            "Size Rank": _fmt_num(r["Size Rank"], 2),
        }

        row_cells = [vals["Model"]]
        for col in COLUMN_ORDER[1:]:
            row_cells.append(maybe_bold(i, col, vals[col]))
        lines.append(" & ".join(row_cells) + " \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("}")
    lines.append(
        "\\caption{HPO model summary across datasets. Ranks are computed within each dataset first, then averaged across datasets. Lower is better for rank/gap/time/size; higher is better for Wins and Top-2.}"
    )
    lines.append("\\label{tab:hpo_summary}")
    lines.append("\\end{table}")
    return "\n".join(lines) + "\n"


def build_summary(metrics_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(metrics_path)

    required_cols = {
        "dataset",
        "model",
        "F1",
        "R2",
        "elapsed_sec",
        "model_size_mb",
    }
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns in {metrics_path}: {sorted(missing_cols)}")

    # Keep only requested datasets.
    target_datasets = set(CLASSIFICATION_DATASETS + REGRESSION_DATASETS)
    df = df[df["dataset"].isin(target_datasets)].copy()

    # Metric used for ranking/gap (task-aware).
    df["task_group"] = np.where(df["dataset"].isin(CLASSIFICATION_DATASETS), "classification", "regression")
    df["score_for_ranking"] = np.where(df["task_group"].eq("classification"), df["F1"], df["R2"])

    # Ranks per dataset.
    perf_ranked = _rank_within_dataset(df, "score_for_ranking", ascending=False, rank_col="perf_rank")
    time_ranked = _rank_within_dataset(df, "elapsed_sec", ascending=True, rank_col="time_rank")
    size_ranked = _rank_within_dataset(df, "model_size_mb", ascending=True, rank_col="size_rank")

    ranked = perf_ranked[["dataset", "model", "task_group", "score_for_ranking", "perf_rank"]].merge(
        time_ranked[["dataset", "model", "time_rank"]], on=["dataset", "model"], how="left"
    ).merge(
        size_ranked[["dataset", "model", "size_rank"]], on=["dataset", "model"], how="left"
    )

    gaps = _gap_to_best_per_dataset(ranked)
    ranked = ranked.merge(gaps, on=["dataset", "model"], how="left")

    models = sorted(ranked["model"].dropna().unique().tolist())
    rows: List[Dict[str, float]] = []

    for m in models:
        gm = ranked[ranked["model"] == m].copy()
        cls = gm[gm["task_group"] == "classification"]
        reg = gm[gm["task_group"] == "regression"]

        row = {
            "Model": m,
            "Mean Rank (Cls.)": _safe_mean(cls["perf_rank"]),
            "Mean Rank (Reg.)": _safe_mean(reg["perf_rank"]),
            "Mean Rank (Overall)": _safe_mean(gm["perf_rank"]),
            "Wins": int((gm["perf_rank"] == 1).sum()),
            "Top-2": int((gm["perf_rank"] <= 2).sum()),
            "Avg. Gap to Best (%)": _safe_mean(gm["gap_percent"]),
            "Time Rank": _safe_mean(gm["time_rank"]),
            "Size Rank": _safe_mean(gm["size_rank"]),
        }
        rows.append(row)

    summary = pd.DataFrame(rows)
    summary = summary.sort_values(by="Mean Rank (Overall)", ascending=True, na_position="last").reset_index(drop=True)

    # dataset-level detail useful for traceability/reproducibility
    detail = ranked.sort_values(["dataset", "perf_rank", "model"], na_position="last").reset_index(drop=True)
    return summary, detail


def write_outputs(
    summary: pd.DataFrame,
    out_csv: Path,
    out_tex: Path,
    out_caption: Path,
    out_notes: Path,
) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    summary.to_csv(out_csv, index=False)
    out_tex.write_text(_build_latex_table(summary), encoding="utf-8")

    caption = (
        "HPO summary across tabular benchmarks. For each dataset, models are ranked first using task-"
        "appropriate predictive score (F1 for classification, R^2 for regression), then ranks are averaged "
        "across datasets. The table combines predictive consistency (mean rank, wins, top-2, gap-to-best) "
        "with efficiency/compactness proxies (time rank, size rank)."
    )
    out_caption.write_text(caption + "\n", encoding="utf-8")

    notes = (
        "Why mean rank: absolute metric scales differ across datasets, so within-dataset ranking provides a "
        "fair cross-dataset comparison baseline.\n"
        "Why mean rank is not enough: a model can have decent average rank yet rarely win, or be highly "
        "inconsistent across datasets.\n"
        "How complementary columns help:\n"
        "- Wins: counts outright best finishes (peak performance frequency).\n"
        "- Top-2: counts near-best finishes (consistency near the frontier).\n"
        "- Avg. Gap to Best (%): quantifies practical distance from the winner, not only ordinal rank.\n"
        "- Avg. Time Rank: compares relative training-speed efficiency per dataset.\n"
        "- Avg. Size Rank: compares relative model compactness per dataset.\n"
    )
    out_notes.write_text(notes, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build paper-ready HPO summary table from aggregate metrics.")
    parser.add_argument(
        "--metrics",
        type=Path,
        default=Path("results/hpo/metrics_aggregate.csv"),
        help="Path to aggregated HPO metrics CSV.",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=Path("results/paper_tables/hpo_summary_table.csv"),
        help="Output CSV path.",
    )
    parser.add_argument(
        "--out-tex",
        type=Path,
        default=Path("results/paper_tables/hpo_summary_table.tex"),
        help="Output LaTeX path.",
    )
    parser.add_argument(
        "--out-caption",
        type=Path,
        default=Path("results/paper_tables/hpo_summary_caption.txt"),
        help="Output caption text path.",
    )
    parser.add_argument(
        "--out-notes",
        type=Path,
        default=Path("results/paper_tables/hpo_summary_notes.txt"),
        help="Output notes text path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary, _detail = build_summary(args.metrics)

    # Round output columns for table readability while keeping integers exact.
    for col in [
        "Mean Rank (Cls.)",
        "Mean Rank (Reg.)",
        "Mean Rank (Overall)",
        "Avg. Gap to Best (%)",
        "Time Rank",
        "Size Rank",
    ]:
        summary[col] = summary[col].round(2)

    write_outputs(summary, args.out_csv, args.out_tex, args.out_caption, args.out_notes)

    top3 = summary.head(3)
    print("Top 3 models by Mean Rank (Overall):")
    for _, r in top3.iterrows():
        print(f"- {r['Model']}: {r['Mean Rank (Overall)']:.2f}")


if __name__ == "__main__":
    main()
