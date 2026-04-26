#!/usr/bin/env python3
"""Figure 2.3 replacement: predictive quality vs compute/storage trade-offs.

Outputs:
- results/paper_figures/figure2_3_model_size_comparison.png
- results/paper_figure_delta_summary/figure2_3_tradeoff_data.csv

Data sources:
- results/paper_tables/table3_hpo_models_mean_ranking.csv
  (preferred for Avg. Gap to Best (%), consistent with HPO summary table)
- results/hpo/metrics_aggregate.csv
  (used to compute model-level average elapsed_sec and model_size_mb)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle


GAP_TABLE = Path("results/paper_tables/table3_hpo_models_mean_ranking.csv")
METRICS_AGG = Path("results/hpo/metrics_aggregate.csv")

OUT_FIG = Path("results/paper_figures/figure2_3_model_size_comparison.png")
OUT_DATA = Path("results/paper_figure_delta_summary/figure2_3_tradeoff_data.csv")

MODEL_ORDER = ["xgboost", "catboost", "lgbm", "rf", "tabnet", "mlp", "tabpfn", "apt"]
MODEL_COLORS = {
    "xgboost": "#1f77b4",
    "catboost": "#ff7f0e",
    "lgbm": "#2ca02c",
    "rf": "#9467bd",
    "tabnet": "#e377c2",
    "mlp": "#d62728",
    "tabpfn": "#17becf",
    "apt": "#8c564b",
}
MODEL_MARKERS = {
    "xgboost": "o",
    "catboost": "s",
    "lgbm": "D",
    "rf": "^",
    "tabnet": "P",
    "mlp": "X",
    "tabpfn": "v",
    "apt": ">",
}


def _load_tradeoff_table() -> pd.DataFrame:
    # 1) Performance quality from paper HPO summary.
    if GAP_TABLE.exists():
        gap_df = pd.read_csv(GAP_TABLE)
        gap_df = gap_df.rename(columns={"Model": "model", "Avg. Gap to Best (%)": "avg_gap_to_best_pct"})
        gap_df["model"] = gap_df["model"].astype(str).str.strip().str.lower()
        gap_df["avg_gap_to_best_pct"] = pd.to_numeric(gap_df["avg_gap_to_best_pct"], errors="coerce")
        gap_df = gap_df[["model", "avg_gap_to_best_pct"]].dropna()
    else:
        # Fallback: compute gap directly from metrics_aggregate if table missing.
        m = pd.read_csv(METRICS_AGG)
        m["best_value"] = pd.to_numeric(m["best_value"], errors="coerce")
        m = m[m["best_value"].notna()].copy()
        best = m.groupby("dataset", as_index=False)["best_value"].max().rename(columns={"best_value": "dataset_best"})
        m = m.merge(best, on="dataset", how="left")
        m["gap_pct"] = 100.0 * (m["dataset_best"] - m["best_value"]) / m["dataset_best"].abs().clip(lower=1e-12)
        gap_df = m.groupby("model", as_index=False)["gap_pct"].mean().rename(columns={"gap_pct": "avg_gap_to_best_pct"})
        gap_df["model"] = gap_df["model"].astype(str).str.strip().str.lower()

    # 2) Cost metrics from aggregated HPO metrics.
    met = pd.read_csv(METRICS_AGG)
    met["model"] = met["model"].astype(str).str.strip().str.lower()
    met["elapsed_sec"] = pd.to_numeric(met["elapsed_sec"], errors="coerce")
    met["model_size_mb"] = pd.to_numeric(met["model_size_mb"], errors="coerce")

    cost_df = (
        met.groupby("model", as_index=False)[["elapsed_sec", "model_size_mb"]]
        .mean()
        .rename(
            columns={
                "elapsed_sec": "avg_elapsed_sec",
                "model_size_mb": "avg_model_size_mb",
            }
        )
    )

    # Merge and keep requested models.
    df = gap_df.merge(cost_df, on="model", how="inner")
    df = df[df["model"].isin(MODEL_ORDER)].copy()
    df["model"] = pd.Categorical(df["model"], categories=MODEL_ORDER, ordered=True)
    df = df.sort_values("model").reset_index(drop=True)
    return df


def main() -> None:
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    OUT_DATA.parent.mkdir(parents=True, exist_ok=True)

    df = _load_tradeoff_table()
    df.to_csv(OUT_DATA, index=False)

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.9), dpi=300)
    fig.patch.set_facecolor("white")
    fig.subplots_adjust(left=0.08, right=0.98, top=0.84, bottom=0.28, wspace=0.22)

    x_col = "avg_gap_to_best_pct"

    # Panel A: performance vs time
    ax = axes[0]
    ax.set_facecolor("#fbfbfb")
    ax.grid(True, color="#dddddd", linewidth=0.65, alpha=0.75)
    for _, r in df.iterrows():
        model = str(r["model"])
        ax.scatter(
            r[x_col],
            r["avg_elapsed_sec"],
            s=88,
            color=MODEL_COLORS.get(model, "#333333"),
            marker=MODEL_MARKERS.get(model, "o"),
            edgecolor="#222222",
            linewidth=0.8,
            zorder=4,
        )
    ax.set_title("A. Predictive Quality vs Training Time", fontsize=11)
    ax.set_xlabel("Avg. Gap to Best (%) — lower is better")
    ax.set_ylabel("Average Training Time (sec)")
    ax.set_yscale("log")
    # Keep the x-axis reversed so smaller performance gap (better) appears to the right.
    ax.invert_xaxis()
    ax.add_patch(
        Rectangle(
            (0.73, 0.0),
            0.27,
            0.30,
            transform=ax.transAxes,
            facecolor="#b6e2be",
            edgecolor="#90c999",
            linewidth=0.8,
            alpha=0.68,
            zorder=0.2,
        )
    )
    ax.text(
        0.985,
        0.035,
        "favorable\n(low gap, low cost)",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=7.7,
        color="#4f6654",
    )

    # Panel B: performance vs model size
    ax = axes[1]
    ax.set_facecolor("#fbfbfb")
    ax.grid(True, color="#dddddd", linewidth=0.65, alpha=0.75)
    for _, r in df.iterrows():
        model = str(r["model"])
        ax.scatter(
            r[x_col],
            r["avg_model_size_mb"],
            s=88,
            color=MODEL_COLORS.get(model, "#333333"),
            marker=MODEL_MARKERS.get(model, "o"),
            edgecolor="#222222",
            linewidth=0.8,
            zorder=4,
        )
    ax.set_title("B. Predictive Quality vs Model Size", fontsize=11)
    ax.set_xlabel("Avg. Gap to Best (%) — lower is better")
    ax.set_ylabel("Average Model Size (MB)")
    ax.set_yscale("log")
    # Keep the x-axis reversed so smaller performance gap (better) appears to the right.
    ax.invert_xaxis()
    ax.add_patch(
        Rectangle(
            (0.73, 0.0),
            0.27,
            0.30,
            transform=ax.transAxes,
            facecolor="#b6e2be",
            edgecolor="#90c999",
            linewidth=0.8,
            alpha=0.68,
            zorder=0.2,
        )
    )
    ax.text(
        0.985,
        0.035,
        "favorable\n(low gap, low cost)",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=7.7,
        color="#4f6654",
    )

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker=MODEL_MARKERS[m],
            color="none",
            markerfacecolor=MODEL_COLORS[m],
            markeredgecolor="#222222",
            markersize=7.6,
            linewidth=0.0,
            label=m,
        )
        for m in MODEL_ORDER
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=8,
        fontsize=8.4,
        frameon=False,
        bbox_to_anchor=(0.5, 0.06),
        handletextpad=0.3,
        columnspacing=1.0,
    )

    fig.suptitle("HPO Trade-offs: Predictive Quality vs Computational/Storage Cost", fontsize=12.5, y=0.96)
    fig.savefig(OUT_FIG, bbox_inches="tight")
    plt.close(fig)

    print("Files used:")
    print(f"- {GAP_TABLE if GAP_TABLE.exists() else METRICS_AGG} (predictive quality aggregate)")
    print(f"- {METRICS_AGG} (avg elapsed_sec and avg model_size_mb)")
    print(f"Saved figure: {OUT_FIG}")
    print(f"Saved plot data: {OUT_DATA}")


if __name__ == "__main__":
    main()


"""
Draft caption:
Figure 2.3 (replacement). Predictive performance versus computational/storage cost across
models after HPO. The x-axis reports Avg. Gap to Best (%) aggregated across datasets
(smaller is better; axis inverted so better is to the right). Panel A compares performance
quality against average training time, and Panel B compares performance quality against
average model size. Dashed lines denote Pareto-efficient models in each panel.
"""
