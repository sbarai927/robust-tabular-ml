#!/usr/bin/env python3
"""Figure 7: clean vs shifted predictive performance as paired-point slope charts.

Data source:
- results/tree_vs_deep_stability_analysis/summary_metrics.csv

Aggregation:
- Dataset-level values are read for drybean, telco, titanic.
- For each model, panel means are computed as the mean of available dataset-level
  clean and shifted scores separately.

Why this redesign is stronger than the previous dumbbell:
- It shows the clean->shift direction explicitly per model with two anchored
  positions (Clean and Shifted), while still exposing underlying dataset-level
  variability as faint background points.
- Visual emphasis remains on model-level degradation magnitude and shifted-level
  competitiveness, which is the key cross-experiment interpretation.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SUMMARY_CSV = Path("results/tree_vs_deep_stability_analysis/summary_metrics.csv")
OUT_FIG = Path("results/paper_figures/figure7_clean_vs_shifted_dumbbell.png")
OUT_DATA = Path("results/paper_figure_delta_summary/figure7_clean_vs_shifted_slope_data.csv")

TARGET_DATASETS = ["drybean", "telco", "titanic"]
TARGET_MODELS = ["catboost", "rf", "mlp", "tabnet", "tabpfn", "apt", "lgbm"]

MODEL_COLORS = {
    "catboost": "#1f77b4",
    "rf": "#9467bd",
    "mlp": "#d62728",
    "tabnet": "#ff7f0e",
    "tabpfn": "#17becf",
    "apt": "#8c564b",
    "lgbm": "#2ca02c",
}


def _load() -> pd.DataFrame:
    df = pd.read_csv(SUMMARY_CSV)
    df = df[df["dataset"].isin(TARGET_DATASETS)].copy()
    df["model"] = df["model"].str.lower()
    for c in ["Accuracy_clean", "Accuracy_shift", "F1_clean", "F1_shift"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _build_plot_data(df: pd.DataFrame, metric: str) -> tuple[pd.DataFrame, list[str]]:
    if metric == "accuracy":
        c_col, s_col = "Accuracy_clean", "Accuracy_shift"
    else:
        c_col, s_col = "F1_clean", "F1_shift"

    rows = []
    unavailable = []
    for model in TARGET_MODELS:
        dm = df[df["model"] == model].copy()
        dm = dm[dm[c_col].notna() & dm[s_col].notna()]
        if dm.empty:
            unavailable.append(model)
            continue
        rows.append(
            {
                "model": model,
                "metric": metric,
                "clean_mean": float(dm[c_col].mean()),
                "shift_mean": float(dm[s_col].mean()),
                "delta_shift_minus_clean": float(dm[s_col].mean() - dm[c_col].mean()),
                "n_datasets": int(dm.shape[0]),
                "clean_values": dm[c_col].tolist(),
                "shift_values": dm[s_col].tolist(),
            }
        )
    pdf = pd.DataFrame(rows)
    if not pdf.empty:
        # Keep stable order in plot.
        pdf["model"] = pd.Categorical(pdf["model"], categories=TARGET_MODELS, ordered=True)
        pdf = pdf.sort_values("model").reset_index(drop=True)
    return pdf, unavailable


def _nudged_positions(yvals: np.ndarray, min_gap: float = 0.018) -> np.ndarray:
    """Simple 1D label collision reduction by enforcing a minimum vertical gap."""
    if yvals.size == 0:
        return yvals
    order = np.argsort(yvals)
    out = yvals.copy()
    prev = -10.0
    for idx in order:
        y = out[idx]
        if y - prev < min_gap:
            y = prev + min_gap
            out[idx] = y
        prev = y
    return out


def _panel(ax: plt.Axes, pdf: pd.DataFrame, metric_name: str) -> None:
    ax.set_facecolor("white")
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.7, alpha=0.8)
    ax.grid(axis="x", color="#ececec", linewidth=0.6, alpha=0.7)

    x_clean, x_shift = 0, 1
    rng = np.random.default_rng(42)

    # Faint dataset-level context points.
    for _, r in pdf.iterrows():
        color = MODEL_COLORS[r["model"]]
        clean_vals = np.asarray(r["clean_values"], dtype=float)
        shift_vals = np.asarray(r["shift_values"], dtype=float)
        jc = rng.normal(0.0, 0.022, size=clean_vals.size)
        js = rng.normal(0.0, 0.022, size=shift_vals.size)
        ax.scatter(
            np.full(clean_vals.size, x_clean) + jc,
            clean_vals,
            s=22,
            color=color,
            alpha=0.22,
            edgecolors="none",
            zorder=1,
        )
        ax.scatter(
            np.full(shift_vals.size, x_shift) + js,
            shift_vals,
            s=22,
            color=color,
            alpha=0.22,
            edgecolors="none",
            zorder=1,
        )

    # Main clean->shift mean transitions.
    for _, r in pdf.iterrows():
        model = r["model"]
        color = MODEL_COLORS[model]
        y0 = float(r["clean_mean"])
        y1 = float(r["shift_mean"])
        ax.plot([x_clean, x_shift], [y0, y1], color=color, linewidth=2.2, alpha=0.95, zorder=3)
        ax.scatter([x_clean, x_shift], [y0, y1], color=color, s=60, zorder=4, edgecolor="white", linewidth=0.7)

    # Direct labels at shifted end with gentle de-overlap.
    y_shift = pdf["shift_mean"].to_numpy(dtype=float)
    y_label = _nudged_positions(y_shift, min_gap=0.02)
    for i, (_, r) in enumerate(pdf.iterrows()):
        model = r["model"]
        color = MODEL_COLORS[model]
        ax.text(
            x_shift + 0.055,
            float(y_label[i]),
            model,
            color=color,
            fontsize=8.3,
            ha="left",
            va="center",
        )

    all_vals = np.concatenate([pdf["clean_mean"].to_numpy(), pdf["shift_mean"].to_numpy()])
    y_min = max(0.0, float(np.min(all_vals) - 0.08))
    y_max = min(1.02, float(np.max(all_vals) + 0.08))
    ax.set_ylim(y_min, y_max)
    ax.set_xlim(-0.22, 1.28)
    ax.set_xticks([x_clean, x_shift], ["Clean", "Shifted"])
    ax.set_ylabel(metric_name)


def main() -> None:
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    OUT_DATA.parent.mkdir(parents=True, exist_ok=True)

    df = _load()
    acc_df, acc_missing = _build_plot_data(df, "accuracy")
    f1_df, f1_missing = _build_plot_data(df, "f1")

    # Save a compact data table used by the figure for traceability.
    out_df = pd.concat([acc_df, f1_df], ignore_index=True)
    out_df = out_df.drop(columns=["clean_values", "shift_values"])
    out_df.to_csv(OUT_DATA, index=False)

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.6), dpi=300)
    fig.subplots_adjust(left=0.08, right=0.97, top=0.84, bottom=0.17, wspace=0.22)

    _panel(axes[0], acc_df, "Accuracy")
    _panel(axes[1], f1_df, "F1")
    axes[0].set_title("A. Accuracy", fontsize=11)
    axes[1].set_title("B. F1", fontsize=11)

    # Compact explanatory legend for symbol semantics.
    legend_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#666666", markeredgecolor="white",
               markersize=7, label="Mean point"),
        Line2D([0], [0], color="#666666", linewidth=2, label="Clean \u2192 Shifted mean transition"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#666666", alpha=0.25,
               markersize=5, label="Dataset-level point"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=3,
        frameon=False,
        fontsize=8.4,
        bbox_to_anchor=(0.5, 0.035),
    )

    missing_models = sorted(set(acc_missing) | set(f1_missing))
    if missing_models:
        fig.text(
            0.5,
            0.095,
            "Unavailable in this shift summary: " + ", ".join(missing_models),
            ha="center",
            va="center",
            fontsize=8.2,
            color="#666666",
        )

    fig.suptitle(
        "Clean-to-Shift Predictive Performance by Model (means over DryBean, Telco, Titanic)",
        fontsize=12.5,
        y=0.97,
    )

    fig.savefig(OUT_FIG, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved figure: {OUT_FIG}")
    print(f"Saved data: {OUT_DATA}")


if __name__ == "__main__":
    main()


"""
Draft paper caption:
Figure 7: Clean-to-shift predictive performance under covariate shift for Accuracy (A)
and F1 (B). For each model, the two main points denote means across DryBean, Telco,
and Titanic; the connecting segment shows the direction and magnitude of degradation
from clean to shifted conditions. Faint points show dataset-level values to expose
variability without overwhelming the mean transition pattern.
"""

