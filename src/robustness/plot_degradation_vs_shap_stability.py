#!/usr/bin/env python3
"""Plot predictive-retention vs SHAP stability across perturbation scenarios.

This script creates a publication-oriented 3-panel figure:
  A) Missingness
  B) High-cardinality stress
  C) Covariate shift

Design goals:
- Scientific honesty: do not hide models with missing SHAP values.
- Include APT/TabPFN wherever predictive data exists.
- Direct model labeling with consistent model identity across panels.

Variability and grouping used in this figure:
- Horizontal error bar: std. dev. of dataset-level predictive retention
  (retention = perturbed_score - clean_score) for each model/panel.
- Vertical error bar: std. dev. of dataset-level SHAP Jaccard (when available).
- Family overlays: covariance ellipses over model-level mean points
  (tree family: catboost/lgbm/rf; neural family: mlp/tabnet).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
from matplotlib.patches import Ellipse, Rectangle
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

# Force non-interactive backend for reproducible headless script execution.
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path("results")
OUT_DIR = ROOT / "paper_figures"
OUT_DATA_DIR = ROOT / "paper_figure_delta_summary"

ROBUSTNESS_DIR = ROOT / "robustness_challenges"
SHAP_DELTA_SUMMARY = ROOT / "robustness_challenges" / "shap_delta_summary.csv"
SHIFT_SUMMARY = ROOT / "tree_vs_deep_stability_analysis" / "summary_metrics.csv"

FIG_PATH = OUT_DIR / "figure10_degradation_vs_shap_stability.png"
DATA_PATH = OUT_DATA_DIR / "figure10_degradation_vs_shap_stability_data.csv"

# Consistent model style across panels.
MODEL_STYLE = {
    "catboost": {"color": "#1f77b4", "marker": "s"},
    "lgbm": {"color": "#2ca02c", "marker": "s"},
    "rf": {"color": "#9467bd", "marker": "s"},
    "mlp": {"color": "#d62728", "marker": "o"},
    "tabnet": {"color": "#ff7f0e", "marker": "o"},
    "apt": {"color": "#8c564b", "marker": "D"},
    "tabpfn": {"color": "#17becf", "marker": "D"},
}

MODEL_FAMILY = {
    "catboost": "tree",
    "lgbm": "tree",
    "rf": "tree",
    "mlp": "neural",
    "tabnet": "neural",
    "apt": "pretrained",
    "tabpfn": "pretrained",
}

FAMILY_STYLE = {
    "tree": {"color": "#5A84B1", "alpha": 0.28, "label": "Tree-based"},
    "neural": {"color": "#D98988", "alpha": 0.28, "label": "Neural"},
}


plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
    }
)


def _primary_metric(row: pd.Series) -> float | None:
    # F1 for classification rows, R2 for regression row(s).
    if "F1" in row and pd.notna(row["F1"]):
        return float(row["F1"])
    if "R2" in row and pd.notna(row["R2"]):
        return float(row["R2"])
    # Fallback for rows where F1 was not recorded but Accuracy exists.
    if "Accuracy" in row and pd.notna(row["Accuracy"]):
        return float(row["Accuracy"])
    return None


def _agg_mean_std(vals: list[float]) -> tuple[float | None, float | None, int]:
    if len(vals) == 0:
        return None, None, 0
    arr = np.asarray(vals, dtype=float)
    return float(np.mean(arr)), float(np.std(arr, ddof=0)), int(len(arr))


def build_panel_ab(scenario: str) -> pd.DataFrame:
    """Build A/B panel using dataset-level robustness outputs.

    Predictive retention per dataset/model:
      retention = primary_metric(perturbed) - primary_metric(clean)
      where primary_metric is F1 (classification) or R2 (regression).

    SHAP stability per dataset/model:
      shap_jaccard from shap_delta_summary.
    """
    metrics_parts = []
    for p in sorted(ROBUSTNESS_DIR.glob("metrics_*.csv")):
        dfp = pd.read_csv(p)
        # Some exported files have blank dataset column; recover from filename.
        ds_from_file = p.stem.replace("metrics_", "", 1)
        if "dataset" not in dfp.columns:
            dfp["dataset"] = ds_from_file
        else:
            dfp["dataset"] = dfp["dataset"].fillna(ds_from_file)
        metrics_parts.append(dfp)
    metrics = pd.concat(metrics_parts, ignore_index=True)

    # Compute primary metric row-wise.
    metrics["primary_metric"] = metrics.apply(_primary_metric, axis=1)
    metrics = metrics[metrics["primary_metric"].notna()].copy()
    # Defensive de-duplication: if repeated rows exist for same dataset/model/scenario,
    # collapse them before retention computation to avoid Cartesian inflation.
    metrics = (
        metrics.groupby(["dataset", "model", "scenario"], as_index=False)["primary_metric"]
        .mean()
    )

    # Build clean baseline lookup.
    clean = metrics[metrics["scenario"] == "clean"][
        ["dataset", "model", "primary_metric"]
    ].rename(columns={"primary_metric": "clean_metric"})

    pert = metrics[metrics["scenario"] == scenario][
        ["dataset", "model", "primary_metric"]
    ].rename(columns={"primary_metric": "pert_metric"})

    ret = pert.merge(clean, on=["dataset", "model"], how="inner")
    ret["predictive_retention_dataset"] = ret["pert_metric"] - ret["clean_metric"]

    # SHAP jaccard by dataset/model/scenario.
    shap = pd.read_csv(SHAP_DELTA_SUMMARY)
    shap["jaccard"] = pd.to_numeric(shap["jaccard"], errors="coerce")
    shap = shap[shap["scenario"] == scenario][["dataset", "model", "jaccard"]].copy()

    # Aggregate x/y variability separately.
    x_stats = (
        ret.groupby("model")["predictive_retention_dataset"]
        .apply(list)
        .to_dict()
    )
    y_stats = (
        shap.dropna(subset=["jaccard"]).groupby("model")["jaccard"].apply(list).to_dict()
    )

    models = sorted(set(ret["model"].unique()) | set(shap["model"].unique()))
    rows = []
    for m in models:
        x_mean, x_std, n_x = _agg_mean_std(x_stats.get(m, []))
        y_mean, y_std, n_y = _agg_mean_std(y_stats.get(m, []))
        rows.append(
            {
                "panel": scenario,
                "model": m,
                "predictive_retention": x_mean,
                "predictive_retention_std": x_std,
                "n_retention": n_x,
                "shap_jaccard": y_mean,
                "shap_jaccard_std": y_std,
                "n_shap": n_y,
                "predictive_source": "metrics_*.csv",
                "shap_source": SHAP_DELTA_SUMMARY.name,
            }
        )
    return pd.DataFrame(rows)


def build_panel_c() -> pd.DataFrame:
    """Build C panel from covariate-shift stability summary.

    Per dataset/model:
      predictive_retention_dataset = F1_shift - F1_clean
      shap_jaccard_dataset = shap_jaccard
    """
    df = pd.read_csv(SHIFT_SUMMARY)
    for col in ["F1_clean", "F1_shift", "shap_jaccard"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Predictive retention available even when SHAP missing.
    pred = df[df["F1_clean"].notna() & df["F1_shift"].notna()].copy()
    pred["predictive_retention_dataset"] = pred["F1_shift"] - pred["F1_clean"]

    x_stats = pred.groupby("model")["predictive_retention_dataset"].apply(list).to_dict()
    y_stats = (
        pred.dropna(subset=["shap_jaccard"])
        .groupby("model")["shap_jaccard"]
        .apply(list)
        .to_dict()
    )

    models = sorted(set(pred["model"].unique()))
    rows = []
    for m in models:
        x_mean, x_std, n_x = _agg_mean_std(x_stats.get(m, []))
        y_mean, y_std, n_y = _agg_mean_std(y_stats.get(m, []))
        rows.append(
            {
                "panel": "covariate_shift",
                "model": m,
                "predictive_retention": x_mean,
                "predictive_retention_std": x_std,
                "n_retention": n_x,
                "shap_jaccard": y_mean,
                "shap_jaccard_std": y_std,
                "n_shap": n_y,
                "predictive_source": SHIFT_SUMMARY.name,
                "shap_source": SHIFT_SUMMARY.name,
            }
        )
    return pd.DataFrame(rows)


def draw_family_overlay(ax: plt.Axes, panel_df: pd.DataFrame, family: str) -> None:
    fam_points = panel_df[
        (panel_df["model"].map(MODEL_FAMILY) == family) & (panel_df["shap_jaccard"].notna())
    ][["predictive_retention", "shap_jaccard"]].dropna()
    if len(fam_points) < 2:
        return

    pts = fam_points.to_numpy(dtype=float)
    mean = pts.mean(axis=0)
    cov = np.cov(pts.T)
    # Degenerate cases are possible with very few points.
    if np.any(~np.isfinite(cov)):
        return
    vals, vecs = np.linalg.eigh(cov)
    vals = np.clip(vals, 1e-6, None)
    order = vals.argsort()[::-1]
    vals = vals[order]
    vecs = vecs[:, order]

    # ~1.5 std ellipse: subtle grouping hint, not inferential boundary.
    n_std = 1.5
    width, height = 2 * n_std * np.sqrt(vals)
    angle = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    style = FAMILY_STYLE[family]
    ell = Ellipse(
        xy=mean,
        width=width,
        height=height,
        angle=angle,
        facecolor=style["color"],
        edgecolor=style["color"],
        linewidth=0.95,
        alpha=style["alpha"],
        zorder=1,
    )
    ax.add_patch(ell)


def _draw_panel(ax: plt.Axes, panel_df: pd.DataFrame, title: str) -> None:
    # Background and grid for paper-like readability.
    ax.set_facecolor("white")
    ax.grid(True, color="#d9d9d9", linewidth=0.6, alpha=0.65)
    ax.set_title(title, fontsize=11, pad=7)

    # A dedicated strip for models with predictive data but no SHAP.
    strip_lo, strip_hi = -0.205, -0.085
    missing_y = -0.145
    ax.axhspan(strip_lo, strip_hi, color="#d9d9d9", alpha=0.9, zorder=0)
    ax.axhline(strip_hi, color="#8d8d8d", linewidth=1.1, zorder=1)
    ax.axhline(strip_lo, color="#8d8d8d", linewidth=1.1, zorder=1)

    # Small "favorable" region (upper-right) and fixed x=0 reference.
    main = panel_df[panel_df["shap_jaccard"].notna()].copy()
    if not main.empty:
        x_ref = float(main["predictive_retention"].quantile(0.80))
        y_ref = float(main["shap_jaccard"].quantile(0.80))
        x_max = float(main["predictive_retention"].max())
        y_max = float(main["shap_jaccard"].max())
        rect = Rectangle(
            (x_ref, y_ref),
            max(1e-6, x_max - x_ref),
            max(1e-6, y_max - y_ref),
            facecolor="#6FCE7A",
            edgecolor="none",
            alpha=0.46,
            zorder=0.5,
        )
        ax.add_patch(rect)
        ax.axhline(y_ref, linestyle="--", color="#9a9a9a", linewidth=0.75, alpha=0.8, zorder=1)

    # Family overlays (main area only).
    draw_family_overlay(ax, panel_df, "tree")
    draw_family_overlay(ax, panel_df, "neural")

    panel_df = panel_df.copy()
    panel_df["has_shap"] = panel_df["shap_jaccard"].notna()

    # Models with SHAP.
    for _, r in panel_df[panel_df["has_shap"]].iterrows():
        m = r["model"]
        style = MODEL_STYLE.get(m, {"color": "#333333", "marker": "o"})
        x = float(r["predictive_retention"])
        y = float(r["shap_jaccard"])
        xerr = float(r["predictive_retention_std"]) if pd.notna(r["predictive_retention_std"]) else 0.0
        yerr = float(r["shap_jaccard_std"]) if pd.notna(r["shap_jaccard_std"]) else 0.0
        ax.errorbar(
            x,
            y,
            xerr=xerr,
            yerr=yerr,
            fmt="none",
            ecolor=style["color"],
            elinewidth=0.9,
            capsize=2.0,
            alpha=0.65,
            zorder=2,
        )
        ax.scatter(
            x,
            y,
            s=58,
            color=style["color"],
            marker=style["marker"],
            edgecolor="white",
            linewidth=0.75,
            zorder=3,
        )

    # Models without SHAP: keep them visible and explicit.
    miss = panel_df[~panel_df["has_shap"]].copy()
    if not miss.empty:
        # small deterministic y jitter to avoid text overlap
        offsets = np.linspace(-0.01, 0.01, num=len(miss))
        for i, (_, r) in enumerate(miss.iterrows()):
            m = r["model"]
            style = MODEL_STYLE.get(m, {"color": "#333333", "marker": "o"})
            x = float(r["predictive_retention"])
            y = missing_y + float(offsets[i])
            ax.scatter(
                x,
                y,
                s=54,
                facecolor="none",
                edgecolor=style["color"],
                marker=style["marker"],
                linewidth=1.1,
                zorder=3,
            )

    ax.set_ylim(-0.24, 1.02)
    # shared y-label will be added at figure level
    ax.set_xlabel("Predictive Retention (perturbed - clean)")
    ax.axvline(0.0, linestyle="--", color="#6f6f6f", linewidth=0.9, alpha=0.9, zorder=1.2)
    ax.axhline(0.0, color="#a7a7a7", linewidth=0.70, alpha=0.75)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DATA_DIR.mkdir(parents=True, exist_ok=True)

    panel_a = build_panel_ab("missingness")
    panel_b = build_panel_ab("high_cardinality")
    panel_c = build_panel_c()

    plot_df = pd.concat([panel_a, panel_b, panel_c], ignore_index=True)
    plot_df.to_csv(DATA_PATH, index=False)

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 5.2))
    fig.subplots_adjust(left=0.07, right=0.995, top=0.83, bottom=0.255, wspace=0.17)

    _draw_panel(axes[0], panel_a, "A. Missingness")
    _draw_panel(axes[1], panel_b, "B. High-cardinality Stress")
    _draw_panel(axes[2], panel_c, "C. Covariate Shift")

    # Panel-wise x-limits reduce dead space and keep x=0 visible in all panels.
    for ax, dfp in zip(axes, [panel_a, panel_b, panel_c]):
        x_min = float(dfp["predictive_retention"].min())
        x_max = float(dfp["predictive_retention"].max())
        pad = max(0.02, 0.12 * (x_max - x_min if x_max > x_min else 1.0))
        left = x_min - pad
        right = x_max + pad
        ax.set_xlim(left, max(right, 0.02))

    # Shared Y label for all panels.
    fig.supylabel("SHAP Top-k Stability (Jaccard)", x=0.015, fontsize=10)

    # Bottom legend block (figure-level) for model/logo identification.
    model_handles = [
        Line2D(
            [0],
            [0],
            marker=MODEL_STYLE[m]["marker"],
            color="none",
            markerfacecolor=MODEL_STYLE[m]["color"],
            markeredgecolor="white",
            markeredgewidth=0.8,
            markersize=7.2,
            label=m,
        )
        for m in MODEL_STYLE
    ]
    fig.legend(
        handles=model_handles,
        loc="lower center",
        ncol=7,
        fontsize=8.0,
        frameon=False,
        bbox_to_anchor=(0.5, 0.105),
        handletextpad=0.25,
        columnspacing=0.95,
    )

    # Concise semantics legend under model identities.
    fig.text(
        0.5,
        0.063,
        "Square=tree, circle=neural, diamond=pretrained | "
        "dashed vertical line: x=0 (no change) | "
        "lower gray band: predictive-only (SHAP unavailable)",
        ha="center",
        va="center",
        fontsize=8.2,
        color="#4f4f4f",
    )
    fig.text(
        0.5,
        0.042,
        "Green region per panel: top-right quadrant starting at the 80th percentile of predictive retention and SHAP Jaccard.",
        ha="center",
        va="center",
        fontsize=7.9,
        color="#4f4f4f",
    )

    fig.suptitle(
        "Performance Retention vs Attribution Stability Under Perturbation",
        fontsize=13,
        y=0.965,
    )

    fig.savefig(FIG_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved figure: {FIG_PATH}")
    print(f"Saved plot data: {DATA_PATH}")


if __name__ == "__main__":
    main()


"""
Draft caption (paper-ready):
Figure X: Relationship between predictive degradation and SHAP top-k stability (Jaccard)
under three perturbation scenarios: (A) missingness, (B) high-cardinality stress, and
(C) covariate shift. Each point denotes a model-level mean over available datasets for the
given scenario. Predictive degradation is reported as clean score minus perturbed score
(larger values indicate stronger performance loss). Higher y-values indicate more stable core
feature-attribution rankings. Models with predictive results but unavailable SHAP stability
(e.g., APT/TabPFN in selected scenarios) are shown explicitly in a dedicated predictive-only
strip to avoid silent omission.
"""
