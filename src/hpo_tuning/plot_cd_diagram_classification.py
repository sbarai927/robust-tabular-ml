#!/usr/bin/env python3
"""Build a publication-style Critical Difference (CD) diagram for classification F1.

Outputs:
- results/paper_figures/figure9_cd_diagram_classification.png
- results/paper_figure_delta_summary/figure9_cd_diagram_classification_ranks.csv
- results/paper_figure_delta_summary/figure9_cd_diagram_classification_nemenyi.csv
- results/paper_figure_delta_summary/figure9_cd_diagram_classification_dataset_ranks.csv
- results/paper_figure_delta_summary/figure9_cd_diagram_classification_caption_note.txt

Method:
- Friedman test over per-dataset model ranks.
- Nemenyi post-hoc via Studentized range distribution.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, studentized_range


METRICS = Path("results/hpo/metrics_aggregate.csv")
OUT_FIG = Path("results/paper_figures/figure9_cd_diagram_classification.png")
OUT_DIR = Path("results/paper_figure_delta_summary")
OUT_RANKS = OUT_DIR / "figure9_cd_diagram_classification_ranks.csv"
OUT_NEMENYI = OUT_DIR / "figure9_cd_diagram_classification_nemenyi.csv"
OUT_DATASET_RANKS = OUT_DIR / "figure9_cd_diagram_classification_dataset_ranks.csv"
OUT_CAPTION = OUT_DIR / "figure9_cd_diagram_classification_caption_note.txt"

CLASSIFICATION_DATASETS = ["drybean", "mnist", "telco", "titanic"]
ALPHA = 0.05


@dataclass
class CDStats:
    ranks_matrix: pd.DataFrame
    mean_ranks: pd.Series
    nemenyi_p: pd.DataFrame
    friedman_stat: float
    friedman_p: float
    cd_value: float


def _maximal_nonsig_intervals(models_sorted: Sequence[str], pvals: pd.DataFrame, alpha: float) -> List[Tuple[int, int]]:
    """Find maximal contiguous intervals where all pairwise comparisons are non-significant."""
    k = len(models_sorted)
    intervals: List[Tuple[int, int]] = []

    for i in range(k):
        for j in range(i + 1, k):
            block = models_sorted[i : j + 1]
            ok = True
            for a_idx in range(len(block)):
                for b_idx in range(a_idx + 1, len(block)):
                    if float(pvals.loc[block[a_idx], block[b_idx]]) < alpha:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                intervals.append((i, j))

    # Keep only maximal intervals (not strict subsets of others).
    maximal: List[Tuple[int, int]] = []
    for i0, j0 in intervals:
        is_subset = False
        for i1, j1 in intervals:
            if (i1 <= i0 and j1 >= j0) and (i1, j1) != (i0, j0):
                is_subset = True
                break
        if not is_subset:
            maximal.append((i0, j0))

    # Stable ordering: long bars first, then by start index.
    maximal = sorted(set(maximal), key=lambda ij: (-(ij[1] - ij[0]), ij[0]))
    return maximal


def _compute_stats(metrics_path: Path) -> CDStats:
    df = pd.read_csv(metrics_path)
    df = df[df["dataset"].isin(CLASSIFICATION_DATASETS)].copy()
    df["F1"] = pd.to_numeric(df["F1"], errors="coerce")
    df["model"] = df["model"].astype(str).str.strip().str.lower()

    perf = df.pivot_table(index="dataset", columns="model", values="F1", aggfunc="first")
    perf = perf.dropna(axis=1, how="any")
    perf = perf.dropna(axis=0, how="any")

    # Higher F1 is better -> lower rank value is better.
    ranks = perf.rank(axis=1, ascending=False, method="average")
    mean_ranks = ranks.mean(axis=0).sort_values(ascending=True)

    stat, p = friedmanchisquare(*[ranks[c].values for c in ranks.columns])

    k = len(ranks.columns)
    n = len(ranks.index)
    se = np.sqrt(k * (k + 1) / (6.0 * n))

    # Nemenyi pairwise p-values using Studentized range (infinite df).
    # For Nemenyi, the Studentized-range argument uses sqrt(2) scaling.
    pvals = pd.DataFrame(np.eye(k), index=ranks.columns, columns=ranks.columns, dtype=float)
    for i, a in enumerate(ranks.columns):
        for j, b in enumerate(ranks.columns):
            if i >= j:
                continue
            q_stat = abs(float(mean_ranks[a] - mean_ranks[b])) / se
            p_ij = float(studentized_range.sf(q_stat * np.sqrt(2.0), k, np.inf))
            pvals.loc[a, b] = p_ij
            pvals.loc[b, a] = p_ij

    # Demsar-style CD: q_alpha from Studentized range divided by sqrt(2).
    q_crit = float(studentized_range.isf(ALPHA, k, np.inf) / np.sqrt(2.0))
    cd = q_crit * se

    return CDStats(
        ranks_matrix=ranks,
        mean_ranks=mean_ranks,
        nemenyi_p=pvals,
        friedman_stat=float(stat),
        friedman_p=float(p),
        cd_value=float(cd),
    )


def _plot_cd(stats: CDStats, out_path: Path) -> None:
    models_sorted = list(stats.mean_ranks.index)
    mean_ranks = stats.mean_ranks
    k = len(models_sorted)

    fig, ax = plt.subplots(figsize=(12.0, 6.3), dpi=300)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # Axis at top.
    y_axis = 0.87
    ax.plot([1, k], [y_axis, y_axis], color="#222222", lw=1.1)
    for x in range(1, k + 1):
        ax.plot([x, x], [y_axis, y_axis + 0.012], color="#222222", lw=1.0)
        ax.text(x, y_axis + 0.023, str(x), ha="center", va="bottom", fontsize=10.5)

    # CD bracket near top-left.
    cd_start = 1.0
    cd_end = min(k, cd_start + stats.cd_value)
    y_cd = 0.94
    ax.plot([cd_start, cd_end], [y_cd, y_cd], color="#222222", lw=1.1)
    ax.plot([cd_start, cd_start], [y_cd - 0.008, y_cd + 0.008], color="#222222", lw=1.1)
    ax.plot([cd_end, cd_end], [y_cd - 0.008, y_cd + 0.008], color="#222222", lw=1.1)
    ax.text((cd_start + cd_end) / 2, y_cd + 0.014, "CD", ha="center", va="bottom", fontsize=10.5)

    # Non-significant groups as horizontal bars.
    intervals = _maximal_nonsig_intervals(models_sorted, stats.nemenyi_p, ALPHA)
    y_grp = 0.82
    for i, j in intervals:
        x1 = float(mean_ranks[models_sorted[i]])
        x2 = float(mean_ranks[models_sorted[j]])
        ax.plot([x1, x2], [y_grp, y_grp], color="black", lw=2.2, solid_capstyle="butt")
        y_grp -= 0.03

    # Labels and connector layout.
    left_models = models_sorted[: (k + 1) // 2]
    right_models = models_sorted[(k + 1) // 2 :]

    y_left = np.linspace(0.58, 0.18, len(left_models))
    y_right = np.linspace(0.58, 0.18, len(right_models))
    x_left_label = 0.58
    x_right_label = k + 0.42

    for m, y in zip(left_models, y_left):
        x = float(mean_ranks[m])
        ax.plot([x, x], [y_axis, y], color="#2a2a2a", lw=1.0)
        ax.plot([x, x_left_label], [y, y], color="#2a2a2a", lw=1.0)
        ax.text(
            x_left_label - 0.02,
            y,
            f"{m} ({x:.2f})",
            ha="right",
            va="center",
            fontsize=10.3,
            color="#1f77b4" if m in left_models[:3] else "#222222",
            fontweight="bold" if m in left_models[:3] else "normal",
        )

    for m, y in zip(right_models, y_right):
        x = float(mean_ranks[m])
        ax.plot([x, x], [y_axis, y], color="#2a2a2a", lw=1.0)
        ax.plot([x, x_right_label], [y, y], color="#2a2a2a", lw=1.0)
        ax.text(x_right_label + 0.02, y, f"{m} ({x:.2f})", ha="left", va="center", fontsize=10.3, color="#222222")

    ax.set_xlim(0.45, k + 0.55)
    ax.set_ylim(0.08, 0.99)
    ax.axis("off")

    ax.set_title("Critical Difference Diagram (Classification, F1)", fontsize=18, pad=8)
    caption = (
        "Average rank across classification datasets: drybean, mnist, telco, titanic.\n"
        f"Lower rank is better. Horizontal bars connect models with no significant difference "
        f"(Nemenyi, $\\alpha$={ALPHA:.2f}). Friedman: statistic={stats.friedman_stat:.3f}, p={stats.friedman_p:.3f}."
    )
    fig.text(0.5, 0.04, caption, ha="center", va="bottom", fontsize=11.2)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    stats = _compute_stats(METRICS)

    ranks_out = stats.mean_ranks.rename("mean_rank").reset_index().rename(columns={"index": "model"})
    ranks_out.to_csv(OUT_RANKS, index=False)
    stats.nemenyi_p.to_csv(OUT_NEMENYI, index=True)
    stats.ranks_matrix.to_csv(OUT_DATASET_RANKS, index=True)

    caption_note = (
        "CD diagram built from per-dataset F1 ranks on classification datasets "
        "(drybean, mnist, telco, titanic). Friedman omnibus test followed by "
        "Nemenyi post-hoc; horizontal bars denote groups that are not significantly "
        f"different at alpha={ALPHA:.2f}. Note: only four datasets are available, so "
        "post-hoc significance should be interpreted cautiously."
    )
    OUT_CAPTION.write_text(caption_note + "\n", encoding="utf-8")

    _plot_cd(stats, OUT_FIG)

    print("Files used:")
    print(f"- {METRICS}")
    print("Saved:")
    print(f"- {OUT_FIG}")
    print(f"- {OUT_RANKS}")
    print(f"- {OUT_NEMENYI}")
    print(f"- {OUT_DATASET_RANKS}")
    print(f"- {OUT_CAPTION}")


if __name__ == "__main__":
    main()
