#!/usr/bin/env python3
"""Create a slide-ready flow diagram for the experimental pipeline."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


OUT_PATH = Path("results/paper_figures/slide4_experimental_design_flow.png")


def add_box(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    title: str,
    body: str,
    face: str,
    edge: str = "#2b2b2b",
    title_fontsize: float = 14,
    body_fontsize: float = 11.3,
    center_text: bool = False,
) -> None:
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        linewidth=1.4,
        edgecolor=edge,
        facecolor=face,
        zorder=2,
    )
    ax.add_patch(patch)
    title_x = x + (0.5 * w if center_text else 0.02 * w)
    body_x = x + (0.5 * w if center_text else 0.02 * w)
    ha = "center" if center_text else "left"

    ax.text(
        title_x,
        y + h - 0.24 * h,
        title,
        fontsize=title_fontsize,
        fontweight="bold",
        va="center",
        ha=ha,
        color="#111111",
        zorder=3,
    )
    ax.text(
        body_x,
        y + h - 0.52 * h,
        body,
        fontsize=body_fontsize,
        va="top",
        ha=ha,
        color="#1f1f1f",
        linespacing=1.3,
        zorder=3,
    )


def add_arrow(ax: plt.Axes, x0: float, y0: float, x1: float, y1: float) -> None:
    arrow = FancyArrowPatch(
        (x0, y0),
        (x1, y1),
        arrowstyle="-|>",
        mutation_scale=14,
        linewidth=1.7,
        color="#3b3b3b",
        zorder=1,
    )
    ax.add_patch(arrow)


def main() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(16, 9), dpi=180)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(
        0.04,
        0.94,
        "Experimental Pipeline",
        fontsize=25,
        fontweight="bold",
        ha="left",
        va="center",
        color="#101010",
    )

    ax.text(
        0.04,
        0.89,
        "Fair HPO with Optuna and a common evaluation protocol across models and datasets",
        fontsize=12.5,
        color="#303030",
        ha="left",
        va="center",
    )

    # Main pipeline boxes.
    b_w, b_h = 0.26, 0.36
    y = 0.43
    x1, x2, x3 = 0.05, 0.37, 0.69

    add_box(
        ax,
        x1,
        y,
        b_w,
        b_h,
        "1) Clean Benchmark",
        "Post-HPO evaluation on clean test data\n\n• Predictive performance\n• Training time\n• Model size",
        face="#e9f2ff",
    )
    add_box(
        ax,
        x2,
        y,
        b_w,
        b_h,
        "2) Robustness Tests",
        "Stress scenarios on the same splits\n\n• Injected missingness\n• Synthetic high-cardinality features",
        face="#eef8ee",
    )
    add_box(
        ax,
        x3,
        y,
        b_w,
        b_h,
        "3) Covariate Shift\nand Stability",
        "Shifted test-set evaluation\n\n• Clean vs shifted performance\n• SHAP top-k stability",
        face="#fff3e8",
        title_fontsize=13.2,
        body_fontsize=11.1,
    )

    add_arrow(ax, x1 + b_w, y + 0.5 * b_h, x2 - 0.012, y + 0.5 * b_h)
    add_arrow(ax, x2 + b_w, y + 0.5 * b_h, x3 - 0.012, y + 0.5 * b_h)

    # Bottom synthesis block.
    s_x, s_y, s_w, s_h = 0.16, 0.13, 0.68, 0.2
    add_box(
        ax,
        s_x,
        s_y,
        s_w,
        s_h,
        "Cross-Experiment Analysis",
        "Compare tree-based, neural, and pretrained tabular models in terms of\n"
        "predictive performance, efficiency, and attribution stability.",
        face="#f6f6f6",
        edge="#444444",
        center_text=True,
    )

    add_arrow(ax, x2 + 0.5 * b_w, y - 0.01, s_x + 0.5 * s_w, s_y + s_h + 0.012)

    fig.savefig(OUT_PATH, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {OUT_PATH}")


if __name__ == "__main__":
    main()
