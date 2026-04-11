#!/usr/bin/env python3
"""Build compact model-wise stability summary table for the paper."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


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


def _mean(xs: list[float]) -> float | None:
    if not xs:
        return None
    return sum(xs) / len(xs)


def _fmt(v: float | None) -> str:
    if v is None:
        return "NA"
    return f"{v:.6f}"


def build_table(summary_csv: Path, out_csv: Path, out_note: Path) -> None:
    # Keep model ordering stable and readable.
    preferred_order = ["catboost", "lgbm", "rf", "xgboost", "mlp", "tabnet", "tabpfn", "apt"]

    acc_clean = defaultdict(list)
    acc_shift = defaultdict(list)
    f1_clean = defaultdict(list)
    f1_shift = defaultdict(list)
    shap_spearman = defaultdict(list)
    shap_jaccard = defaultdict(list)
    n_perf = defaultdict(int)
    n_shap = defaultdict(int)

    models_seen: set[str] = set()

    with summary_csv.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            model = row.get("model", "").strip()
            if not model:
                continue
            models_seen.add(model)

            a_c = _to_float(row.get("Accuracy_clean"))
            a_s = _to_float(row.get("Accuracy_shift"))
            f_c = _to_float(row.get("F1_clean"))
            f_s = _to_float(row.get("F1_shift"))
            s_sp = _to_float(row.get("shap_spearman"))
            s_jc = _to_float(row.get("shap_jaccard"))

            if a_c is not None:
                acc_clean[model].append(a_c)
            if a_s is not None:
                acc_shift[model].append(a_s)
            if f_c is not None:
                f1_clean[model].append(f_c)
            if f_s is not None:
                f1_shift[model].append(f_s)
            if (a_c is not None) or (a_s is not None) or (f_c is not None) or (f_s is not None):
                n_perf[model] += 1

            if s_sp is not None:
                shap_spearman[model].append(s_sp)
            if s_jc is not None:
                shap_jaccard[model].append(s_jc)
            if (s_sp is not None) or (s_jc is not None):
                n_shap[model] += 1

    order = [m for m in preferred_order if m in models_seen] + [m for m in sorted(models_seen) if m not in preferred_order]

    rows: list[dict[str, str]] = []
    for m in order:
        m_acc_clean = _mean(acc_clean[m])
        m_acc_shift = _mean(acc_shift[m])
        m_f1_clean = _mean(f1_clean[m])
        m_f1_shift = _mean(f1_shift[m])
        m_spear = _mean(shap_spearman[m])
        m_jacc = _mean(shap_jaccard[m])

        row = {
            "model": m,
            "mean_accuracy_clean": _fmt(m_acc_clean),
            "mean_accuracy_shift": _fmt(m_acc_shift),
            "mean_accuracy_delta_shift_minus_clean": _fmt(None if (m_acc_clean is None or m_acc_shift is None) else (m_acc_shift - m_acc_clean)),
            "mean_f1_clean": _fmt(m_f1_clean),
            "mean_f1_shift": _fmt(m_f1_shift),
            "mean_f1_delta_shift_minus_clean": _fmt(None if (m_f1_clean is None or m_f1_shift is None) else (m_f1_shift - m_f1_clean)),
            "mean_shap_spearman": _fmt(m_spear),
            "mean_shap_jaccard": _fmt(m_jacc),
            "n_datasets_with_perf": str(n_perf[m]),
            "n_datasets_with_shap": str(n_shap[m]),
        }
        rows.append(row)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    cols = [
        "model",
        "mean_accuracy_clean",
        "mean_accuracy_shift",
        "mean_accuracy_delta_shift_minus_clean",
        "mean_f1_clean",
        "mean_f1_shift",
        "mean_f1_delta_shift_minus_clean",
        "mean_shap_spearman",
        "mean_shap_jaccard",
        "n_datasets_with_perf",
        "n_datasets_with_shap",
    ]
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    note = (
        "Caption note: Stability summary is model-wise mean over datasets in "
        "results/tree_vs_deep_stability_analysis/summary_metrics.csv (drybean, telco, titanic). "
        "Delta columns are computed as shift - clean (negative means degradation under shift). "
        "SHAP Spearman/Jaccard means are averaged over available entries only; NA indicates unavailable values "
        "(e.g., models that failed in this environment or models without SHAP stability computation)."
    )
    out_note.write_text(note + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build compact stability model summary table.")
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=Path("results/tree_vs_deep_stability_analysis/summary_metrics.csv"),
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=Path("results/paper_tables/table5_stability_model_summary.csv"),
    )
    parser.add_argument(
        "--out-note",
        type=Path,
        default=Path("results/paper_tables/table5_stability_model_summary_note.txt"),
    )
    args = parser.parse_args()

    build_table(args.summary_csv, args.out_csv, args.out_note)
    print(f"Wrote: {args.out_csv}")
    print(f"Wrote: {args.out_note}")


if __name__ == "__main__":
    main()
