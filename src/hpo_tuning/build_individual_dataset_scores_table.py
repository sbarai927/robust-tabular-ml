"""Build per-dataset HPO score tables for the paper.

This script uses only saved repository outputs. It does not retrain models.
Classification rows use final test F1 from results/hpo/metrics_aggregate.csv;
regression rows use final test R2 from the same file.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
METRICS_PATH = ROOT / "results" / "hpo" / "metrics_aggregate.csv"
OUT_DIR = ROOT / "results" / "paper_tables"

DATASETS = {
    "diamonds_v2": {"task": "Regression", "metric": "R2"},
    "diamonds_v3": {"task": "Regression", "metric": "R2"},
    "drybean": {"task": "Multiclass classification", "metric": "F1"},
    "mnist": {"task": "Multiclass classification", "metric": "F1"},
    "telco": {"task": "Binary classification", "metric": "F1"},
    "titanic": {"task": "Binary classification", "metric": "F1"},
}

MODEL_ORDER = [
    ("rf", "RF"),
    ("xgboost", "XGBoost"),
    ("lgbm", "LightGBM"),
    ("catboost", "CatBoost"),
    ("mlp", "MLP"),
    ("tabnet", "TabNet"),
    ("tabpfn", "TabPFN"),
    ("apt", "APT"),
]

MANUSCRIPT_SENTENCE = (
    "In addition to the aggregated ranks in Table III, "
    "Table~\\ref{tab:individual_dataset_scores} reports individual "
    "dataset-level scores, showing which datasets drive the average ranking."
)


def fmt_score(value: object, metric: str | None = None) -> str:
    if pd.isna(value):
        return "NA"
    if metric == "R2":
        return f"{float(value):.5f}"
    return f"{float(value):.3f}"


def markdown_table(df: pd.DataFrame) -> str:
    headers = list(df.columns)
    rows = [[str(v) for v in row] for row in df.to_numpy()]
    widths = [
        max(len(str(header)), *(len(row[i]) for row in rows)) for i, header in enumerate(headers)
    ]
    header_line = "| " + " | ".join(str(h).ljust(widths[i]) for i, h in enumerate(headers)) + " |"
    sep_line = "| " + " | ".join("-" * widths[i] for i in range(len(headers))) + " |"
    body = [
        "| " + " | ".join(row[i].ljust(widths[i]) for i in range(len(headers))) + " |"
        for row in rows
    ]
    return "\n".join([header_line, sep_line, *body]) + "\n"


def latex_escape(text: object) -> str:
    return str(text).replace("_", r"\_")


def latex_table(df: pd.DataFrame) -> str:
    latex_df = df.copy()
    latex_df = latex_df.map(latex_escape)
    body = latex_df.to_latex(
        index=False,
        escape=False,
        column_format="lllrrrrrrrr",
        na_rep="NA",
    )
    return "\n".join(
        [
            r"\begin{table*}[t]",
            r"\centering",
            r"\small",
            r"\caption{Individual dataset-level test scores after HPO. "
            r"Classification datasets report F1; regression datasets report $R^2$. "
            r"NA indicates that the corresponding model was not evaluated for that task.}",
            r"\label{tab:individual_dataset_scores}",
            body,
            r"\end{table*}",
            "",
        ]
    )


def build_table(metrics: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    metrics = metrics.copy()
    metrics["dataset"] = metrics["dataset"].astype(str)
    metrics["model"] = metrics["model"].astype(str)

    notes: list[str] = []
    rows: list[dict[str, object]] = []
    for dataset, meta in DATASETS.items():
        row: dict[str, object] = {
            "Dataset": dataset,
            "Task": meta["task"],
            "Metric": "F1" if meta["metric"] == "F1" else "R2",
        }
        ds_rows = metrics[metrics["dataset"] == dataset]
        if ds_rows.empty:
            notes.append(f"Missing all HPO metric rows for dataset={dataset}.")

        for model_key, model_label in MODEL_ORDER:
            model_rows = ds_rows[ds_rows["model"] == model_key]
            if model_rows.empty:
                row[model_label] = pd.NA
                notes.append(f"Missing score for dataset={dataset}, model={model_key}.")
                continue

            value = model_rows.iloc[0][meta["metric"]]
            if pd.isna(value):
                row[model_label] = pd.NA
                notes.append(
                    f"Metric {meta['metric']} unavailable for dataset={dataset}, model={model_key}."
                )
            else:
                row[model_label] = float(value)

        rows.append(row)

    table = pd.DataFrame(rows)
    return table, notes


def find_metric_inconsistencies(metrics: pd.DataFrame) -> list[str]:
    notes: list[str] = []

    cls_datasets = [d for d, meta in DATASETS.items() if meta["metric"] == "F1"]
    reg_datasets = [d for d, meta in DATASETS.items() if meta["metric"] == "R2"]

    cls_missing = metrics[
        metrics["dataset"].isin(cls_datasets) & metrics["F1"].isna()
    ][["dataset", "model"]]
    reg_missing = metrics[
        metrics["dataset"].isin(reg_datasets) & metrics["R2"].isna()
    ][["dataset", "model"]]

    if not cls_missing.empty:
        notes.append(
            "Classification rows with missing F1: "
            + ", ".join(f"{r.dataset}/{r.model}" for r in cls_missing.itertuples())
        )
    if not reg_missing.empty:
        notes.append(
            "Regression rows with missing R2: "
            + ", ".join(f"{r.dataset}/{r.model}" for r in reg_missing.itertuples())
        )

    cls_best_mismatch = metrics[
        metrics["dataset"].isin(cls_datasets)
        & metrics["F1"].notna()
        & metrics["best_value"].notna()
        & ((metrics["F1"] - metrics["best_value"]).abs() > 1e-8)
    ][["dataset", "model", "F1", "best_value"]]
    if not cls_best_mismatch.empty:
        notes.append(
            "Classification F1 differs from best_value for some rows; this table uses final test F1, "
            "while best_value is the saved Optuna objective/validation value."
        )

    return notes


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics = pd.read_csv(METRICS_PATH)

    raw_table, missing_notes = build_table(metrics)
    display_table = raw_table.copy().astype("object")
    for idx, row in display_table.iterrows():
        for _, model_label in MODEL_ORDER:
            display_table.at[idx, model_label] = fmt_score(row[model_label], row["Metric"])

    csv_path = OUT_DIR / "table6_individual_dataset_scores.csv"
    md_path = OUT_DIR / "table6_individual_dataset_scores.md"
    tex_path = OUT_DIR / "table6_individual_dataset_scores.tex"
    sentence_path = OUT_DIR / "table6_individual_dataset_scores_sentence.txt"
    notes_path = OUT_DIR / "table6_individual_dataset_scores_notes.txt"

    raw_table.to_csv(csv_path, index=False, na_rep="NA")
    md_path.write_text(markdown_table(display_table), encoding="utf-8")
    tex_path.write_text(latex_table(display_table), encoding="utf-8")
    sentence_path.write_text(MANUSCRIPT_SENTENCE + "\n", encoding="utf-8")

    consistency_notes = find_metric_inconsistencies(metrics)
    source_lines = [
        "Source files used:",
        f"- {METRICS_PATH.relative_to(ROOT)}",
        "- Per-model metrics.csv files under results/hpo/<dataset>/<model>/ were inspected for availability.",
        "",
        "Extraction policy:",
        "- Values were extracted from saved HPO result files only; no scores were recomputed.",
        "- Classification datasets use final test F1 from metrics_aggregate.csv.",
        "- Regression datasets use final test R2 from metrics_aggregate.csv.",
        "- best_value is not used as the table score because it stores the Optuna objective/validation value.",
        "",
        "Missing values:",
        "- TabPFN and APT are NA for diamonds_v2 and diamonds_v3 because those regression datasets do not have saved HPO metric rows for these classification/default baselines.",
    ]
    if missing_notes:
        source_lines.extend(["", "Detected missing dataset/model scores:", *[f"- {n}" for n in missing_notes]])
    if consistency_notes:
        source_lines.extend(["", "Metric consistency notes:", *[f"- {n}" for n in consistency_notes]])

    notes_path.write_text("\n".join(source_lines) + "\n", encoding="utf-8")

    print(f"Wrote {csv_path.relative_to(ROOT)}")
    print(f"Wrote {md_path.relative_to(ROOT)}")
    print(f"Wrote {tex_path.relative_to(ROOT)}")
    print(f"Wrote {sentence_path.relative_to(ROOT)}")
    print(f"Wrote {notes_path.relative_to(ROOT)}")
    print()
    print(markdown_table(display_table))


if __name__ == "__main__":
    main()
