#!/usr/bin/env python3
"""Upload drybean HPO trials to W&B for parallel-coordinates plots.

Each trial row is logged as one W&B run so the Runs table can be used to
create parallel-coordinates panels. No model training is performed.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable

import pandas as pd

try:
    import wandb
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "wandb is not installed. Install with `pip install wandb` in your env."
    ) from exc


DEFAULT_MODELS = ["catboost", "lgbm", "mlp", "rf", "tabnet", "xgboost"]
META_COLS = {"score", "state", "trial_number"}


def _is_scalar_finite(v) -> bool:
    if v is None:
        return False
    if isinstance(v, (int, float)):
        return not (isinstance(v, float) and (math.isnan(v) or math.isinf(v)))
    return True


def _is_numeric_scalar(v) -> bool:
    return isinstance(v, (int, float)) and not (
        isinstance(v, float) and (math.isnan(v) or math.isinf(v))
    )


def _iter_models(models_arg: str) -> Iterable[str]:
    return [m.strip().lower() for m in models_arg.split(",") if m.strip()]


def upload_model_trials(
    root: Path,
    dataset: str,
    model: str,
    entity: str,
    project: str,
    group: str,
    metric_name: str,
    allow_non_complete: bool,
) -> tuple[int, int]:
    trials_path = root / dataset / model / "trials.csv"
    if not trials_path.exists():
        print(f"[skip] missing file: {trials_path}")
        return 0, 0

    df = pd.read_csv(trials_path)
    if "score" not in df.columns:
        print(f"[skip] score column missing: {trials_path}")
        return 0, 0

    original_n = len(df)
    work = df.copy()
    if "state" in work.columns and not allow_non_complete:
        work = work[work["state"].fillna("").eq("COMPLETE")]
    work = work[work["score"].notna()].copy()

    if work.empty:
        print(f"[skip] no usable rows in {trials_path}")
        return original_n, 0

    logged = 0
    for ridx, row in work.iterrows():
        trial_number = (
            int(row["trial_number"])
            if "trial_number" in row and pd.notna(row["trial_number"])
            else int(ridx)
        )

        config = {
            "dataset": dataset,
            "model": model,
            "trial_number": trial_number,
            "source_trials_csv": str(trials_path),
        }
        for c in df.columns:
            if c in META_COLS:
                continue
            v = row.get(c)
            if _is_scalar_finite(v):
                config[f"param_{c}"] = v

        run = wandb.init(
            entity=entity,
            project=project,
            group=group,
            job_type="hpo_trials_upload",
            name=f"{dataset}_{model}_trial_{trial_number:03d}",
            reinit=True,
            config=config,
            tags=[dataset, model, "hpo", "parallel_coords"],
        )
        # Log objective + numeric params as history metrics so Parallel Coordinates
        # can pick them as axes even when config columns are not auto-detected.
        history_payload = {
            metric_name: float(row["score"]),
            "score": float(row["score"]),
            "trial_number": trial_number,
        }
        for c in df.columns:
            if c in META_COLS:
                continue
            v = row.get(c)
            if _is_numeric_scalar(v):
                history_payload[f"param_{c}"] = float(v)
        run.log(history_payload)
        run.summary[metric_name] = float(row["score"])
        run.summary["score"] = float(row["score"])
        if "state" in row and pd.notna(row["state"]):
            run.summary["state"] = str(row["state"])
        run.finish()
        logged += 1

    print(f"[ok] {dataset}/{model}: logged {logged}/{original_n} rows")
    return original_n, logged


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Upload drybean model trials.csv rows as W&B runs for parallel coordinates."
    )
    parser.add_argument("--entity", required=True, help="W&B entity/username")
    parser.add_argument("--project", required=True, help="W&B project name")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("results/hpo"),
        help="Root folder containing dataset/model/trials.csv structure.",
    )
    parser.add_argument("--dataset", default="drybean", help="Dataset folder name.")
    parser.add_argument(
        "--models",
        default=",".join(DEFAULT_MODELS),
        help="Comma-separated model list (default: 6 main models).",
    )
    parser.add_argument(
        "--group",
        default="drybean_hpo_parallel_coords",
        help="W&B run group for these uploaded trial-runs.",
    )
    parser.add_argument(
        "--metric-name",
        default="f1",
        help="Metric key to log in W&B (e.g., f1).",
    )
    parser.add_argument(
        "--allow-non-complete",
        action="store_true",
        help="Include non-COMPLETE rows if score exists.",
    )
    args = parser.parse_args()

    total_rows = 0
    total_logged = 0
    models = list(_iter_models(args.models))
    for model in models:
        n_rows, n_logged = upload_model_trials(
            root=args.root,
            dataset=args.dataset,
            model=model,
            entity=args.entity,
            project=args.project,
            group=args.group,
            metric_name=args.metric_name,
            allow_non_complete=args.allow_non_complete,
        )
        total_rows += n_rows
        total_logged += n_logged

    print("\nUpload complete")
    print(f"- dataset: {args.dataset}")
    print(f"- models: {models}")
    print(f"- rows seen: {total_rows}")
    print(f"- runs logged: {total_logged}")


if __name__ == "__main__":
    main()
