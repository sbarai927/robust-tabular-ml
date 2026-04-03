#!/usr/bin/env python3
"""Upload best-config insights rows (dataset x model) to W&B.

Intended for clean Fig3/Fig4 parallel coordinates where each panel has ~6 lines
(one per dataset) for a fixed model.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd

try:
    import wandb
except Exception as exc:  # pragma: no cover
    raise RuntimeError("wandb is not installed. Install with `pip install wandb`.") from exc


def is_numeric(v) -> bool:
    return isinstance(v, (int, float)) and not (
        isinstance(v, float) and (math.isnan(v) or math.isinf(v))
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload best-config insight rows to W&B")
    parser.add_argument("--entity", required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("results/paper_figure_delta_summary/figure3_4_best_config_insights_all_models.csv"),
    )
    parser.add_argument("--group", default="fig34_best_config_insights_v1")
    parser.add_argument(
        "--models",
        default="catboost,lgbm,rf,xgboost,mlp,tabnet",
        help="comma-separated filter list",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    df = df[df["model"].isin(models)].copy()

    logged = 0
    for _, r in df.iterrows():
        dataset = str(r["dataset"])
        model = str(r["model"])

        cfg = {
            "dataset": dataset,
            "dataset_order": int(r["dataset_order"]) if pd.notna(r["dataset_order"]) else None,
            "dataset_axis": int(r["dataset_order"]) if pd.notna(r["dataset_order"]) else None,
            "task": str(r["task"]),
            "model": model,
            "model_family": str(r["model_family"]),
            "source_csv": str(args.csv),
        }

        # Add only model-relevant params as config.param_*
        for c, v in r.items():
            if c in {
                "dataset",
                "dataset_order",
                "task",
                "model",
                "model_family",
                "primary_score",
                "dataset_rank",
                "gap_to_best_pct",
                "score_norm_within_dataset",
                "win_flag",
                "top2_flag",
                "model_mean_rank",
                "model_mean_gap_pct",
            }:
                continue
            if is_numeric(v):
                cfg[f"param_{c}"] = float(v)

        run = wandb.init(
            entity=args.entity,
            project=args.project,
            group=args.group,
            job_type="best_config_insights_upload",
            name=f"{dataset}_{model}_bestconfig",
            tags=["fig3", "fig4", "parallel_coords", model, dataset],
            config=cfg,
            reinit=True,
        )

        payload = {
            "primary_score": float(r["primary_score"]),
            "dataset_order": int(r["dataset_order"]) if pd.notna(r["dataset_order"]) else -1,
            "dataset_axis": int(r["dataset_order"]) if pd.notna(r["dataset_order"]) else -1,
            "dataset_rank": float(r["dataset_rank"]),
            "gap_to_best_pct": float(r["gap_to_best_pct"]),
            "score_norm_within_dataset": float(r["score_norm_within_dataset"]),
            "win_flag": int(r["win_flag"]),
            "top2_flag": int(r["top2_flag"]),
            "model_mean_rank": float(r["model_mean_rank"]),
            "model_mean_gap_pct": float(r["model_mean_gap_pct"]),
        }
        # Also log numeric params as history metrics so W&B Parallel Coordinates
        # can reliably expose them in "Axes", regardless of config parsing.
        for k, v in cfg.items():
            if k.startswith("param_") and is_numeric(v):
                payload[k] = float(v)
        run.log(payload)
        for k, v in payload.items():
            run.summary[k] = v
        run.finish()
        logged += 1

    print(f"Logged {logged} runs from {args.csv} to group={args.group}")


if __name__ == "__main__":
    main()
