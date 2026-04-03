#!/usr/bin/env python3
"""Backfill per-trial score/state columns into existing HPO trials.csv files.

This script does not retrain models. It attempts to recover trial objective values from
Optuna studies (study.pkl) and writes:
- score
- state
- trial_number
into results/hpo/<dataset>/<model>/trials.csv.

If only baseline artifacts exist (e.g., tabpfn/apt), it uses metrics.csv best_value for
single-row trials.csv.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd


def is_lfs_pointer(path: Path) -> bool:
    if not path.exists() or path.stat().st_size > 512:
        return False
    try:
        head = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return False
    return head.startswith("version https://git-lfs.github.com/spec/v1")


def build_trial_table_from_study(study_obj) -> Optional[pd.DataFrame]:
    if not hasattr(study_obj, "trials"):
        return None
    rows: List[Dict[str, object]] = []
    for t in sorted(study_obj.trials, key=lambda x: x.number):
        row = dict(t.params)
        row["trial_number"] = t.number
        row["score"] = t.value if t.value is not None else np.nan
        row["state"] = t.state.name if hasattr(t.state, "name") else str(t.state)
        rows.append(row)
    return pd.DataFrame(rows)


def enable_pickle_compat() -> None:
    """Patch known cross-version pickle incompatibilities (numpy/optuna)."""
    # numpy RNG pickle compatibility
    import numpy.random._pickle as nrp

    class _DummyRandomState:
        def __setstate__(self, state):
            self._state = state

    class _DummyGenerator:
        def __setstate__(self, state):
            self._state = state

    class _DummyBitGenerator:
        def __setstate__(self, state):
            self._state = state

    nrp.__randomstate_ctor = lambda *args, **kwargs: _DummyRandomState()
    nrp.__generator_ctor = lambda *args, **kwargs: _DummyGenerator()
    nrp.__bit_generator_ctor = lambda *args, **kwargs: _DummyBitGenerator()

    # optuna TPE internal namedtuple compatibility
    from optuna.samplers._tpe import parzen_estimator as pe

    orig_cls = pe._ParzenEstimatorParameters

    class _CompatParzen(orig_cls):
        __slots__ = ()

        def __new__(cls, *args):
            n = len(orig_cls._fields)
            args = list(args)
            if len(args) > n:
                args = args[:n]
            while len(args) < n:
                # Last field in current optuna versions is categorical_distance_func.
                if len(args) == n - 1:
                    args.append({})
                else:
                    args.append(False)
            return super(_CompatParzen, cls).__new__(cls, *args)

    pe._ParzenEstimatorParameters = _CompatParzen


def backfill_one(model_dir: Path, overwrite: bool = True) -> str:
    trials_path = model_dir / "trials.csv"
    if not trials_path.exists():
        return "skip:no_trials_csv"

    try:
        trials_df = pd.read_csv(trials_path)
    except Exception as exc:
        return f"skip:bad_trials_csv:{exc}"

    # Skip unless overwrite requested when score already exists.
    if (not overwrite) and ("score" in trials_df.columns):
        return "skip:has_score"

    # 1) Try Optuna study.pkl
    study_pkl = model_dir / "study.pkl"
    if study_pkl.exists() and not is_lfs_pointer(study_pkl):
        try:
            enable_pickle_compat()
            study_obj = joblib.load(study_pkl)
            optuna_df = build_trial_table_from_study(study_obj)
            if optuna_df is not None and len(optuna_df) > 0:
                out = trials_df.copy()
                n = min(len(out), len(optuna_df))
                out.loc[: n - 1, "score"] = optuna_df.iloc[:n]["score"].values
                out.loc[: n - 1, "state"] = optuna_df.iloc[:n]["state"].values
                out.loc[: n - 1, "trial_number"] = optuna_df.iloc[:n]["trial_number"].values
                if len(out) > len(optuna_df):
                    out.loc[n:, ["score", "state", "trial_number"]] = np.nan
                out.to_csv(trials_path, index=False)
                return f"ok:optuna:{n}/{len(out)}"
        except Exception as exc:
            # fall through to baseline mode
            _ = exc

    model_name = model_dir.name.lower()
    is_baseline_model = model_name in {"tabpfn", "apt"}

    # 2) Baseline fallback (tabpfn/apt style): single row + best_value in metrics.csv
    metrics_path = model_dir / "metrics.csv"
    if metrics_path.exists() and (is_baseline_model or len(trials_df) == 1):
        try:
            metrics = pd.read_csv(metrics_path)
            if ("best_value" in metrics.columns) and (len(metrics) > 0):
                best_value = float(metrics.iloc[0]["best_value"])
                out = trials_df.copy()
                if len(out) >= 1:
                    out.loc[0, "score"] = best_value
                    out.loc[0, "state"] = "COMPLETE"
                    out.loc[0, "trial_number"] = 0
                    if len(out) > 1:
                        out.loc[1:, ["score", "state", "trial_number"]] = np.nan
                    out.to_csv(trials_path, index=False)
                    return "ok:baseline"
        except Exception as exc:
            return f"skip:metrics_error:{exc}"

    if study_pkl.exists() and is_lfs_pointer(study_pkl):
        return "skip:lfs_pointer"
    return "skip:no_source"


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill score/state/trial_number into HPO trials.csv")
    parser.add_argument("--root", type=Path, default=Path("results/hpo"))
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing score column values")
    args = parser.parse_args()

    root = args.root
    model_dirs = sorted([p.parent for p in root.glob("*/*/trials.csv")])

    stats: Dict[str, int] = {}
    detailed: List[tuple[str, str]] = []

    for d in model_dirs:
        status = backfill_one(d, overwrite=args.overwrite)
        stats[status] = stats.get(status, 0) + 1
        detailed.append((str(d), status))

    print(f"Processed {len(model_dirs)} model folders under {root}")
    for k, v in sorted(stats.items()):
        print(f"- {k}: {v}")

    skipped = [(p, s) for p, s in detailed if s.startswith("skip")]
    if skipped:
        print("\nSkipped details:")
        for p, s in skipped:
            print(f"  {p} -> {s}")


if __name__ == "__main__":
    main()
