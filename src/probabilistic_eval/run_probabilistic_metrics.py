"""
Evaluate probabilistic classification metrics for trained models on 3 datasets
- Datasets: Titanic (binary), Dry Bean (multiclass), Telco Churn (binary)
- Models: RandomForest, LightGBM, CatBoost, MLP, XGBoost
- Assumes models are trained and saved via Optuna in results/hpo/<dataset>/<model>/best_model.pkl
- Assumes best hyperparameters are saved in best_params.json per model folder

Steps:
1. Load test sets for each dataset from original preprocessed CSVs in data/
2. Reload the best trained model from best_model.pkl
3. Predict class probabilities on the test set
4. Compute classification metrics:
   - Log Loss (cross-entropy)
   - Brier Score
   - Accuracy and F1 score for reference
   - Calibration Curve (save plot)
5. Save predictions (probs and labels) as .npy or .csv in results/probabilistic_eval/<dataset>/<model>/
6. Save computed metrics as metrics.json or metrics.csv in same folder
7. Save calibration curve as calibration_plot.png

Save everything under:
results/probabilistic_eval/<dataset>/<model>/




Extend the existing probabilistic evaluation pipeline to include 3 new analyses:

1. Compute Expected Calibration Error (ECE) for each model and dataset.
2. Plot confidence distribution histograms for correct vs incorrect predictions (i.e., predicted probability of the true class).
3. For the multiclass dataset (DryBean), compute and save per-class log loss or Brier score.

Requirements:
- Integrate all additions into the existing script.
- Save all results under `results/probabilistic_eval/{dataset}/{model}/` using the following file names:
  - `ece_score.json` — contains ECE value.
  - `confidence_hist.png` — confidence distribution plots (correct vs incorrect).
  - `per_class_metrics.json` — for DryBean only; log loss or Brier score per class.
- Load predictions from `predictions.csv` already saved per model.
- Ensure each script works without needing model retraining — no access to `.pkl` models is required.
- Use sklearn or scipy metrics only; avoid obscure libraries.
- Add comments for each major block and keep functions modular.

Example:  
For DryBean and CatBoost, generate:
- `results/probabilistic_eval/drybean/catboost/ece_score.json`
- `results/probabilistic_eval/drybean/catboost/confidence_hist.png`
- `results/probabilistic_eval/drybean/catboost/per_class_metrics.json`

"""

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd

from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, f1_score, log_loss, brier_score_loss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Ensure Matplotlib has a writable cache to avoid warnings and slowdowns
os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt  # noqa: E402

RANDOM_STATE = 42


@dataclass
class DatasetConfig:
    name: str
    path: Path
    target: str
    is_multiclass: bool


DATASETS: Dict[str, DatasetConfig] = {
    "titanic": DatasetConfig(
        "titanic",
        Path("data/titanic_binary_classification.csv"),
        "Survived",
        False,
    ),
    "drybean": DatasetConfig(
        "drybean",
        Path("data/drybean_multiclass_classification.csv"),
        "Class",
        True,
    ),
    "telco": DatasetConfig(
        "telco",
        Path("data/telco_churn_classification.csv"),
        "Churn",
        False,
    ),
}

MODELS = {"rf", "lgbm", "catboost", "xgboost", "mlp", "tabnet"}


def load_dataset(cfg: DatasetConfig) -> Tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(cfg.path)
    df = df.dropna().reset_index(drop=True)
    if not np.issubdtype(df[cfg.target].dtype, np.number):
        df[cfg.target] = df[cfg.target].astype(str)
    X = df.drop(columns=[cfg.target])
    y = df[cfg.target]
    return X, y


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    cat_cols = [c for c in X.columns if X[c].dtype == object]
    num_cols = [c for c in X.columns if c not in cat_cols]
    transformers = []
    if num_cols:
        transformers.append(("num", StandardScaler(), num_cols))
    if cat_cols:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols))
    return ColumnTransformer(transformers)


def _prob_columns(classes: np.ndarray) -> List[str]:
    return [f"prob_{str(cls).replace(' ', '_')}" for cls in classes]


def _brier_multiclass(y_true: np.ndarray, probs: np.ndarray, classes: np.ndarray) -> float:
    one_hot = (y_true[:, None] == classes[None, :]).astype(float)
    return float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))


def _normalized_label(label: object) -> str:
    return str(label).replace(" ", "_")


def _infer_classes_from_predictions(preds_df: pd.DataFrame) -> List[str]:
    prob_cols = [c for c in preds_df.columns if c.startswith("prob_")]
    suffixes = [c[len("prob_") :] for c in prob_cols]
    unique_labels = preds_df["y_true"].unique().tolist()
    normalized_map = {_normalized_label(lbl): str(lbl) for lbl in unique_labels}
    return [normalized_map.get(suffix, suffix) for suffix in suffixes]


def _extract_probs_and_labels(
    preds_df: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # Load probabilities and align class labels to prediction columns.
    prob_cols = [c for c in preds_df.columns if c.startswith("prob_")]
    probs = preds_df[prob_cols].to_numpy()
    classes = np.array([str(c) for c in _infer_classes_from_predictions(preds_df)])
    y_true = preds_df["y_true"].to_numpy()
    y_pred = preds_df["y_pred"].to_numpy()
    return y_true, y_pred, probs, classes


def compute_ece(confidences: np.ndarray, correct: np.ndarray, n_bins: int = 10) -> float:
    # Expected Calibration Error (ECE) using equal-width bins.
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        bin_mask = (confidences > bins[i]) & (confidences <= bins[i + 1])
        if not np.any(bin_mask):
            continue
        bin_acc = float(np.mean(correct[bin_mask]))
        bin_conf = float(np.mean(confidences[bin_mask]))
        ece += abs(bin_acc - bin_conf) * (np.sum(bin_mask) / len(confidences))
    return float(ece)


def save_ece(out_dir: Path, ece_value: float) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "ece_score.json", "w") as f:
        json.dump({"ece": ece_value}, f, indent=2)


def plot_confidence_histogram(
    out_dir: Path,
    true_class_probs: np.ndarray,
    correct: np.ndarray,
) -> None:
    # Plot confidence distribution for correct vs incorrect predictions.
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 4))
    plt.hist(
        true_class_probs[correct],
        bins=20,
        alpha=0.7,
        label="Correct",
        color="#2ca02c",
    )
    plt.hist(
        true_class_probs[~correct],
        bins=20,
        alpha=0.7,
        label="Incorrect",
        color="#d62728",
    )
    plt.title("Confidence Distribution (True Class Prob)")
    plt.xlabel("Predicted probability of true class")
    plt.ylabel("Count")
    plt.legend(fontsize=8, loc="best")
    plt.tight_layout()
    plt.savefig(out_dir / "confidence_hist.png", dpi=150)
    plt.close()


def save_per_class_metrics(
    out_dir: Path,
    y_true_str: np.ndarray,
    probs: np.ndarray,
    classes: np.ndarray,
) -> None:
    # Compute per-class Brier scores for multiclass datasets.
    per_class = {}
    for idx, cls in enumerate(classes):
        y_true_bin = (y_true_str == cls).astype(int)
        per_class[str(cls)] = float(np.mean((probs[:, idx] - y_true_bin) ** 2))
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "per_class_metrics.json", "w") as f:
        json.dump(per_class, f, indent=2)


def postprocess_predictions(out_dir: Path, is_multiclass: bool) -> None:
    preds_path = out_dir / "predictions.csv"
    if not preds_path.exists():
        raise FileNotFoundError(f"Missing predictions.csv at {preds_path}")
    preds_df = pd.read_csv(preds_path)
    y_true, y_pred, probs, classes = _extract_probs_and_labels(preds_df)
    y_true_str = np.array([str(v) for v in y_true])

    # Compute ECE using confidence from max probability and correctness.
    confidences = np.max(probs, axis=1)
    correct = (y_pred == y_true)
    ece_value = compute_ece(confidences, correct)
    save_ece(out_dir, ece_value)

    # Plot confidence distribution for correct vs incorrect predictions.
    # Map true class labels to probability indices once for speed and stability.
    class_to_index = {cls: idx for idx, cls in enumerate(classes)}
    true_class_probs = np.array(
        [probs[i, class_to_index[y_true_str[i]]] for i in range(len(y_true_str))]
    )
    plot_confidence_histogram(out_dir, true_class_probs, correct)

    # Save per-class metrics for multiclass datasets only.
    if is_multiclass:
        save_per_class_metrics(out_dir, y_true_str, probs, classes)


def evaluate_model(
    cfg: DatasetConfig,
    model_name: str,
    out_dir: Path,
    model_path: Path,
    test_size: float = 0.2,
) -> None:
    X, y = load_dataset(cfg)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    preprocessor = build_preprocessor(X_train)
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)
    if hasattr(X_train_proc, "toarray"):
        X_test_proc = X_test_proc.toarray()

    model = joblib.load(model_path)
    if not hasattr(model, "predict_proba"):
        raise RuntimeError(f"Model {model_name} does not support predict_proba")

    probs = model.predict_proba(X_test_proc)
    classes = getattr(model, "classes_", None)
    if classes is None:
        raise RuntimeError(f"Model {model_name} missing classes_ for probability alignment")

    y_true = np.asarray(y_test)
    y_pred = classes[np.argmax(probs, axis=1)]

    metrics: Dict[str, float] = {
        "log_loss": float(log_loss(y_true, probs, labels=classes)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(
            f1_score(
                y_true,
                y_pred,
                average="macro" if cfg.is_multiclass else "binary",
                pos_label=classes[1] if not cfg.is_multiclass else None,
            )
        ),
    }
    if cfg.is_multiclass:
        metrics["brier_score"] = _brier_multiclass(y_true, probs, classes)
    else:
        pos_label = classes[1]
        pos_idx = int(np.where(classes == pos_label)[0][0])
        y_true_bin = (y_true == pos_label).astype(int)
        metrics["brier_score"] = float(brier_score_loss(y_true_bin, probs[:, pos_idx]))

    out_dir.mkdir(parents=True, exist_ok=True)
    preds_df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred})
    for col_name, col_vals in zip(_prob_columns(classes), probs.T):
        preds_df[col_name] = col_vals
    preds_df.to_csv(out_dir / "predictions.csv", index=False)

    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    plt.figure(figsize=(6, 4))
    if cfg.is_multiclass:
        for idx, cls in enumerate(classes):
            y_true_bin = (y_true == cls).astype(int)
            frac_pos, mean_pred = calibration_curve(y_true_bin, probs[:, idx], n_bins=10)
            plt.plot(mean_pred, frac_pos, marker="o", label=str(cls))
        plt.legend(fontsize=8, loc="best")
    else:
        y_true_bin = (y_true == classes[1]).astype(int)
        frac_pos, mean_pred = calibration_curve(y_true_bin, probs[:, 1], n_bins=10)
        plt.plot(mean_pred, frac_pos, marker="o", label="Positive class")
        plt.legend(fontsize=8, loc="best")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    plt.title(f"Calibration: {cfg.name} / {model_name}")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives")
    plt.tight_layout()
    plt.savefig(out_dir / "calibration_plot.png", dpi=150)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run probabilistic evaluation for trained models.")
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS.keys()))
    parser.add_argument("--models", nargs="+", default=sorted(MODELS))
    parser.add_argument("--postprocess-only", action="store_true")
    args = parser.parse_args()

    datasets = [d for d in args.datasets if d in DATASETS]
    models = [m for m in args.models if m in MODELS]

    for dataset_name in datasets:
        cfg = DATASETS[dataset_name]
        for model_name in models:
            out_dir = Path("results/probabilistic_eval") / dataset_name / model_name
            try:
                if not args.postprocess_only:
                    model_dir = Path("results/hpo") / dataset_name / model_name
                    model_path = model_dir / "best_model.pkl"
                    if not model_path.exists():
                        print(f"⚠️ Missing model for {dataset_name}/{model_name}, skipping.")
                        continue
                    evaluate_model(cfg, model_name, out_dir, model_path)
                    print(f"✔ Saved probabilistic eval for {dataset_name} / {model_name} to {out_dir}")
                postprocess_predictions(out_dir, cfg.is_multiclass)
                print(f"✔ Saved postprocess outputs for {dataset_name} / {model_name} to {out_dir}")
            except Exception as exc:
                out_dir.mkdir(parents=True, exist_ok=True)
                with open(out_dir / "error.txt", "w") as f:
                    f.write(f"{type(exc).__name__}: {exc}\n")
                print(f"⚠️ Failed probabilistic eval for {dataset_name} / {model_name}: {exc}")


if __name__ == "__main__":
    main()
