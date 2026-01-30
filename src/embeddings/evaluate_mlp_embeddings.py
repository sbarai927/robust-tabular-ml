"""
Evaluate the usefulness of embeddings learned by MLPs for downstream classification tasks.

Datasets:
- diamonds_v2.csv (regression)
- telco_churn_classification.csv (binary classification)

Steps:

1. Load preprocessed datasets from data/ directory.
   - diamonds_v2: Predicting `total_sales_price`
   - telco_churn: Predicting `Churn` or optionally `Contract`

2. Load best-trained MLP models from:
   results/hpo/<dataset>/mlp/best_model.pkl

3. Extract penultimate layer embeddings (last hidden layer before output).
   - Use PyTorch forward hooks or define a modified forward function.
   - Save embeddings as .npy to:
     results/embeddings/<dataset>_mlp_embeddings.npy

4. Define downstream classification task:
   - For diamonds_v2: Bin `total_sales_price` into 3 quantiles (low/mid/high)
   - For telco: Use original `Churn` or another proxy categorical column (e.g. `Contract`)
   - Save new labels for use in downstream task

5. Train and evaluate the following classifiers on:
   a. MLP embeddings
   b. Raw input features
   c. PCA-transformed raw features (10D)
   - Classifiers: Logistic Regression, Random Forest, optional shallow MLP
   - Use train/test split or cross-validation (e.g. 5-fold)

6. Compute and save metrics:
   - Accuracy, F1, ROC-AUC
   - Training time
   - Embedding vector dimensionality

7. Visualize embeddings:
   - Apply t-SNE or PCA on embeddings and raw features
   - Save plots to:
     results/embeddings/plots/<dataset>_embedding_tsne.png

8. Save all outputs to:
   - results/embeddings/metrics_<dataset>.csv
   - results/embeddings/plots/
   - results/embeddings/downstream_predictions_<dataset>.csv (optional)

Make the script modular and include quoted comments at the top describing its purpose.




"Augment the existing `evaluate_mlp_embeddings.py` script to support three deeper embedding evaluation experiments for RQ3: (1) embedding generalization across tasks, (2) representation similarity analysis, and (3) robustness of embeddings under perturbations.

🔧 Base Setup:
- Reuse existing embeddings from MLP and LightGBM models stored as `.npy` files.
- Datasets: Use `diamonds_v2` and `telco` (already have saved MLP models and downstream labels).
- Save all new outputs under `results/embeddings_extended_analysis/`.

🧠 1. Embedding Generalization Across Tasks:
- Train MLP on original task (e.g., telco churn), extract embeddings.
- Reuse same embeddings to predict a new downstream label (e.g., Contract type).
- Compare accuracy, ROC-AUC, and F1 of:
  - embeddings → downstream label
  - raw features → downstream label
  - PCA(10) features → downstream label
- Save metrics to `generalization_metrics_{dataset}.csv` and predictions to `generalization_predictions_{dataset}.csv`.

🔍 2. Representation Similarity Analysis:
- Compute similarity between embeddings from:
  - MLP vs. LGBM on the same dataset.
- Use CKA (via `cka` Python package) or optionally SVCCA.
- Save CKA values in `cka_similarity_{dataset}.json`.

🛡️ 3. Embedding Robustness under Perturbation:
- Take clean-trained MLP, inject noise or missingness into **inputs at inference time**.
- Run downstream tasks on:
  - perturbed embeddings
  - clean embeddings
- Visualize embedding drift (PCA/TSNE plots).
- Compare performance drops and save:
  - `robustness_metrics_{dataset}.csv`
  - `embedding_drift_{dataset}.png`

Ensure modular design so each experiment can be toggled via CLI:
--run-generalization
--run-cka
--run-robustness

Add argparse and log output summaries to terminal."


"""

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

try:
    from sklearn.manifold import TSNE
except Exception:
    TSNE = None

RANDOM_STATE = 42
RESULTS_DIR = Path("results/embeddings")
PLOTS_DIR = RESULTS_DIR / "plots"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
EXTENDED_DIR = Path("results/embeddings_extended_analysis")
EXTENDED_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class DatasetConfig:
    name: str
    path: Path
    target: str
    task: str


DATASETS: Dict[str, DatasetConfig] = {
    "diamonds_v2": DatasetConfig(
        "diamonds_v2", Path("data/diamonds_v2.csv"), "total_sales_price", "regression"
    ),
    "telco": DatasetConfig(
        "telco", Path("data/telco_churn_classification.csv"), "Churn", "classification"
    ),
}

TREE_EMBEDDING_MODELS = {"lgbm"}


def load_dataset(cfg: DatasetConfig) -> pd.DataFrame:
    df = pd.read_csv(cfg.path)
    df = df.dropna().reset_index(drop=True)
    return df


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    # Match train_optuna preprocessing: scale numeric + one-hot categorical.
    cat_cols = [c for c in X.columns if X[c].dtype == object]
    num_cols = [c for c in X.columns if c not in cat_cols]
    transformers = []
    if num_cols:
        transformers.append(("num", StandardScaler(), num_cols))
    if cat_cols:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols))
    return ColumnTransformer(transformers)


def extract_mlp_embeddings(mlp_model, X: np.ndarray) -> np.ndarray:
    # Forward pass through hidden layers to extract penultimate activations.
    activ = mlp_model.activation

    def _apply_activation(z: np.ndarray) -> np.ndarray:
        if activ == "relu":
            return np.maximum(0.0, z)
        if activ == "tanh":
            return np.tanh(z)
        if activ == "logistic":
            return 1.0 / (1.0 + np.exp(-z))
        return z

    hidden = X
    for idx in range(len(mlp_model.coefs_) - 1):
        hidden = _apply_activation(hidden @ mlp_model.coefs_[idx] + mlp_model.intercepts_[idx])
    return hidden


def make_downstream_labels(df: pd.DataFrame, cfg: DatasetConfig, telco_label: str) -> pd.Series:
    # Create classification labels for downstream tasks.
    if cfg.name == "diamonds_v2":
        bins = pd.qcut(df[cfg.target], q=3, labels=["low", "mid", "high"], duplicates="drop")
        if bins.isna().any():
            raise ValueError("Quantile binning produced NaNs; check target distribution.")
        return bins.astype(str)
    label_col = telco_label if telco_label in df.columns else cfg.target
    return df[label_col].astype(str)


def apply_label_noise(y: pd.Series, rate: float, rng: np.random.Generator) -> pd.Series:
    # Flip labels at a specified rate to simulate noisy targets.
    if rate <= 0:
        return y
    y_noisy = y.copy()
    classes = y_noisy.unique().tolist()
    if len(classes) < 2:
        return y_noisy
    noisy_idx = rng.choice(y_noisy.index, size=int(rate * len(y_noisy)), replace=False)
    for idx in noisy_idx:
        current = y_noisy.loc[idx]
        choices = [c for c in classes if c != current]
        y_noisy.loc[idx] = rng.choice(choices)
    return y_noisy


def apply_imbalance_indices(y: pd.Series, ratio: float, rng: np.random.Generator) -> np.ndarray:
    # Downsample majority classes to simulate imbalance and return kept indices.
    if ratio <= 0 or ratio >= 1:
        return y.index.to_numpy()
    counts = y.value_counts()
    min_count = counts.min()
    target_count = max(int(min_count / max(ratio, 1e-6)), min_count)
    keep_idx = []
    for cls, cnt in counts.items():
        cls_idx = y[y == cls].index.to_numpy()
        if cnt > target_count:
            keep_idx.extend(rng.choice(cls_idx, size=target_count, replace=False))
        else:
            keep_idx.extend(cls_idx)
    return np.array(keep_idx)


def select_generalization_label(df: pd.DataFrame, cfg: DatasetConfig, telco_label: str) -> str:
    # Choose a downstream label that differs from the primary task.
    if cfg.name == "telco":
        if telco_label in df.columns:
            return telco_label
        return cfg.target
    for candidate in ["cut", "color", "clarity", "lab"]:
        if candidate in df.columns and candidate != cfg.target:
            return candidate
    return cfg.target


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    # Linear CKA similarity between two representations.
    Xc = X - X.mean(axis=0, keepdims=True)
    Yc = Y - Y.mean(axis=0, keepdims=True)
    hsic = np.linalg.norm(Xc.T @ Yc, ord="fro") ** 2
    norm_x = np.linalg.norm(Xc.T @ Xc, ord="fro")
    norm_y = np.linalg.norm(Yc.T @ Yc, ord="fro")
    denom = max(norm_x * norm_y, 1e-12)
    return float(hsic / denom)


def perturb_inputs(
    X: pd.DataFrame, mode: str, rng: np.random.Generator, rate: float = 0.1
) -> pd.DataFrame:
    # Inject noise or missingness at inference time.
    X_mut = X.copy()
    if mode == "noise":
        num_cols = X_mut.select_dtypes(include=[np.number]).columns.tolist()
        for col in num_cols:
            scale = X_mut[col].std() or 1.0
            X_mut[col] = X_mut[col] + rng.normal(0.0, scale * rate, size=len(X_mut))
    else:
        cols = X_mut.columns.tolist()
        selected = rng.choice(cols, size=max(1, int(0.2 * len(cols))), replace=False)
        for col in selected:
            mask = rng.random(len(X_mut)) < rate
            X_mut.loc[mask, col] = np.nan
    return X_mut


def save_extended_metrics(path: Path, rows: List[Dict[str, float]]) -> None:
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)


def run_generalization_experiment(
    cfg: DatasetConfig,
    df: pd.DataFrame,
    embeddings: np.ndarray,
    preprocessor: ColumnTransformer,
    args,
) -> None:
    # Evaluate embeddings vs raw vs PCA on a new downstream label.
    label_col = select_generalization_label(df, cfg, args.telco_label)
    y = df[label_col].astype(str).to_numpy()
    X_model = df.drop(columns=[cfg.target])
    X_proc = preprocessor.transform(X_model)
    if hasattr(X_proc, "toarray"):
        X_proc = X_proc.toarray()

    X_train, X_test, y_train, y_test, emb_train, emb_test = train_test_split(
        X_proc,
        y,
        embeddings,
        test_size=0.2,
        random_state=args.split_seed,
        stratify=y,
    )
    pca = PCA(n_components=10, random_state=RANDOM_STATE)
    pca_train = pca.fit_transform(X_train)
    pca_test = pca.transform(X_test)

    classifiers = ["logreg", "rf", "mlp"]
    metrics_rows = []
    preds_rows = []
    feature_sets = {
        "embeddings": (emb_train, emb_test),
        "raw": (X_train, X_test),
        "pca10": (pca_train, pca_test),
    }
    for feature_name, (X_tr, X_te) in feature_sets.items():
        for clf in classifiers:
            metrics, y_pred, y_proba = train_and_eval(
                X_tr, X_te, y_train, y_test, clf, len(np.unique(y)) > 2
            )
            metrics.update(
                {
                    "dataset": cfg.name,
                    "feature_set": feature_name,
                    "classifier": clf,
                    "label": label_col,
                }
            )
            metrics_rows.append(metrics)
            preds_rows.extend(
                [
                    {
                        "dataset": cfg.name,
                        "feature_set": feature_name,
                        "classifier": clf,
                        "label": label_col,
                        "y_true": y_test[i],
                        "y_pred": y_pred[i],
                    }
                    for i in range(len(y_test))
                ]
            )

    save_extended_metrics(
        EXTENDED_DIR / f"generalization_metrics_{cfg.name}.csv", metrics_rows
    )
    pd.DataFrame(preds_rows).to_csv(
        EXTENDED_DIR / f"generalization_predictions_{cfg.name}.csv", index=False
    )
    print(
        f"✔ Generalization metrics saved for {cfg.name} (label={label_col})"
    )


def run_cka_experiment(cfg: DatasetConfig) -> None:
    # Compare MLP vs LGBM embeddings using linear CKA.
    mlp_path = RESULTS_DIR / f"{cfg.name}_mlp_embeddings.npy"
    lgbm_path = RESULTS_DIR / f"{cfg.name}_lgbm_leaf_embeddings.npy"
    if not mlp_path.exists() or not lgbm_path.exists():
        print(f"⚠️ Missing embeddings for CKA on {cfg.name}, skipping.")
        return
    mlp_emb = np.load(mlp_path)
    lgbm_emb = np.load(lgbm_path)
    n = min(len(mlp_emb), len(lgbm_emb))
    cka_val = linear_cka(mlp_emb[:n], lgbm_emb[:n])
    out_path = EXTENDED_DIR / f"cka_similarity_{cfg.name}.json"
    with open(out_path, "w") as f:
        json.dump({"cka_linear": cka_val}, f, indent=2)
    print(f"✔ CKA similarity saved for {cfg.name}: {cka_val:.4f}")


def run_robustness_experiment(
    cfg: DatasetConfig,
    df: pd.DataFrame,
    y_downstream: pd.Series,
    preprocessor: ColumnTransformer,
    mlp_model,
    args,
) -> None:
    # Evaluate downstream performance under perturbed embeddings.
    rng = np.random.default_rng(args.split_seed)
    y = y_downstream.astype(str).to_numpy()
    X_model = df.drop(columns=[cfg.target])
    X_clean_proc = preprocessor.transform(X_model)
    if hasattr(X_clean_proc, "toarray"):
        X_clean_proc = X_clean_proc.toarray()
    clean_emb = extract_mlp_embeddings(mlp_model, X_clean_proc)

    X_pert = perturb_inputs(X_model, mode="noise", rng=rng, rate=0.1)
    X_pert_proc = preprocessor.transform(X_pert)
    if hasattr(X_pert_proc, "toarray"):
        X_pert_proc = X_pert_proc.toarray()
    pert_emb = extract_mlp_embeddings(mlp_model, X_pert_proc)

    X_train_c, X_test_c, y_train, y_test, emb_train_c, emb_test_c = train_test_split(
        X_clean_proc,
        y,
        clean_emb,
        test_size=0.2,
        random_state=args.split_seed,
        stratify=y,
    )
    _, _, _, _, emb_train_p, emb_test_p = train_test_split(
        X_pert_proc,
        y,
        pert_emb,
        test_size=0.2,
        random_state=args.split_seed,
        stratify=y,
    )

    classifiers = ["logreg", "rf", "mlp"]
    rows = []
    for clf in classifiers:
        clean_metrics, _, _ = train_and_eval(
            emb_train_c, emb_test_c, y_train, y_test, clf, len(np.unique(y)) > 2
        )
        clean_metrics.update(
            {"dataset": cfg.name, "classifier": clf, "scenario": "clean_embeddings"}
        )
        pert_metrics, _, _ = train_and_eval(
            emb_train_p, emb_test_p, y_train, y_test, clf, len(np.unique(y)) > 2
        )
        pert_metrics.update(
            {"dataset": cfg.name, "classifier": clf, "scenario": "perturbed_embeddings"}
        )
        rows.extend([clean_metrics, pert_metrics])

    save_extended_metrics(EXTENDED_DIR / f"robustness_metrics_{cfg.name}.csv", rows)

    # Drift visualization via PCA.
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    coords_clean = pca.fit_transform(clean_emb)
    coords_pert = pca.transform(pert_emb)
    import matplotlib.pyplot as plt

    plt.figure(figsize=(6, 4))
    plt.scatter(coords_clean[:, 0], coords_clean[:, 1], s=8, alpha=0.5, label="clean")
    plt.scatter(coords_pert[:, 0], coords_pert[:, 1], s=8, alpha=0.5, label="perturbed")
    plt.legend(fontsize=8, loc="best")
    plt.title(f"Embedding Drift (PCA): {cfg.name}")
    plt.tight_layout()
    plt.savefig(EXTENDED_DIR / f"embedding_drift_{cfg.name}.png", dpi=150)
    plt.close()
    print(f"✔ Robustness metrics saved for {cfg.name}")


def evaluate_classification(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    is_multiclass: bool,
) -> Dict[str, float]:
    # Compute accuracy, F1, and ROC-AUC where applicable.
    if is_multiclass:
        f1_val = f1_score(y_true, y_pred, average="macro")
    else:
        le = LabelEncoder()
        y_true_enc = le.fit_transform(y_true)
        y_pred_enc = le.transform(y_pred)
        f1_val = f1_score(y_true_enc, y_pred_enc, average="binary")
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_val),
    }
    try:
        le = LabelEncoder()
        y_true_enc = le.fit_transform(y_true)
        if is_multiclass:
            metrics["roc_auc"] = float(
                roc_auc_score(y_true_enc, y_proba, multi_class="ovr", average="macro")
            )
        else:
            metrics["roc_auc"] = float(roc_auc_score(y_true_enc, y_proba[:, 1]))
    except Exception:
        metrics["roc_auc"] = float("nan")
    return metrics


def train_and_eval(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    classifier_name: str,
    is_multiclass: bool,
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    # Train a classifier and evaluate on the test set.
    if classifier_name == "logreg":
        clf = LogisticRegression(max_iter=1000, multi_class="auto")
        model = Pipeline([("scaler", StandardScaler()), ("clf", clf)])
    elif classifier_name == "rf":
        model = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)
    else:
        clf = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300, random_state=RANDOM_STATE)
        model = Pipeline([("scaler", StandardScaler()), ("clf", clf)])

    start = time.time()
    model.fit(X_train, y_train)
    elapsed = time.time() - start
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    metrics = evaluate_classification(y_test, y_pred, y_proba, is_multiclass)
    metrics["train_time_sec"] = float(elapsed)
    return metrics, y_pred, y_proba


def run_split_eval(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    classifiers: List[str],
    is_multiclass: bool,
) -> Tuple[List[Dict[str, float]], pd.DataFrame]:
    # Train/test evaluation with prediction logging.
    metrics_rows: List[Dict[str, float]] = []
    preds_rows = []
    for clf_name in classifiers:
        metrics, y_pred, y_proba = train_and_eval(
            X_train, X_test, y_train, y_test, clf_name, is_multiclass
        )
        metrics["classifier"] = clf_name
        metrics_rows.append(metrics)
        preds_rows.extend(
            [
                {
                    "classifier": clf_name,
                    "y_true": y_test[i],
                    "y_pred": y_pred[i],
                }
                for i in range(len(y_test))
            ]
        )
    preds_df = pd.DataFrame(preds_rows)
    return metrics_rows, preds_df


def run_cv_eval(
    X: np.ndarray,
    y: np.ndarray,
    classifiers: List[str],
    is_multiclass: bool,
    n_splits: int,
) -> List[Dict[str, float]]:
    # Cross-validated evaluation for more stable metrics.
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    metrics_rows: List[Dict[str, float]] = []
    for clf_name in classifiers:
        fold_metrics = []
        for train_idx, test_idx in cv.split(X, y):
            metrics, _, _ = train_and_eval(
                X[train_idx],
                X[test_idx],
                y[train_idx],
                y[test_idx],
                clf_name,
                is_multiclass,
            )
            fold_metrics.append(metrics)
        avg = {
            "accuracy": float(np.mean([m["accuracy"] for m in fold_metrics])),
            "f1": float(np.mean([m["f1"] for m in fold_metrics])),
            "roc_auc": float(np.mean([m["roc_auc"] for m in fold_metrics])),
            "train_time_sec": float(np.mean([m["train_time_sec"] for m in fold_metrics])),
            "classifier": clf_name,
        }
        metrics_rows.append(avg)
    return metrics_rows


def plot_embeddings(X: np.ndarray, y: np.ndarray, out_path: Path, method: str) -> None:
    # Save a 2D visualization of embeddings.
    if method == "tsne":
        if TSNE is None:
            raise RuntimeError("TSNE not available in this environment.")
        reducer = TSNE(n_components=2, random_state=RANDOM_STATE, perplexity=30, init="pca")
    else:
        reducer = PCA(n_components=2, random_state=RANDOM_STATE)

    X_vis = X
    if X_vis.shape[0] > 2000:
        idx = np.random.default_rng(RANDOM_STATE).choice(X_vis.shape[0], 2000, replace=False)
        X_vis = X_vis[idx]
        y = y[idx]

    coords = reducer.fit_transform(X_vis)
    df_plot = pd.DataFrame({"x": coords[:, 0], "y": coords[:, 1], "label": y})

    import matplotlib.pyplot as plt

    plt.figure(figsize=(6, 4))
    for label in np.unique(y):
        subset = df_plot[df_plot["label"] == label]
        plt.scatter(subset["x"], subset["y"], s=10, alpha=0.6, label=str(label))
    plt.legend(fontsize=7, loc="best")
    plt.title(f"Embedding visualization ({method})")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def extract_lgbm_leaf_embeddings(model, X: np.ndarray) -> np.ndarray:
    # Use LightGBM leaf indices as tree-based embeddings.
    return model.predict(X, pred_leaf=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate MLP embeddings for downstream tasks.")
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS.keys()))
    parser.add_argument("--telco-label", default="Churn")
    parser.add_argument("--cv", type=int, default=0)
    parser.add_argument("--plot-method", choices=["tsne", "pca"], default="pca")
    parser.add_argument("--label-noise", type=float, default=0.0)
    parser.add_argument("--imbalance-ratio", type=float, default=0.0)
    parser.add_argument("--split-seed", type=int, default=RANDOM_STATE)
    parser.add_argument("--run-generalization", action="store_true")
    parser.add_argument("--run-cka", action="store_true")
    parser.add_argument("--run-robustness", action="store_true")
    args = parser.parse_args()

    classifiers = ["logreg", "rf", "mlp"]
    for dataset_name in args.datasets:
        cfg = DATASETS[dataset_name]
        df = load_dataset(cfg)
        rng = np.random.default_rng(args.split_seed)
        y_target = df[cfg.target]
        y_downstream = make_downstream_labels(df, cfg, args.telco_label)
        if args.label_noise > 0:
            y_downstream = apply_label_noise(y_downstream, args.label_noise, rng)
        if args.imbalance_ratio > 0:
            keep_idx = apply_imbalance_indices(y_downstream, args.imbalance_ratio, rng)
            df = df.loc[keep_idx].reset_index(drop=True)
            y_downstream = y_downstream.loc[keep_idx].reset_index(drop=True)
            y_target = df[cfg.target]

        label_col = args.telco_label if cfg.name == "telco" else cfg.target
        drop_cols = [cfg.target]
        if label_col != cfg.target and label_col in df.columns:
            drop_cols.append(label_col)
        X_downstream = df.drop(columns=drop_cols)
        X_model = df.drop(columns=[cfg.target])

        # Match MLP training preprocessing (fit on original target split).
        stratify = y_target if cfg.task == "classification" else None
        X_base_train, _, _, _ = train_test_split(
            X_model, y_target, test_size=0.2, random_state=RANDOM_STATE, stratify=stratify
        )
        preprocessor = build_preprocessor(X_base_train)
        preprocessor.fit(X_base_train)
        X_full_proc = preprocessor.transform(X_model)
        if hasattr(X_full_proc, "toarray"):
            X_full_proc = X_full_proc.toarray()

        # Downstream split uses labels for the downstream task.
        X_train, X_test, y_train, y_test = train_test_split(
            X_full_proc,
            y_downstream,
            test_size=0.2,
            random_state=args.split_seed,
            stratify=y_downstream,
        )

        mlp_path = Path("results/hpo") / cfg.name / "mlp" / "best_model.pkl"
        if not mlp_path.exists():
            raise FileNotFoundError(f"Missing MLP model at {mlp_path}")
        mlp_model = joblib.load(mlp_path)

        # Extract embeddings for full dataset and save.
        embeddings_full = extract_mlp_embeddings(mlp_model, X_full_proc)
        emb_path = RESULTS_DIR / f"{cfg.name}_mlp_embeddings.npy"
        np.save(emb_path, embeddings_full)
        labels_path = RESULTS_DIR / f"{cfg.name}_downstream_labels.csv"
        pd.DataFrame({"label": y_downstream}).to_csv(labels_path, index=False)

        # Prepare embeddings for train/test split.
        embeddings_train = extract_mlp_embeddings(mlp_model, X_train)
        embeddings_test = extract_mlp_embeddings(mlp_model, X_test)

        metrics_rows: List[Dict[str, float]] = []
        preds_df_list = []

        feature_sets_split = {
            "embeddings": (embeddings_train, embeddings_test, embeddings_full.shape[1]),
            "raw": (X_train, X_test, X_train.shape[1]),
        }

        # PCA features on raw input.
        pca = PCA(n_components=10, random_state=RANDOM_STATE)
        pca_train = pca.fit_transform(X_train)
        pca_test = pca.transform(X_test)
        feature_sets_split["pca10"] = (pca_train, pca_test, pca_train.shape[1])

        pca_full = pca.transform(X_full_proc)
        feature_sets_full = {
            "embeddings": (embeddings_full, embeddings_full.shape[1]),
            "raw": (X_full_proc, X_full_proc.shape[1]),
            "pca10": (pca_full, pca_full.shape[1]),
        }

        # Optional tree-based embeddings via LightGBM leaf indices.
        lgbm_path = Path("results/hpo") / cfg.name / "lgbm" / "best_model.pkl"
        if lgbm_path.exists():
            lgbm_model = joblib.load(lgbm_path)
            lgbm_leaf_full = extract_lgbm_leaf_embeddings(lgbm_model, X_full_proc)
            np.save(RESULTS_DIR / f"{cfg.name}_lgbm_leaf_embeddings.npy", lgbm_leaf_full)
            lgbm_leaf_train = extract_lgbm_leaf_embeddings(lgbm_model, X_train)
            lgbm_leaf_test = extract_lgbm_leaf_embeddings(lgbm_model, X_test)
            feature_sets_split["lgbm_leaf"] = (
                lgbm_leaf_train,
                lgbm_leaf_test,
                lgbm_leaf_train.shape[1],
            )
            feature_sets_full["lgbm_leaf"] = (lgbm_leaf_full, lgbm_leaf_full.shape[1])

        for feature_name, (X_tr, X_te, dim) in feature_sets_split.items():
            if args.cv and args.cv > 1:
                X_full, dim_full = feature_sets_full[feature_name]
                rows = run_cv_eval(
                    X_full,
                    y_downstream.to_numpy(),
                    classifiers,
                    y_downstream.nunique() > 2,
                    args.cv,
                )
                for row in rows:
                    row.update({"feature_set": feature_name, "embedding_dim": dim_full})
                    metrics_rows.append(row)
            else:
                rows, preds_df = run_split_eval(
                    X_tr,
                    X_te,
                    y_train.to_numpy(),
                    y_test.to_numpy(),
                    classifiers,
                    y_downstream.nunique() > 2,
                )
                for row in rows:
                    row.update({"feature_set": feature_name, "embedding_dim": dim})
                    metrics_rows.append(row)
                preds_df["feature_set"] = feature_name
                preds_df_list.append(preds_df)

        metrics_df = pd.DataFrame(metrics_rows)
        metrics_df["dataset"] = cfg.name
        metrics_df["label_noise"] = args.label_noise
        metrics_df["imbalance_ratio"] = args.imbalance_ratio
        metrics_df["split_seed"] = args.split_seed
        metrics_df.to_csv(RESULTS_DIR / f"metrics_{cfg.name}.csv", index=False)

        if preds_df_list:
            preds_all = pd.concat(preds_df_list, ignore_index=True)
            preds_all["dataset"] = cfg.name
            preds_all["label_noise"] = args.label_noise
            preds_all["imbalance_ratio"] = args.imbalance_ratio
            preds_all["split_seed"] = args.split_seed
            preds_all.to_csv(RESULTS_DIR / f"downstream_predictions_{cfg.name}.csv", index=False)

        # Visualization for embeddings and raw features.
        plot_embeddings(
            embeddings_full,
            y_downstream.to_numpy(),
            PLOTS_DIR / f"{cfg.name}_embedding_{args.plot_method}.png",
            method=args.plot_method,
        )
        plot_embeddings(
            X_full_proc,
            y_downstream.to_numpy(),
            PLOTS_DIR / f"{cfg.name}_raw_{args.plot_method}.png",
            method=args.plot_method,
        )

        # Extended analysis toggles.
        if args.run_generalization:
            run_generalization_experiment(cfg, df, embeddings_full, preprocessor, args)
        if args.run_cka:
            run_cka_experiment(cfg)
        if args.run_robustness:
            run_robustness_experiment(cfg, df, y_downstream, preprocessor, mlp_model, args)


if __name__ == "__main__":
    main()
