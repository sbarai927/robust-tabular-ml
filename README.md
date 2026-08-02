# robust-tabular-ml

## Understanding Tree Model Dominance in Tabular Learning

> Reproducible research repository for a controlled comparison of classical tree ensembles, neural tabular models, and pretrained tabular models across clean predictive performance, hyperparameter optimization, robustness stress tests, covariate shift, computational cost, and feature-attribution stability.

This repository accompanies the study:

**Understanding Tree Model Dominance in Tabular Learning — Performance, Robustness, and Stability Across Classical, Deep and Foundation (Pre-trained) Models**

The central question is not only which model obtains the best clean-data score, but also which model families remain reliable when tabular inputs become incomplete, structurally more difficult, or distributionally shifted.

> **Scope note:** The results are evidence from a controlled six-dataset benchmark. They should not be interpreted as a universal ranking of all tabular-learning methods.

---

## Manuscripts

- [Original manuscript](docs/main_original.pdf)
- [Revised manuscript](docs/main_revised.pdf)
- [Final reviewer-addressed revision](docs/main_revised_final.pdf)

---

## Table of Contents

1. [Project overview](#project-overview)
2. [Research questions](#research-questions)
3. [Benchmark design](#benchmark-design)
4. [Datasets](#datasets)
5. [Models](#models)
6. [Evaluation metrics](#evaluation-metrics)
7. [Headline findings](#headline-findings)
8. [Repository setup](#repository-setup)
9. [Running the experiments](#running-the-experiments)
10. [Reproducing tables and analyses](#reproducing-tables-and-analyses)
11. [Result organization](#result-organization)
12. [Repository map](#repository-map)
13. [Reproducibility notes and limitations](#reproducibility-notes-and-limitations)
14. [Citation](#citation)
15. [Contact](#contact)

---

## Project Overview

The project evaluates tabular models from three broad families:

- **Classical tree ensembles:** Random Forest, XGBoost, LightGBM, and CatBoost
- **Neural tabular models:** MLP and TabNet
- **Pretrained/foundation-style models:** TabPFN and APT

The benchmark combines five complementary views:

1. **Clean predictive performance after HPO**
2. **Predictive quality versus training-time and model-size trade-offs**
3. **Robustness to injected missingness**
4. **Robustness to synthetic high-cardinality categorical structure**
5. **Predictive and SHAP-based stability under covariate shift**

Reviewer-requested extensions additionally provide:

- exact per-dataset model scores;
- repeated-seed uncertainty checks;
- train–test gap analysis;
- completed XGBoost and LightGBM robustness/shift coverage;
- explicit failure records where SHAP or pretrained-model outputs were unavailable.

---

## Research Questions

The repository supports the following practical research questions:

- **RQ1 — Clean performance:** Which model families achieve the strongest and most consistent results across heterogeneous classification and regression datasets?
- **RQ2 — Efficiency:** How do predictive quality, fitting time, and serialized model size interact?
- **RQ3 — Missingness robustness:** How strongly do model performance and feature reliance change when selected input columns contain additional missing values?
- **RQ4 — Categorical stress:** How do models respond to uninformative high-cardinality categorical nuisance structure?
- **RQ5 — Covariate shift:** Do models preserve useful performance and similar dominant feature attributions when the test distribution changes?
- **RQ6 — Stability and uncertainty:** How sensitive are final selected configurations to data splitting, random initialization, and train–test overfitting?

---

## Benchmark Design

### Clean/HPO stage

The main HPO driver is [`src/hpo_tuning/train_optuna.py`](src/hpo_tuning/train_optuna.py).

- Optuna TPE sampler
- seed: `42`
- default budget: `30` trials per tunable dataset–model pair
- classification objective: F1, using macro-F1 for multiclass datasets
- regression objective in the current implementation: RMSE minimization
- final regression outputs additionally report R² and MAE
- TabPFN and APT use default/fixed pretrained configurations rather than the full Optuna search

### Missingness stress

The robustness runner selects 20% of input columns and masks 15% of values within those columns in both the training and test partitions. The same clean-data hyperparameter configuration is reused for the stressed condition.

### High-cardinality stress

The robustness runner adds two synthetic categorical nuisance columns with 150 possible category suffixes. The generated columns are not designed to carry target signal. For pretrained models, the script uses a compatible overwrite-based variant when adding new dimensions is not supported.

### Covariate shift

The shift analysis perturbs the test features while keeping labels unchanged:

- numerical features receive a `0.2σ` mean shift, Gaussian noise with standard deviation `0.5σ`, and 2% outlier injections of magnitude `3σ`;
- categorical features receive a controlled mode-frequency replacement and unseen-category injection.

Predictive performance is evaluated before and after shift. SHAP importance rankings are compared using Spearman rank correlation and top-k Jaccard overlap where computation is available.

### Repeated-seed analysis

The lightweight uncertainty analysis repeats final selected configurations using seeds `0`, `42`, and `123`. It is intentionally not a full nested cross-validation procedure.

---

## Datasets

The repository uses six prepared CSV datasets under [`data/`](data/).

| Dataset ID | Task | Target | Rows | Features | Notes |
|---|---|---|---:|---:|---|
| `diamonds_v2` | Regression | `total_sales_price` | 219,703 | 24 | Pearson filtering threshold 0.85 |
| `diamonds_v3` | Regression | `total_sales_price` | 219,703 | 22 | Pearson filtering threshold 0.70 |
| `drybean` | Multiclass classification | `Class` | 4,970 | 12 | Numerical bean-shape descriptors |
| `mnist` | Multiclass classification | `label` | 1,797 | 64 | scikit-learn 8×8 tabular digits; not the original 70k MNIST dataset |
| `telco` | Binary classification | `Churn` | 7,032 | 20 | Mixed, categorical-heavy churn data |
| `titanic` | Binary classification | `Survived` | 183 | 11 | Complete-case mixed-feature subset |

The checked-in filenames are:

```text
data/diamonds_v2.csv
data/diamonds_v3.csv
data/drybean_multiclass_classification.csv
data/mnist_tabular_digits.csv
data/telco_churn_classification.csv
data/titanic_binary_classification.csv
```

Dataset ownership and licensing remain with the original data providers. Source citations are documented in the manuscript.

---

## Models

| Model | Family | Main task support in this repository | Tuning strategy |
|---|---|---|---|
| Random Forest | Tree ensemble | Classification and regression | Optuna HPO |
| XGBoost | Gradient-boosted trees | Classification and regression | Optuna HPO |
| LightGBM | Gradient-boosted trees | Classification and regression | Optuna HPO |
| CatBoost | Gradient-boosted trees | Classification and regression | Optuna HPO |
| MLP | Neural baseline | Classification and regression | Optuna HPO |
| TabNet | Neural tabular model | Classification and regression | Optuna HPO |
| TabPFN | Pretrained tabular model | Classification | Default pretrained configuration |
| APT | Pretrained tabular model | Classification in the main pipeline | Fixed pretrained checkpoint |

The HPO driver also contains auxiliary FT-Transformer-style experimental support, but it is not part of the final eight-model manuscript comparison.

---

## Evaluation Metrics

### Predictive metrics

- **Binary classification:** Accuracy and positive-class F1
- **Multiclass classification:** Accuracy and macro-F1
- **Regression:** R², RMSE, and MAE

### Efficiency metrics

- wall-clock fitting time;
- serialized model size;
- relative time and size ranks across datasets.

For pretrained models, reported repository runtime excludes the cost of external pretraining and covers only dataset-level setup, fitting/context construction, and evaluation.

### Robustness and stability metrics

- clean-to-perturbed predictive delta;
- clean-to-shifted predictive delta;
- SHAP top-10 overlap count;
- SHAP top-10 Jaccard similarity;
- SHAP importance-rank Spearman correlation;
- train–test score gap;
- repeated-seed mean and standard deviation.

SHAP is used as a relative attribution-stability diagnostic, not as evidence of causal feature use.

---

## Headline Findings

The committed result summaries support the following restrained conclusions:

- XGBoost, CatBoost, and LightGBM occupy the strongest overall mean-rank region across the six clean benchmark datasets.
- Random Forest is especially strong on the two regression variants, while boosted trees are more consistent across the mixed task suite.
- Tree-based models generally retain higher top-feature overlap than MLP and TabNet under missingness and synthetic high-cardinality stress.
- Under covariate shift, the strongest shifted model depends on the metric: boosted trees remain strong in shifted accuracy, while TabNet is competitive in shifted F1.
- Among models with available SHAP outputs, CatBoost, LightGBM, and Random Forest show the strongest attribution stability under shift.
- TabPFN and APT reduce dataset-specific tuning effort but do not consistently establish a new performance–robustness frontier in this controlled benchmark.

Detailed values are available in [`results/paper_tables/`](results/paper_tables/) and the final manuscript.

---

## Repository Setup

### Prerequisites

- Python 3.12 is recommended for the core repository environment.
- Git LFS is required to retrieve tracked `.pkl`, `.npy`, and `.db` artifacts.
- A C/C++ toolchain may be required when binary wheels for LightGBM, XGBoost, or CatBoost are unavailable.
- Foundation-model dependencies are comparatively heavy and may require separate environments.

### Clone and install the core environment

```bash
git lfs install
git clone https://github.com/sbarai927/robust-tabular-ml.git
cd robust-tabular-ml
git lfs pull

python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Optional model and explainability dependencies

The current top-level `requirements.txt` captures the core repository environment but does not fully pin every optional deep/foundation dependency. Install only the components required for the experiments you intend to run.

Typical optional packages include:

```bash
pip install shap tqdm pytorch-tabnet tabpfn
```

Install PyTorch using the wheel appropriate for your operating system and hardware.

APT is included under [`external/APT/`](external/APT/). Its upstream installation procedure is:

```bash
cd external/APT
pip install -r requirements.txt
pip install -e .
cd ../..
```

APT's upstream project targets a separate Python/PyTorch environment. On constrained or incompatible systems, run APT independently and preserve the resulting repository outputs rather than forcing it into the core environment.

---

## Running the Experiments

Run commands from the repository root.

### 1. Hyperparameter optimization

Run one dataset–model pair:

```bash
python src/hpo_tuning/train_optuna.py \
  --dataset telco \
  --model catboost \
  --trials 30
```

Run the configured benchmark matrix:

```bash
python src/hpo_tuning/train_optuna.py --trials 30
```

Useful options:

```text
--dataset <one or more dataset IDs>
--model <one or more model IDs>
--trials <number of trials>
--timeout <seconds per study; 0 disables>
--overwrite
```

Existing Optuna SQLite studies are resumed when possible. Existing complete results are skipped unless `--overwrite` is supplied.

### 2. Missingness and high-cardinality robustness

Example for the principal tunable models:

```bash
python src/robustness/run_missingness_highcard_robustness.py \
  --datasets diamonds_v2 telco drybean mnist \
  --models rf xgboost lgbm catboost mlp tabnet
```

Include pretrained baselines only when their dependencies and saved HPO/default artifacts are available:

```bash
python src/robustness/run_missingness_highcard_robustness.py \
  --datasets telco drybean mnist \
  --models tabpfn apt
```

Recompute only feature-importance outputs from existing artifacts:

```bash
python src/robustness/run_missingness_highcard_robustness.py \
  --datasets telco drybean \
  --models rf catboost mlp \
  --shap-only
```

### 3. Covariate shift and SHAP stability

```bash
python src/robustness/run_shap_stability_shift.py \
  --datasets titanic telco drybean \
  --models rf xgboost lgbm catboost mlp tabnet
```

Limit the SHAP sample size on resource-constrained machines:

```bash
python src/robustness/run_shap_stability_shift.py \
  --datasets telco \
  --models catboost mlp \
  --max-shap-samples 50
```

### 4. Repeated-seed uncertainty

```bash
python src/hpo_tuning/run_repeated_seed_uncertainty.py \
  --seeds 0 42 123 \
  --max-rows-per-dataset 10000
```

By default, pretrained baselines are documented as skipped because repeated random initialization is not directly comparable to the tunable model refits used in this lightweight check.

---

## Reproducing Tables and Analyses

These scripts read saved repository outputs and do not rerun the full benchmark unless explicitly stated.

### HPO model summary

```bash
python src/hpo_tuning/build_hpo_summary_table.py
```

### Individual dataset scores

```bash
python src/hpo_tuning/build_individual_dataset_scores_table.py
```

### Train–test gap table

```bash
python src/hpo_tuning/build_train_test_gap_table.py
```

### Robustness summaries

```bash
python src/robustness/build_robustness_delta_summaries.py
python src/robustness/build_robustness_model_summary_table.py
python src/robustness/build_robustness_compact_table.py
```

### Covariate-shift stability summary

```bash
python src/robustness/build_stability_model_summary_table.py
```

### Classification critical-difference diagram

```bash
python src/hpo_tuning/plot_cd_diagram_classification.py
```

Publication-oriented CSV, LaTeX, caption, and note files are written primarily to [`results/paper_tables/`](results/paper_tables/). Figure source data are retained under [`results/paper_figure_delta_summary/`](results/paper_figure_delta_summary/).

---

## Result Organization

### HPO outputs

```text
results/hpo/<dataset>/<model>/
├── best_params.json
├── metrics.csv
├── trials.csv
├── study.db
├── study.pkl
└── best_model.pkl
```

### Robustness outputs

```text
results/robustness_challenges/<dataset>/<model>/<scenario>/
├── metrics.json
├── injection_log.json
├── meta_<scenario>.json
└── best_model.pkl
```

Aggregate robustness files include:

```text
results/robustness_challenges/metrics_<dataset>.csv
results/robustness_challenges/shap_top_k_<dataset>_<model>.json
results/robustness_challenges/shap_delta_<dataset>_<model>.json
```

### Covariate-shift outputs

```text
results/tree_vs_deep_stability_analysis/<dataset>/<model>/
├── performance_shift.json
├── shap_importance_clean.csv
├── shap_importance_shift.csv
├── shap_stability.json
└── shap_error.json              # retained when predictive evaluation succeeds but SHAP fails
```

### Manuscript-ready outputs

```text
results/paper_tables/                 # paper tables, LaTeX, notes, reviewer analyses
results/paper_figures/                # publication figures
results/paper_figure_delta_summary/   # source CSVs behind figures and rank tests
```

---

## Repository Map

```text
.
├── data/                              # Prepared benchmark CSV datasets
├── docs/                              # Original and revised manuscript PDFs
├── external/
│   └── APT/                           # Vendored upstream APT implementation
├── results/
│   ├── hpo/                           # Per-dataset/model HPO artifacts
│   ├── robustness_challenges/         # Clean, missingness, and cardinality runs
│   ├── tree_vs_deep_stability_analysis/ # Covariate-shift and SHAP stability
│   ├── embeddings/                    # Auxiliary embedding analyses
│   ├── embeddings_extended_analysis/  # Extended generalization/robustness analyses
│   ├── paper_tables/                  # Final and reviewer-requested tables
│   ├── paper_figures/                 # Manuscript figures
│   └── paper_figure_delta_summary/    # Figure source data and statistical summaries
├── src/
│   ├── hpo_tuning/                    # HPO, uncertainty, gap, ranking, and plotting utilities
│   ├── robustness/                    # Robustness, covariate shift, SHAP, and summary utilities
│   ├── embeddings/                    # Neural embedding evaluation code
│   └── probabilistic_eval/            # Auxiliary probabilistic evaluation utilities
├── requirements.txt                   # Core pinned Python environment
├── .gitattributes                     # Git LFS tracking for binary experiment artifacts
└── README.md
```

---

## Reproducibility Notes and Limitations

- **Run from the repository root.** Most paths are repository-relative.
- **Git LFS is required** for saved model, study, and database artifacts.
- **The current HPO implementation minimizes RMSE for regression.** R² is reported as a final evaluation metric. Manuscript text and repository code should remain consistent on this distinction.
- **Fixed trial counts are a bounded practical HPO protocol**, not proof of equal search fairness across model families.
- **The repeated-seed analysis is lightweight**, not nested cross-validation.
- **The robustness perturbations are controlled heuristic stress tests**, not calibrated simulations of every deployment environment.
- **The pretrained comparison is intentionally asymmetric:** TabPFN and APT use default/fixed pretrained configurations, while the other principal models receive dataset-specific HPO.
- **External pretraining cost is excluded** from the runtime comparison.
- **SHAP coverage is incomplete** for some model–dataset combinations. Predictive results are retained, and explicit error/NA records are used instead of fabricated stability values.
- **Telco RF and APT obtain F1 = 0 in the saved benchmark** because they predict only the majority non-churn class; this is a real result rather than a missing value.
- **The benchmark is moderate in scope:** six datasets, four robustness datasets, and three covariate-shift datasets.
- **SHAP does not establish causality.** It is used only to compare attribution consistency under controlled changes.

---

## Citation

A formal proceedings citation/DOI should replace this provisional entry when available.

```bibtex
@misc{barai2026tree_model_dominance,
  author       = {Suvendu Barai},
  title        = {Understanding Tree Model Dominance in Tabular Learning: Performance, Robustness, and Stability Across Classical, Deep and Foundation (Pre-trained) Models},
  year         = {2026},
  howpublished = {Manuscript and reproducibility repository},
  url          = {https://github.com/sbarai927/robust-tabular-ml}
}
```

Please also cite the original model, dataset, Optuna, and SHAP publications used by the study. Full references are provided in the manuscript PDFs.

---

## Contact

**Suvendu Barai**  
Communication Systems and Networks  
Technische Hochschule Köln  
Email: `suvendu.barai@smail.th-koeln.de`

---

## License

No top-level project license is currently declared. Dataset files and the vendored APT implementation remain subject to their respective original licences. Contact the author before redistributing or reusing repository content beyond normal academic inspection and citation.
