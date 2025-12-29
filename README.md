# STAT 411 Project — Modeling (Regression + Classification)

This repo contains a **two-track modeling workflow** for the target **`popularity`**:

- **Regression track (continuous target):** predict numeric `popularity`.
- **Classification track (binarized target):** predict whether a track is **popular**.

The earlier data pipeline (missingness → tidy → KNN imputation → rare-genre handling → one-hot) produces the main modeling dataset:

- `dataset_knn_plus_genre_onehot_clean.csv`

## Quick map of scripts

### Regression (continuous `popularity`)

- `modeling.py`
  - Section F regression modeling (OLS baseline + diagnostics + Ridge/Lasso/ElasticNet tuning).
  - Writes:
    - `cda_outputs/modeling_metrics.csv`
    - `cda_outputs/modeling_best_params.json`
    - Residual plots under `plots/`

- `model_eveluation.py`
  - Model Evaluation & Conclusion for regression.
  - Compares feature representations:
    - all features
    - RFE feature selection (fit on train only)
    - leakage-safe PCA exports (continuous-only, and all-features)
  - Writes:
    - `cda_outputs/model_evaluation_comparison.csv`
    - `cda_outputs/model_evaluation_best_params.json`
    - `cda_outputs/model_evaluation_coefficients_*.csv`

### Classification (binarized `popularity`)

- `modeling_classification.py`
  - Section F classification extension.
  - Defines:

    $$\text{popular\_bin} = \mathbf{1}(\text{popularity} \ge 49)$$

    where 49 is the dataset’s **75th percentile** (≈ top 25%), giving a meaningful “popular” class.
  - Compares:
    - unbalanced logistic regression
    - cost-sensitive logistic regression (`class_weight='balanced'`)
    - optional SMOTE (disabled automatically if the environment can’t import it)
  - Writes:
    - `cda_outputs/classification_metrics.csv`
    - `cda_outputs/classification_best_params.json`
    - `plots/classification_best_confusion_matrix.png`

### Other useful scripts

- `future_engineering.py`
  - Recitation-style feature engineering + (optional) selection + DR.
  - Exports leakage-safe PCA train/test CSVs under `cda_outputs/`.

- `CDA.py`
  - Confirmatory data analysis. Configured to **print to terminal only** (no CSV outputs).

- `readd_popularity.py`
  - Restores missing `popularity` values in `dataset.csv` from `dataset1.csv` and writes `dataset_popularity_restored.csv`.

## Outputs

Most model artifacts are written to:

- `cda_outputs/` (metrics tables, best params, coefficient tables, PCA exports)
- `plots/` (diagnostic plots, confusion matrix)

## Notes on methodology

- All preprocessing (e.g., scaling) is done inside a `Pipeline` during CV to avoid leakage.
- For classification, metrics emphasize **ROC-AUC / F1 / sensitivity / specificity** (accuracy alone can mislead under class imbalance).

## Suggested run order

A typical modeling run:

1) (If needed) run your data pipeline scripts to regenerate `dataset_knn_plus_genre_onehot_clean.csv`
2) Run regression:
   - `modeling.py`
   - `model_eveluation.py`
3) Run classification:
   - `modeling_classification.py`
