# Model Evaluation & Conclusion (STAT 411)

This document summarizes the **statistical modeling** results for both tracks:

- **Regression track:** predict continuous `popularity`.
- **Classification track:** predict `popular_bin = 1(popularity >= 49)` (top 25%).

All metrics reported are based on a fixed train/test split (`random_state=42`) and leakage-safe preprocessing (scalers inside CV pipelines).

---

## Regression track (continuous popularity)

### Models compared
Across each representation we compared:

- OLS (baseline linear regression)
- Ridge
- Lasso
- Elastic Net

(Regularization hyperparameters were tuned via CV.)

### Representations compared
- **All features** (original modeling matrix)
- **Feature selection:** RFE top-30 (fit on training split only)
- **Dimension reduction:** PCA
  - PCA continuous-only
  - PCA all-features

### Metrics used (regression)
- RMSE
- MAE
- $R^2$

### Results summary (from `cda_outputs/model_evaluation_comparison.csv`)
Key takeaways:

- **Best overall regression variant:** `all_features` (OLS / Ridge are essentially tied)
  - RMSE ≈ **17.01**, $R^2$ ≈ **0.325**
- **PCA all-features** performs **close** to all-features, with a small drop
  - RMSE ≈ **17.14**, $R^2$ ≈ **0.312**
- **RFE-30** underperforms the full feature set
  - RMSE ≈ **17.96**, $R^2$ ≈ **0.248**
- **PCA continuous-only** performs poorly (genre features matter a lot)
  - RMSE ≈ **20.45**, $R^2$ ≈ **0.021**

### Coefficient interpretation (are they logical?)
From the OLS coefficients on `all_features` (saved in `cda_outputs/model_evaluation_coefficients_all_features_ols.csv`):

- Many of the largest-magnitude coefficients are **genre indicators**, which is expected:
  genre is strongly associated with popularity in Spotify-like data.
- Continuous audio features (e.g., `valence`) also appear with non-trivial coefficients.

Interpretation notes:
- Because genre is one-hot encoded, each genre coefficient is interpreted as an **offset**
  relative to the reference set of genres (and in the presence of all the other features).
- Signs can sometimes be counterintuitive due to **correlation / confounding** between audio features and genre.

### Assumption checks
Diagnostics were generated in `modeling.py`:

- `plots/modeling_residuals_vs_fitted.png`
- `plots/modeling_qq.png`

What we look for:
- Residuals vs fitted: no strong curvature/pattern (linearity) and roughly constant spread (homoscedasticity).
- QQ plot: approximate straight-line behavior (residual normality; mostly matters for inference).

If mild violations exist:
- Regularization helps stabilize coefficients under multicollinearity.
- For strict inference/assumptions, we could consider transformations or robust regression,
  but those go beyond the “statistical models” requested.

**Conclusion (regression):** OLS is a strong baseline and regularization does not change performance much,
which suggests the feature set is informative but the achievable $R^2$ is moderate.

---

## Classification track (binarized popularity)

### Problem setup
We define:

$$\text{popular\_bin} = \mathbf{1}(\text{popularity} \ge 49)$$

49 is the **75th percentile** of popularity in this dataset, producing an imbalanced positive class (~25%).

### Models compared (statistical)
- Logistic Regression (L2), **unbalanced**
- Logistic Regression (L2), **cost-sensitive** with `class_weight='balanced'`
- (Optional) SMOTE + Logistic Regression (disabled in this environment due to package incompatibility)

### Representations compared
- All features
- Feature selection (RFE-30)
- PCA continuous-only
- PCA all-features

### Metrics used (imbalanced classification; rubric)
We do **not** rely on accuracy alone. We report:

- Sensitivity (Recall for the positive class)
- Specificity
- Cohen’s Kappa
- ROC-AUC
- F1-score

(Additionally logged: Precision and PR-AUC.)

### Results summary (from `cda_outputs/classification_metrics.csv`)
Key comparisons:

- Models with `class_weight='balanced'` achieve **much higher sensitivity** (catch more popular tracks),
  typically improving F1.
- Unbalanced models tend to have **high specificity** (few false positives) but low sensitivity.

**Final model choice (rubric-aligned):**
We select the best classification model by **highest F1**, tie-break by ROC-AUC and Kappa.
This selects:

- `pca_all_features :: logreg_classweight_l2`
  - ROC-AUC ≈ **0.8254**
  - F1 ≈ **0.5946**
  - Sensitivity ≈ **0.7889**
  - Specificity ≈ **0.7068**
  - Kappa ≈ **0.4077**

The confusion matrix plot for the selected/best model is saved at:
- `plots/classification_best_confusion_matrix.png`

### Coefficient interpretation (logistic regression)
Logistic regression coefficients represent the change in **log-odds** of being popular.

- With PCA representations, coefficients are on **PC components**, so they’re less interpretable
  in terms of original features.
- For interpretability, the `all_features` logistic model is the best place to discuss individual
  predictors (e.g., specific genres / audio features), even if PCA edges it out on ROC-AUC.

**Conclusion (classification):** The cost-sensitive logistic regression is the best rubric-aligned choice
because it balances false positives/false negatives better under class imbalance and improves F1 and sensitivity.

---

## “Kaç model denedik, hangisini seçtik?”

A small helper script `leaderboards.py` writes report tables:

- `cda_outputs/leaderboard_regression.csv`
- `cda_outputs/leaderboard_classification.csv`

Counts (from last run):
- Regression models evaluated: **16**
- Classification models evaluated: **8**

Selected models:
- Regression: `all_features :: ols`
- Classification: `pca_all_features :: logreg_classweight_l2`
