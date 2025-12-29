# STAT 411 — Data Mining Term Project Report (Fall 2024–2025)

**Department:** METU Statistics  
**Course:** STAT 411 — Data Mining  
**Topic:** Spotify-style track popularity modeling with a full data mining pipeline  

> This is the “master-level” version of the report: we document every major pipeline step (including artificial missingness + multiple imputation attempts), tabulate every saved CSV artifact, and interpret each figure with the statistical tests/metrics used.
>
> This report follows the course guideline document (`project info_2025.doc`) in sections A–G.
> All results referenced here are reproducible from the scripts in this workspace.

---

## A. Introduction & Data Source

### A.1 Problem statement
We study which audio and metadata features are associated with track **popularity**, and build **statistical predictive models** that can:

- predict **continuous popularity** (regression), and
- classify whether a track is **popular** (classification).

### A.2 Dataset
The project uses a Spotify-like tracks dataset (`dataset.csv` / `dataset1.csv`) with:

- a continuous dependent variable: `popularity`
- numeric audio features (e.g., danceability, loudness, valence, etc.)
- categorical variables such as track genre (converted to one-hot indicators)

**Model-ready dataset:** `dataset_knn_plus_genre_onehot_clean.csv`

### A.3 Analysis tracks (two targets)
We study the target variable in two complementary ways:

1) **Regression target:** continuous `popularity`.
2) **Classification target:** a binarized popularity indicator (defined in Section F.2) to satisfy the “imbalanced learning” requirement and to create an interpretable business-style question (“is this song in the top popularity tier?”).

This dual-track design matters: a model can have decent rank-separation (high ROC-AUC) while still having only moderate $R^2$.

---

## B. Data Cleaning & Preprocessing

This phase ensures a tidy dataset suitable for downstream analysis.

### B.1 Tidy operations
- Standardized and cleaned key identifiers (track/artist/album naming columns).
- Removed or handled duplicates and inconsistencies.

(Implemented through the project’s data preparation scripts, including `tidy_data.py` and related helpers.)

### B.2 Missing value analysis & imputation

This project’s preprocessing includes *two* types of missingness:

1) **“Natural” missingness already present** in the raw dataset(s).
2) **Artificial missingness introduced on purpose** to objectively compare imputation methods.

#### B.2.1 Artificial missingness injection (to evaluate imputers)
Implemented in `introduce_missingness.py`.

- For each numeric feature, a random fraction (5–10%) of entries is set to NA.
- The exact NA positions are saved to `na_indices.json` so the evaluation is reproducible.

Interpretation:
- This creates a controlled “ground-truth known” setting: we can hide observed values, impute them, and quantify how much each imputation method distorts downstream modeling.

#### B.2.2 Tidy cleaning (range checks + deduplication)
Implemented in `tidy_data.py`.

- Invalid ranges are set to NA (instead of silently clamping), because clamping would bias distributions.
- Duplicate track identifiers are removed (dedupe by `track_id`) so train/test splits don’t accidentally contain identical songs in both sets.

#### B.2.3 Imputation methods tried and the final choice
Implemented in `imputation.py`.

We explicitly tried multiple imputations:

- **Simple imputation** (mean/median style; fast baseline).
- **KNN imputation** (uses neighborhood structure; often better for correlated audio features).
- **KNN + merged genre information** (uses genre structure more effectively).
- **Iterative imputation** exists in code but was kept optional (commented) due to runtime/stability considerations.

Selection logic:
- We compared imputers using a simple, consistent downstream model (Ridge baseline) on sampled subsets.
- The chosen “production” dataset for modeling is:

**Model-ready dataset:** `dataset_knn_plus_genre_onehot_clean.csv`.

---

## C. Exploratory Data Analysis (EDA)

We explored distributions and relationships between variables.

### C.1 Univariate visuals
Generated plots (examples):

- Popularity distribution: `plots/hist_popularity.png`
- Numeric feature histograms: `plots/hist_all_numeric.png`
- Numeric feature boxplots: `plots/box_numeric.png`

Modernized versions are used when available:

- Numeric feature boxplots (modern): `plots/box_numeric_modern.png`

### C.2 Bivariate and multivariate visuals
- Correlation heatmaps:
  - Pearson: `plots/heatmap_corr_pearson_triangle_annot.png`
  - Spearman: `plots/heatmap_corr_spearman_triangle_annot.png`

- Popularity vs key numeric features (scatter):
- Popularity vs key numeric features (scatter, modernized where available):
  - `plots/scatter_popularity_vs_danceability_modern.png`
  - `plots/scatter_popularity_vs_loudness_modern.png`
  - `plots/scatter_popularity_vs_speechiness_modern.png`
  - `plots/scatter_popularity_vs_instrumentalness_modern.png`
  - `plots/scatter_popularity_vs_acousticness_modern.png`

- Popularity by genre (box + jitter):
- Popularity by genre (box + jitter, modern):
  - `plots/box_jitter_popularity_by_genre_top12_modern.png`

**EDA interpretation (high level):**
- Popularity is noisy and only moderately predictable from numeric audio features alone.
- Genre contributes meaningful signal; many genres show different popularity distributions.

Figure-by-figure interpretation (what to look for):

- `plots/hist_popularity.png`: Popularity is not symmetric and shows a heavy concentration around mid-to-low values. This hints that normality-based methods may be fragile, motivating rank-based correlation (Spearman) and non-parametric group tests later.
- `plots/box_numeric_modern.png`: Several audio features are strongly skewed (e.g., `instrumentalness`). This skewness often leads to outliers and non-normal residuals in linear models.
- `plots/heatmap_corr_spearman_triangle_annot.png`: Spearman correlations are generally small in magnitude, telling us we should expect only moderate $R^2$ from purely linear numeric-only relationships.
- `plots/scatter_popularity_vs_instrumentalness_modern.png`: A weak negative monotonic pattern: higher instrumentalness tends to associate with lower popularity; the pattern is subtle but consistent at large $N$.
- `plots/box_jitter_popularity_by_genre_top12_modern.png`: Genre distributions differ notably (shift + spread), suggesting genre acts like a “context variable” for popularity.

---

## D. Confirmatory Data Analysis (CDA)

We used CDA to formally test relationships suggested by EDA, while being explicit about assumptions.

- Numeric–popularity correlations summary (saved artifact):
  - `cda_outputs/cda_numeric_popularity_correlations.csv`

- Genre vs popularity summaries and post-hoc results (saved artifacts):
  - `cda_outputs/cda_genre_popularity_summary.csv`
  - `cda_outputs/cda_genre_posthoc_mannwhitney_holm.csv`

### D.1 Numeric features vs popularity (correlation tests)

Implemented in `CDA.py`.

Key assumption check (sampled Shapiro test):

- Popularity normality (Shapiro sampled): $p \approx 8.147\times10^{-30}$ → strong evidence against normality.

Decision rule used in code:

- If both variables look approximately normal (by sampled Shapiro), use **Pearson**.
- Otherwise, use **Spearman** (rank correlation).

Observed top associations (from current run output, top 10 by $|\rho|$):

| feature | method | corr | p-value |
|---|---|---:|---:|
| instrumentalness | Spearman | -0.118155 | 6.246390e-261 |
| loudness | Spearman | 0.066571 | 8.429443e-84 |
| speechiness | Spearman | -0.064301 | 2.680229e-78 |
| danceability | Spearman | 0.053627 | 5.487606e-55 |
| time_signature | Spearman | 0.037861 | 2.977034e-28 |
| mode | Spearman | -0.016370 | 1.894780e-06 |
| energy | Spearman | -0.013585 | 7.688894e-05 |
| liveness | Spearman | -0.012784 | 1.988189e-04 |
| duration_ms | Spearman | 0.011596 | 7.387218e-04 |
| tempo | Spearman | 0.009126 | 7.909273e-03 |

Interpretation:
- Even the strongest correlation is small in magnitude ($|\rho| \approx 0.12$). With $N\approx 84{,}699$, p-values become extremely small even for weak effects. So we focus on **effect size** (corr magnitude) rather than just significance.

### D.2 Popularity differences across genres (group tests)

Implemented in `CDA.py` using the **top 10 genres** to keep group comparisons interpretable.

Assumption checks:

- Genre-wise Shapiro tests are extremely small → non-normality within groups.
- Levene’s test for equal variances: stat $\approx 162.3$, $p \approx 6.066\times10^{-287}$ → heteroscedasticity.

Decision: assumptions not met → use **Kruskal–Wallis** (non-parametric ANOVA).

Test statistics (from current run output):

- Kruskal–Wallis: $H \approx 3407$, $p \approx 0$, effect size $\epsilon^2 \approx 0.3605$.

Interpretation:
- There is a large genre effect on popularity distribution (effect size is substantial). This supports including genre indicators in predictive modeling.

Post-hoc:

- Pairwise **Mann–Whitney U** tests with **Holm correction** are used to identify which genre pairs differ, controlling family-wise error.

### D.3 Tabulating every saved CDA CSV

Below are the project’s saved CDA tables (these are the same artifacts referenced above):

- `cda_outputs/cda_numeric_popularity_correlations.csv`
- `cda_outputs/cda_genre_popularity_summary.csv`
- `cda_outputs/cda_genre_posthoc_mannwhitney_holm.csv`

---

## E. Feature Engineering & Dimension Reduction

Implemented primarily in `future_engineering.py`.

### E.1 Feature engineering
We added a small number of interpretable engineered variables (Recitation-style), such as:
- duration transformations
- interaction-style features

### E.2 Feature selection
We included a wrapper-style feature selection method:

- **RFE (Recursive Feature Elimination)** selecting top-30 features (fit on training split only).

### E.3 Dimension reduction
**PCA (mandatory):** we created leakage-safe PCA representations exported as train/test CSVs.

- Continuous-only PCA
  - `cda_outputs/pca_continuous_only_train.csv`
  - `cda_outputs/pca_continuous_only_test.csv`
  - variance table: `cda_outputs/pca_continuous_only_variance.csv`
- All-features PCA
  - `cda_outputs/pca_all_features_train.csv`
  - `cda_outputs/pca_all_features_test.csv`
  - variance table: `cda_outputs/pca_all_features_variance.csv`

**PCA plots:**
- Cumulative variance: `plots/pca_cumulative_explained_variance.png`
- 2D PCA visualization: `plots/pca_2d_popularity.png`

**Non-linear DR (visualization):**
- t-SNE: `plots/tsne_2d_popularity.png`
- UMAP: `plots/umap_2d_popularity.png`

Interpretation:
- PCA with **all features** preserves predictive signal relatively well.
- PCA with **continuous-only** loses important information (notably genre).

---

## F. Statistical Modeling (Prediction / Classification)

We use only **statistical models** (not ML black-box models) and follow a consistent experimental setup:

- Train/Test split with fixed seed: `random_state=42`.
- CV method:
  - regression: KFold
  - classification: StratifiedKFold
- Leakage-safe pipelines (scaling inside CV).

### F.1 Regression: predict continuous popularity
Implemented in `modeling.py`.

**Models:**
- OLS baseline
- Ridge / Lasso / Elastic Net with hyperparameter tuning

**Diagnostics (assumptions):**
- Residuals vs fitted: `plots/modeling_residuals_vs_fitted.png`
- QQ plot: `plots/modeling_qq.png`

**Regression metrics (test set):**
(From `cda_outputs/modeling_metrics.csv`)

| Model | RMSE | MAE | R² |
|---|---:|---:|---:|
| OLS | 17.0091 | 12.1470 | 0.3252 |
| Ridge | 17.0092 | 12.1501 | 0.3252 |
| ElasticNet | 17.0098 | 12.1649 | 0.3251 |
| Lasso | 17.0099 | 12.1639 | 0.3251 |

Interpretation:
- Regularization does not change test performance much → baseline linear model is already near the best for this feature set.
- The achievable $R^2$ is moderate, suggesting popularity is partially predictable but driven by unobserved factors too.

### F.1.1 Tabulating every saved regression modeling/evaluation CSV

Saved artifacts in `cda_outputs/` used for this section:

- `modeling_metrics.csv`
- `model_evaluation_comparison.csv`
- `model_evaluation_coefficients_all_features_*.csv`
- `model_evaluation_coefficients_feature_selection_rfe_30_*.csv`
- `model_evaluation_coefficients_pca_continuous_only_*.csv`
- `model_evaluation_coefficients_pca_all_features_*.csv`
- `leaderboard_regression.csv`

### F.2 Classification: popular vs not popular (imbalanced)
Implemented in `modeling_classification.py`.

#### Label (binarization)
We define:

$$\text{popular\_bin} = \mathbf{1}(\text{popularity} \ge 49)$$

49 is the dataset’s 75th percentile, giving ~25% positives.

#### Imbalanced learning requirement
Rubric requires comparing imbalanced vs balanced approaches. We compare:

- logistic regression unbalanced
- logistic regression with `class_weight='balanced'` (cost-sensitive learning)

SMOTE was attempted but is disabled because the installed `imbalanced-learn` is not compatible with the current scikit-learn version in this environment.

#### Classification metrics (rubric)
We do **not** rely on accuracy alone; we report:

- Sensitivity (Recall)
- Specificity
- Kappa
- ROC-AUC
- F1

(We also report precision and PR-AUC as additional context.)

### F.2.1 Tabulating every saved classification CSV

Saved artifacts in `cda_outputs/` used for this section:

- `classification_metrics.csv`
- `leaderboard_classification.csv`

---

## G. Model Evaluation & Conclusion (what we would submit as “final choice”)

This project generates *many* candidate models. To stay rubric-aligned, we select one “final” model per task.

### G.1 Final regression model choice
From `cda_outputs/leaderboard_regression.csv`, the selected regression model is:

- **Representation:** all features
- **Estimator:** OLS

Interpretation:
- Regularized models did not materially improve test RMSE/MAE/$R^2$.
- Therefore, we prefer OLS for interpretability and simplicity.

### G.2 Final classification model choice (rubric-aligned)
From `cda_outputs/leaderboard_classification.csv`, the selected classification model is:

- **Representation:** PCA (all features)
- **Estimator:** Logistic Regression with class-weight balancing (L2)

Why this choice:
- The rubric emphasizes *imbalanced-aware* performance. We select by **F1 first**, then **ROC-AUC**, then **Kappa** (implemented in `leaderboards.py`).
- This model provides strong recall while keeping specificity reasonable, indicating good separation without collapsing to “predict popular for everyone”.

### G.3 Final notes on limitations and validity
- **Large $N$ caveat:** Many p-values are astronomically small; we interpret *effect sizes* and practical relevance.
- **Popularity is partially exogenous:** Marketing, playlists, artist fame, and time effects are not in the features; so moderate $R^2$ is expected.
- **Linear models are a constraint by design:** We intentionally used statistical models to satisfy the course requirement; stronger predictive performance is possible with non-linear ML, but that is out-of-scope.

## G. Model Evaluation & Conclusion

This section compares models across representations and interprets coefficients.

### G.1 Regression: representation comparison (all-features vs RFE vs PCA)
Regression evaluation is implemented in `model_eveluation.py`.

**Compared representations:**
- all_features
- feature_selection_rfe_30
- pca_continuous_only
- pca_all_features

**Compared models:** OLS, Ridge, Lasso, Elastic Net.

**Summary (best by RMSE within each representation):**
(From `cda_outputs/leaderboard_regression.csv`)

| Representation | Selected model | RMSE | MAE | R² |
|---|---|---:|---:|---:|
| all_features | OLS | 17.0091 | 12.1470 | 0.3252 |
| pca_all_features | Ridge | 17.1394 | 12.2801 | 0.3124 |
| feature_selection_rfe_30 | Ridge | 17.9595 | 13.8537 | 0.2476 |
| pca_continuous_only | OLS | 20.4541 | 17.0061 | 0.0208 |

**Regression conclusion:**
- Best overall is **all-features OLS**.
- PCA all-features is close but slightly worse.
- Continuous-only PCA is weak → categorical/genre info is critical.

### G.2 Regression coefficient interpretation
Coefficient tables were saved for each variant/model, e.g.:

- `cda_outputs/model_evaluation_coefficients_all_features_ols.csv`
- `cda_outputs/model_evaluation_coefficients_all_features_ridge.csv`
- and similarly for RFE/PCA variants.

Interpretation (what looks logical?):
- Many large coefficients are **genre indicators**; genre is expected to shift popularity baseline.
- Audio features (e.g., `valence`) appear among the larger coefficients.
- Some signs may appear counterintuitive due to **correlation/confounding** between genre and audio features.

Assumption validation:
- Residual diagnostic plots (Section F.1) are the primary checks for linearity/variance/normality.
- If minor heteroscedasticity exists, the main goal here is **prediction** (not strict inference). Regularization is used to stabilize coefficients under collinearity.

### G.3 Classification evaluation (balanced vs unbalanced)
Classification evaluation lives in `modeling_classification.py` and the table:

- `cda_outputs/classification_metrics.csv`

**Rubric metrics comparison:**
- Cost-sensitive (`class_weight='balanced'`) greatly improves **sensitivity** and usually improves **F1**.
- Unbalanced models have high **specificity** but can miss many positives (low sensitivity).

**Final classification model selection (rubric-aligned):**
Chosen by highest **F1**, tie-break by ROC-AUC then Kappa (see `leaderboards.py`).

Selected:
- `pca_all_features :: logreg_classweight_l2`
  - ROC-AUC ≈ 0.8254
  - F1 ≈ 0.5946
  - Sensitivity ≈ 0.7889
  - Specificity ≈ 0.7068
  - Kappa ≈ 0.4077

Confusion matrix plot:
- `plots/classification_best_confusion_matrix.png`

### Overall conclusion
- Regression provides a moderate-quality predictor ($R^2 \approx 0.33$), with OLS and regularized linear models performing similarly.
- Classification (binarized popularity) is meaningful under class imbalance; cost-sensitive logistic regression achieves better sensitivity and F1.
- Genre features are consistently important and appear prominently in coefficient interpretations.

---

## Reproducibility (how to re-run)

All scripts run under the workspace virtual environment:

- Regression: `modeling.py`, `model_eveluation.py`
- Classification: `modeling_classification.py`
- Leaderboards: `leaderboards.py`

Outputs:
- Tables/CSVs: `cda_outputs/`
- Figures: `plots/`
