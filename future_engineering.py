import warnings

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import mutual_info_regression
from sklearn.feature_selection import RFE

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


# =========================================================
# RECITATION 6 STYLE: FEATURE ENGINEERING + FEATURE SELECTION + DIM REDUCTION
#   Target: popularity
#   Dataset: dataset_EDA.csv (already aligned + contains categoricals)
# =========================================================

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")

# Default dataset for modeling: numeric + KNN-imputed + genre one-hot
DATA_PATH = "/Users/ismet/Desktop/411_Project/dataset_knn_plus_genre_onehot_clean.csv"
PLOTS_DIR = "/Users/ismet/Desktop/411_Project/plots"
OUTPUTS_DIR = "/Users/ismet/Desktop/411_Project/cda_outputs"
RANDOM_STATE = 0

# Flags controlling which feature groups we include
# - *_missing columns are missingness indicator features from earlier preprocessing
# - genre_* columns are one-hot encoded genre dummies
INCLUDE_MISSING_INDICATORS = True
INCLUDE_GENRE_ONEHOT = True

# If you're going to do statistical modeling on the PCA components,
# it is best to export PCA-transformed train/test datasets.
# IMPORTANT: to avoid leakage, PCA must be fit on the training split only.
SAVE_PCA_DATASETS = True

# Choose PCA dimensionality:
# - Option A: keep enough components to explain a chosen variance fraction
# - Option B: set a fixed number of components (set to None to disable)
PCA_VARIANCE_TO_KEEP = 0.95
PCA_N_COMPONENTS_FIXED = None


print("\n=============================")
print("FUTURE ENGINEERING (Recitation 6)")
print("=============================\n")

df = pd.read_csv(DATA_PATH)
print("PATH:", DATA_PATH)
print("SHAPE:", df.shape)
print("\nHEAD:")
print(df.head())


# ---------------------------------------------------------
# BASIC DATA PREP (lightweight)
# ---------------------------------------------------------

target = "popularity"

keep_id_cols = ["track_id", "artists", "album_name", "track_name"]

# In the one-hot dataset we expect no categoricals; keep this defensive for other inputs.
cat_cols = [c for c in ["track_genre", "track_genre_merged"] if c in df.columns]

if target not in df.columns:
	raise ValueError(f"Expected target column '{target}' not found.")


print("\nColumns:")
print(list(df.columns))


# Feature engineering examples (kept simple / explainable)
# - duration_min + beats_in_track (tempo * duration)
# - energy_x_danceability interaction
# - loudness_abs

if "duration_ms" in df.columns and "tempo" in df.columns:
	df["duration_min"] = df["duration_ms"] / 60000.0
	df["beats_in_track"] = df["tempo"] * df["duration_min"]

if "energy" in df.columns and "danceability" in df.columns:
	df["energy_x_danceability"] = df["energy"] * df["danceability"]

if "loudness" in df.columns:
	df["loudness_abs"] = df["loudness"].abs()


df_model = df.copy()

# If categoricals exist (not expected for onehot_clean), one-hot them.
if len(cat_cols) > 0:
	df_model = pd.get_dummies(df_model, columns=cat_cols, drop_first=True)

# Drop ID-like columns from modeling matrix
drop_cols = [c for c in keep_id_cols if c in df_model.columns]

X_all = df_model.drop(columns=[target] + drop_cols, errors="ignore")
y = df_model[target]

print("\nAfter feature engineering + one-hot:")
print("X_all shape:", X_all.shape)
print("y shape:", y.shape)


def ensure_output_dirs() -> None:
	import os

	os.makedirs(PLOTS_DIR, exist_ok=True)
	os.makedirs(OUTPUTS_DIR, exist_ok=True)


def build_feature_matrix(X_input: pd.DataFrame, *, include_genre: bool, include_missing: bool) -> pd.DataFrame:
	X_out = X_input.copy()

	genre_cols = [c for c in X_out.columns if c.startswith("genre_")]
	missing_cols = [c for c in X_out.columns if c.endswith("_missing")]

	if not include_genre and len(genre_cols) > 0:
		X_out = X_out.drop(columns=genre_cols)

	if not include_missing and len(missing_cols) > 0:
		X_out = X_out.drop(columns=missing_cols)

	# Ensure numeric
	X_out = X_out.apply(pd.to_numeric, errors="coerce")
	return X_out


X = build_feature_matrix(
	X_all,
	include_genre=INCLUDE_GENRE_ONEHOT,
	include_missing=INCLUDE_MISSING_INDICATORS,
)

na_total = int(X.isna().sum().sum())
print("Total NA in X:", na_total)
if na_total > 0:
	print("Dropping rows with NA (should be rare for dataset_EDA.csv)")
	keep_mask = ~X.isna().any(axis=1)
	X = X.loc[keep_mask].copy()
	y = y.loc[keep_mask].copy()


# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
	X, y, test_size=0.2, random_state=RANDOM_STATE
)

print("\nTrain/Test sizes:")
print("X_train:", X_train.shape, "X_test:", X_test.shape)


print("\nFeature flags:")
print("INCLUDE_GENRE_ONEHOT:", INCLUDE_GENRE_ONEHOT)
print("INCLUDE_MISSING_INDICATORS:", INCLUDE_MISSING_INDICATORS)

ensure_output_dirs()


# ---------------------------------------------------------
# FILTER METHODS
#   (1) Variance threshold (removes near-constant features)
#   (2) Correlation with target (Pearson)
#   (3) SelectKBest: f_regression and mutual_info
# ---------------------------------------------------------

print("\n=============================")
print("FILTER METHODS")
print("=============================")


# (1) Variance threshold
vt = VarianceThreshold(threshold=0.0)
vt.fit(X_train)
vt_mask = vt.get_support()
vt_selected = X_train.columns[vt_mask]
print("\nVarianceThreshold kept:", len(vt_selected), "/", X_train.shape[1])


# (2) Correlation with target
train_with_y = X_train.copy()
train_with_y[target] = y_train

corrs = train_with_y.corr(numeric_only=True)[target].drop(labels=[target], errors="ignore")
corrs = corrs.sort_values(key=lambda s: s.abs(), ascending=False)

print("\nTop 15 absolute correlations with popularity (train):")
print(corrs.head(15))


# (3a) SelectKBest (F-test)
k = min(30, X_train.shape[1])
skb_f = SelectKBest(score_func=f_regression, k=k)
skb_f.fit(X_train, y_train)

f_scores = pd.Series(skb_f.scores_, index=X_train.columns).sort_values(ascending=False)
print("\nSelectKBest (f_regression) top 15 scores:")
print(f_scores.head(15))


# (3b) SelectKBest (Mutual info)
skb_mi = SelectKBest(score_func=mutual_info_regression, k=k)
skb_mi.fit(X_train, y_train)

mi_scores = pd.Series(skb_mi.scores_, index=X_train.columns).sort_values(ascending=False)
print("\nSelectKBest (mutual_info_regression) top 15 scores:")
print(mi_scores.head(15))


# ---------------------------------------------------------
# WRAPPER METHOD: RFE
#   Use LinearRegression as in Recitation 6.
# ---------------------------------------------------------

print("\n=============================")
print("WRAPPER METHOD")
print("=============================")


rfe_k = min(20, X_train.shape[1])
lr = LinearRegression()
rfe = RFE(estimator=lr, n_features_to_select=rfe_k)
rfe.fit(X_train, y_train)

selected_features_rfe = X_train.columns[rfe.support_]
rfe_ranking = pd.DataFrame({"feature": X_train.columns, "ranking": rfe.ranking_}).sort_values(
	"ranking", ascending=True
)

print(f"\nRFE selected {len(selected_features_rfe)} features:")
print(list(selected_features_rfe)[:25])
print("\nRFE top 25 ranks:")
print(rfe_ranking.head(25))


# Quick model check with RFE set
X_train_rfe = X_train[selected_features_rfe]
X_test_rfe = X_test[selected_features_rfe]

lr.fit(X_train_rfe, y_train)
pred = lr.predict(X_test_rfe)
print("\nLinearRegression on RFE set:")
print("Test RMSE:", np.sqrt(mean_squared_error(y_test, pred)))
print("Test R2:", r2_score(y_test, pred))


# ---------------------------------------------------------
# EMBEDDED METHODS
#   (1) Lasso (sparse coefficients)
#   (2) Tree-based importance (RandomForest)
# ---------------------------------------------------------

print("\n=============================")
print("EMBEDDED METHODS")
print("=============================")


# Lasso needs scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

lasso = Lasso(alpha=0.001, random_state=RANDOM_STATE, max_iter=5000)
lasso.fit(X_train_scaled, y_train)

lasso_coef = pd.Series(lasso.coef_, index=X_train.columns)
lasso_nonzero = lasso_coef[lasso_coef != 0].sort_values(key=lambda s: s.abs(), ascending=False)

print("\nLasso non-zero coefficients:", len(lasso_nonzero))
print("Top 15 |coef|:")
print(lasso_nonzero.head(15))

pred_lasso = lasso.predict(X_test_scaled)
print("\nLasso model:")
print("Test RMSE:", np.sqrt(mean_squared_error(y_test, pred_lasso)))
print("Test R2:", r2_score(y_test, pred_lasso))


rf = RandomForestRegressor(
	n_estimators=300,
	random_state=RANDOM_STATE,
	n_jobs=-1,
	max_depth=None,
)
rf.fit(X_train, y_train)

rf_importance = pd.Series(rf.feature_importances_, index=X_train.columns).sort_values(ascending=False)
print("\nRandomForest feature importance top 15:")
print(rf_importance.head(15))

pred_rf = rf.predict(X_test)
print("\nRandomForest model:")
print("Test RMSE:", np.sqrt(mean_squared_error(y_test, pred_rf)))
print("Test R2:", r2_score(y_test, pred_rf))


# ---------------------------------------------------------
# DIMENSION REDUCTION
#   PCA (mandatory)
#   + t-SNE and UMAP (if available)
#   Run in two modes:
#     1) continuous-only  (exclude genre_* and *_missing)
#     2) all-features     (include genre_* and *_missing depending on flags)
# ---------------------------------------------------------

print("\n=============================")
print("DIMENSION REDUCTION")
print("=============================")


def run_dr_plots(X_for_dr: pd.DataFrame, y_for_dr: pd.Series, tag: str) -> None:
	"""Run PCA + t-SNE + UMAP and save plots with a tag."""

	# PCA should use standardized features
	X_scaled = StandardScaler().fit_transform(X_for_dr)

	pca_full = PCA(random_state=RANDOM_STATE)
	pca_full.fit(X_scaled)

	explained = pca_full.explained_variance_ratio_
	cum_explained = np.cumsum(explained)

	print(f"\n[{tag}] PCA explained variance ratio (first 10):")
	print(np.round(explained[:10], 4))

	n_90 = int(np.argmax(cum_explained >= 0.90) + 1)
	n_95 = int(np.argmax(cum_explained >= 0.95) + 1)
	print(f"[{tag}] Components to reach 90% variance: {n_90}")
	print(f"[{tag}] Components to reach 95% variance: {n_95}")

	plt.figure(figsize=(7, 4))
	plt.plot(np.arange(1, len(cum_explained) + 1), cum_explained, linewidth=1.5)
	plt.axhline(0.90, color="gray", linestyle="--", linewidth=1)
	plt.axhline(0.95, color="gray", linestyle=":", linewidth=1)
	plt.xlabel("# Components")
	plt.ylabel("Cumulative explained variance")
	plt.title(f"PCA cumulative explained variance ({tag})")
	plt.tight_layout()
	plt.savefig(f"{PLOTS_DIR}/pca_cumulative_explained_variance_{tag}.png", dpi=180)
	plt.close()

	pca_2 = PCA(n_components=2, random_state=RANDOM_STATE)
	X_pca2 = pca_2.fit_transform(X_scaled)

	plt.figure(figsize=(7, 5))
	sc = plt.scatter(X_pca2[:, 0], X_pca2[:, 1], c=y_for_dr.values, s=10, cmap="viridis", alpha=0.8)
	plt.colorbar(sc, label="popularity")
	plt.title(f"PCA (2D) colored by popularity ({tag})")
	plt.xlabel("PC1")
	plt.ylabel("PC2")
	plt.tight_layout()
	plt.savefig(f"{PLOTS_DIR}/pca_2d_popularity_{tag}.png", dpi=180)
	plt.close()

	# t-SNE for visualization (use PCA init to stabilize)
	tsne = TSNE(
		n_components=2,
		perplexity=30,
		learning_rate="auto",
		init="pca",
		random_state=RANDOM_STATE,
	)

	print(f"[{tag}] Running t-SNE (this can take a bit)...")
	X_tsne2 = tsne.fit_transform(X_scaled)

	plt.figure(figsize=(7, 5))
	sc = plt.scatter(X_tsne2[:, 0], X_tsne2[:, 1], c=y_for_dr.values, s=10, cmap="viridis", alpha=0.8)
	plt.colorbar(sc, label="popularity")
	plt.title(f"t-SNE (2D) colored by popularity ({tag})")
	plt.xlabel("t-SNE 1")
	plt.ylabel("t-SNE 2")
	plt.tight_layout()
	plt.savefig(f"{PLOTS_DIR}/tsne_2d_popularity_{tag}.png", dpi=180)
	plt.close()

	# UMAP (optional)
	try:
		import umap

		print(f"[{tag}] UMAP is available — running UMAP...")
		reducer = umap.UMAP(
			n_components=2,
			n_neighbors=30,
			min_dist=0.1,
			metric="euclidean",
			random_state=RANDOM_STATE,
		)
		X_umap2 = reducer.fit_transform(X_scaled)

		plt.figure(figsize=(7, 5))
		sc = plt.scatter(X_umap2[:, 0], X_umap2[:, 1], c=y_for_dr.values, s=10, cmap="viridis", alpha=0.8)
		plt.colorbar(sc, label="popularity")
		plt.title(f"UMAP (2D) colored by popularity ({tag})")
		plt.xlabel("UMAP 1")
		plt.ylabel("UMAP 2")
		plt.tight_layout()
		plt.savefig(f"{PLOTS_DIR}/umap_2d_popularity_{tag}.png", dpi=180)
		plt.close()

	except Exception as e:
		print(f"[{tag}] UMAP not available (skip). Reason:")
		print(repr(e))


def export_pca_train_test(
	*,
	X_full: pd.DataFrame,
	y_full: pd.Series,
	tag: str,
	include_genre: bool,
	include_missing: bool,
) -> None:
	"""Export PCA components for statistical modeling.

	This function:
	- builds a feature matrix from X_full using the same include/exclude rules
	- splits into train/test using the same RANDOM_STATE and test_size
	- fits scaler + PCA on train only
	- transforms train/test
	- writes CSVs and a small metadata CSV (variance curve)
	"""
	X_mode = build_feature_matrix(X_full, include_genre=include_genre, include_missing=include_missing)

	# Align with earlier NA handling policy (should be none for onehot_clean)
	na_total_mode = int(X_mode.isna().sum().sum())
	if na_total_mode > 0:
		keep_mask = ~X_mode.isna().any(axis=1)
		X_mode = X_mode.loc[keep_mask].copy()
		y_mode = y_full.loc[keep_mask].copy()
	else:
		y_mode = y_full

	X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(
		X_mode, y_mode, test_size=0.2, random_state=RANDOM_STATE
	)

	# Fit scaler on train only
	scaler = StandardScaler()
	X_train_scaled = scaler.fit_transform(X_train_m)
	X_test_scaled = scaler.transform(X_test_m)

	# Pick PCA dimensionality
	if PCA_N_COMPONENTS_FIXED is not None:
		pca = PCA(n_components=int(PCA_N_COMPONENTS_FIXED), random_state=RANDOM_STATE)
		pca.fit(X_train_scaled)
	else:
		pca_full = PCA(random_state=RANDOM_STATE)
		pca_full.fit(X_train_scaled)
		cum = np.cumsum(pca_full.explained_variance_ratio_)
		n_keep = int(np.argmax(cum >= float(PCA_VARIANCE_TO_KEEP)) + 1)
		pca = PCA(n_components=n_keep, random_state=RANDOM_STATE)
		pca.fit(X_train_scaled)

	X_train_pca = pca.transform(X_train_scaled)
	X_test_pca = pca.transform(X_test_scaled)

	col_names = [f"PC{i}" for i in range(1, X_train_pca.shape[1] + 1)]
	train_out = pd.DataFrame(X_train_pca, columns=col_names, index=X_train_m.index)
	test_out = pd.DataFrame(X_test_pca, columns=col_names, index=X_test_m.index)

	train_out[target] = y_train_m
	test_out[target] = y_test_m

	base = f"pca_{tag}"
	train_path = f"{OUTPUTS_DIR}/{base}_train.csv"
	test_path = f"{OUTPUTS_DIR}/{base}_test.csv"
	train_out.to_csv(train_path, index=False)
	test_out.to_csv(test_path, index=False)

	# Also export variance ratio for documentation
	var_df = pd.DataFrame(
		{
			"component": np.arange(1, len(pca.explained_variance_ratio_) + 1),
			"explained_variance_ratio": pca.explained_variance_ratio_,
			"cumulative_explained_variance_ratio": np.cumsum(pca.explained_variance_ratio_),
		}
	)
	var_path = f"{OUTPUTS_DIR}/{base}_variance.csv"
	var_df.to_csv(var_path, index=False)

	print(f"\n[{tag}] Exported PCA datasets for modeling:")
	print(" ", train_path)
	print(" ", test_path)
	print(" ", var_path)
	print(f"[{tag}] PCA components saved:", X_train_pca.shape[1])


# Build DR matrices
X_continuous_only = build_feature_matrix(X_all, include_genre=False, include_missing=False)
X_all_features = build_feature_matrix(
	X_all, include_genre=INCLUDE_GENRE_ONEHOT, include_missing=INCLUDE_MISSING_INDICATORS
)

print("\nDR matrices:")
print("continuous-only shape:", X_continuous_only.shape)
print("all-features shape   :", X_all_features.shape)

run_dr_plots(X_continuous_only, y, tag="continuous_only")
run_dr_plots(X_all_features, y, tag="all_features")


if SAVE_PCA_DATASETS:
	print("\n=============================")
	print("EXPORT PCA DATASETS (LEAKAGE-SAFE)")
	print("=============================")

	export_pca_train_test(
		X_full=X_all,
		y_full=y,
		tag="continuous_only",
		include_genre=False,
		include_missing=False,
	)

	export_pca_train_test(
		X_full=X_all,
		y_full=y,
		tag="all_features",
		include_genre=INCLUDE_GENRE_ONEHOT,
		include_missing=INCLUDE_MISSING_INDICATORS,
	)


print("\nSaved DR plots to:", PLOTS_DIR)
print("(files are tagged with _continuous_only or _all_features)")

print("\nDone.")
