import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge

# Enable IterativeImputer (experimental in sklearn)
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer


# -----------------------------------------------------------------------------
# 0) Read data
# -----------------------------------------------------------------------------
df = pd.read_csv("/Users/ismet/Desktop/411_Project/dataset_tidy.csv", na_values=["NA", ""])

# Set True if you want to regenerate the full-data KNN-imputed CSVs.
RUN_FULL_KNN_SAVE = False

print("Rows, Cols:", df.shape)
print("Target (popularity) missing:", int(df["popularity"].isna().sum()))

print("\nMissing values per column (top 15):")
miss = df.isna().sum().sort_values(ascending=False)
print(miss.head(15))

print("\nMissing rate per column (top 15):")
print((df.isna().mean().sort_values(ascending=False)).head(15))


# -----------------------------------------------------------------------------
# 1) Define X, y
# -----------------------------------------------------------------------------

# We will predict popularity.
# We drop identifier-like columns to reduce leakage/noise.
drop_cols = ["track_id", "track_name", "artists"]
drop_cols = [c for c in drop_cols if c in df.columns]

df2 = df.drop(columns=drop_cols)

# Drop rows where y is missing (we cannot train on those).
df2 = df2[df2["popularity"].notna()].copy()

y = df2["popularity"].astype(float)
X = df2.drop(columns=["popularity"])

# -----------------------------------------------------------------------------
# Add missingness indicators (Recitation idea: keep info that a value was missing)
# -----------------------------------------------------------------------------
miss_cols = [c for c in X.columns if X[c].isna().any()]
for c in miss_cols:
	X[c + "_missing"] = X[c].isna().astype(int)

print("\nMissingness indicators added:", len(miss_cols))

print("\nAfter dropping rows with missing popularity:")
print("Rows:", X.shape[0])


# -----------------------------------------------------------------------------
# 2) Train/test split (leakage-safe)
# -----------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
	X, y, test_size=0.2, random_state=411
)

# KNN and IterativeImputer can be expensive on large datasets.
X_train_small = X_train.sample(n=min(12000, len(X_train)), random_state=411)
y_train_small = y_train.loc[X_train_small.index]
X_test_small = X_test.sample(n=min(6000, len(X_test)), random_state=411)
y_test_small = y_test.loc[X_test_small.index]


# -----------------------------------------------------------------------------
# 3) Column types
# -----------------------------------------------------------------------------
num_cols = X.select_dtypes(include=["number"]).columns.tolist()
cat_cols = [c for c in X.columns if c not in num_cols]

print("\nNumeric columns:", len(num_cols))
print("Categorical columns:", len(cat_cols))


# -----------------------------------------------------------------------------
# Option A: track_genre rare categories -> "Other" (train-based)
# -----------------------------------------------------------------------------
min_count = 200
if "track_genre" in X.columns:
	vc = X_train["track_genre"].value_counts(dropna=False)
	rare = vc[vc < min_count].index
	print("\ntrack_genre unique (train):", int(X_train["track_genre"].nunique(dropna=True)))
	print(f"Rare track_genre (<{min_count}) in train:", int(len(rare)))

	def merge_genre(s):
		# ensure no NaN goes into OneHotEncoder (keep it as a string category)
		s = s.fillna("Missing").astype(str)
		return s.where(~s.isin(rare), other="Other")

	X_train["track_genre_merged"] = merge_genre(X_train["track_genre"].astype(str))
	X_test["track_genre_merged"] = merge_genre(X_test["track_genre"].astype(str))
	X_train_small["track_genre_merged"] = merge_genre(X_train_small["track_genre"].astype(str))
	X_test_small["track_genre_merged"] = merge_genre(X_test_small["track_genre"].astype(str))

	# Also add it to full X for saving a dataset version later
	X["track_genre_merged"] = merge_genre(X["track_genre"].astype(str))

	# We will evaluate two versions:
	#  - raw: track_genre one-hot
	#  - merged: track_genre_merged one-hot
	cat_cols_raw = [c for c in cat_cols if c != "album_name"]  # avoid huge one-hot
	cat_cols_merged = [c for c in cat_cols_raw if c != "track_genre"] + ["track_genre_merged"]
else:
	cat_cols_raw = [c for c in cat_cols if c != "album_name"]
	cat_cols_merged = cat_cols_raw


def run_model(preprocess, name):
	# Fast baseline model (works with sparse one-hot)
	model = Ridge(alpha=1.0, random_state=411)

	pipe = Pipeline([
		("prep", preprocess),
		("model", model),
	])

	print(f"\nFitting: {name}")
	pipe.fit(X_train_small, y_train_small)
	pred = pipe.predict(X_test_small)
	rmse = mean_squared_error(y_test_small, pred) ** 0.5
	print(f"\n{name} -> RMSE: {rmse:.4f}")
	return pipe, rmse


# -----------------------------------------------------------------------------
# 4) Imputation methods (Recitation-5 style)
# -----------------------------------------------------------------------------

# A) SimpleImputer (median for numeric, most_frequent for categorical)
preprocess_simple = ColumnTransformer(
	transformers=[
		("num", Pipeline([
			("imp", SimpleImputer(strategy="median")),
		]), num_cols),
		("cat", Pipeline([
			("imp", SimpleImputer(strategy="most_frequent")),
			("oh", OneHotEncoder(handle_unknown="ignore")),
		]), cat_cols),
	],
	remainder="drop",
)

pipe_simple, rmse_simple = run_model(preprocess_simple, "SimpleImputer (median/mode)")


# B) KNNImputer (numeric only)
# Note: KNN needs scaling. Also, use a smaller numeric feature set to make distances meaningful.
knn_num_cols = [c for c in num_cols if c not in ["__index__", "Unnamed: 0"]]
preprocess_knn = ColumnTransformer(
	transformers=[
		("num", Pipeline([
			("scale", StandardScaler()),
			("imp", KNNImputer(n_neighbors=5, weights="distance")),
		]), knn_num_cols),
		("cat", Pipeline([
			("imp", SimpleImputer(strategy="most_frequent")),
			("oh", OneHotEncoder(handle_unknown="ignore")),
		]), cat_cols_raw),
	],
	remainder="drop",
)

pipe_knn, rmse_knn = run_model(preprocess_knn, "KNNImputer (k=5, distance)")


# KNN numeric + merged track_genre one-hot
preprocess_knn_merged = ColumnTransformer(
	transformers=[
		("num", Pipeline([
			("scale", StandardScaler()),
			("imp", KNNImputer(n_neighbors=5, weights="distance")),
		]), knn_num_cols),
		("cat", Pipeline([
			("imp", SimpleImputer(strategy="most_frequent")),
			("oh", OneHotEncoder(handle_unknown="ignore")),
		]), cat_cols_merged),
	],
	remainder="drop",
)

pipe_knn_m, rmse_knn_m = run_model(preprocess_knn_merged, f"KNNImputer (k=5) + track_genre_merged (min_count={min_count})")


# C) IterativeImputer (MICE-like) numeric + categories handled separately
preprocess_mice = ColumnTransformer(
	transformers=[
		("num", Pipeline([
			("imp", IterativeImputer(random_state=100, max_iter=10)),
		]), num_cols),
		("cat", Pipeline([
			("imp", SimpleImputer(strategy="most_frequent")),
			("oh", OneHotEncoder(handle_unknown="ignore")),
		]), cat_cols),
	],
	remainder="drop",
)

# (Optional) IterativeImputer can be slow; keep it off for now.
# pipe_mice, rmse_mice = run_model(preprocess_mice, "IterativeImputer (MICE-like)")


# -----------------------------------------------------------------------------
# 5) Save KNN-imputed dataset (full data)
#    NOTE: KNN uses numeric columns only. We save a clean numeric CSV.
# -----------------------------------------------------------------------------
knn_numeric = Pipeline([
	("scale", StandardScaler()),
	("imp", KNNImputer(n_neighbors=5, weights="distance")),
])

X_num_full = X[knn_num_cols].copy()

# (A) Full-data fit (NOT leakage-safe, but gives a single imputed dataset)
X_num_full_imp_safe = None
if RUN_FULL_KNN_SAVE:
	print("\nKNN full-data imputation (numeric only) ...")
	X_num_full_imp = knn_numeric.fit_transform(X_num_full)
	df_knn_full = df2[["popularity"]].copy()
	df_knn_full[knn_num_cols] = X_num_full_imp

	out_full = "/Users/ismet/Desktop/411_Project/dataset_tidy_knn_imputed_full.csv"
	df_knn_full.to_csv(out_full, index=False)
	print("Saved:", out_full)

	# (B) Leakage-safe version: fit on train only, then transform all data
	print("\nKNN train-fit imputation (numeric only, leakage-safe) ...")
	knn_numeric.fit(X_train[knn_num_cols])
	X_num_full_imp_safe = knn_numeric.transform(X_num_full)
	df_knn_safe = df2[["popularity"]].copy()
	df_knn_safe[knn_num_cols] = X_num_full_imp_safe

	out_safe = "/Users/ismet/Desktop/411_Project/dataset_tidy_knn_imputed_trainfit.csv"
	df_knn_safe.to_csv(out_safe, index=False)
	print("Saved:", out_safe)


	# Save a lightweight dataset with merged genre (no one-hot, readable)
	if "track_genre_merged" in X.columns:
		out_genre = "/Users/ismet/Desktop/411_Project/dataset_tidy_knn_numeric_plus_genre_merged.csv"
		df_out = df2[["popularity"]].copy()
		df_out[knn_num_cols] = X_num_full_imp_safe
		df_out["track_genre_merged"] = X["track_genre_merged"].values
		df_out.to_csv(out_genre, index=False)
		print("Saved:", out_genre)


print("\nDone.")
