"""STAT 411 Project — Model Evaluation & Conclusion (Regression / statistical models).

What this script does (rubric-aligned):
1) Compares multiple *statistical* regression models:
   - Baseline OLS
   - Ridge / Lasso / ElasticNet (with hyperparameter tuning)

2) Compares feature representations:
   A) All features (original): dataset_knn_plus_genre_onehot_clean.csv
   B) Feature selection (RFE top-k): uses sklearn RFE on training split only
   C) Dimension reduction (PCA): uses pre-exported leakage-safe PCA train/test CSVs

3) Metrics (regression): RMSE, MAE, R^2
4) Interprets coefficients: saves coefficient tables for the best linear models.

Classification note:
	The binarized-popularity *classification* evaluation lives in
	modeling_classification.py and the summary CSV cda_outputs/classification_metrics.csv
	(sensitivity, specificity, kappa, ROC-AUC, F1, etc.).

Outputs:
   - cda_outputs/model_evaluation_comparison.csv
   - cda_outputs/model_evaluation_best_params.json
   - cda_outputs/model_evaluation_coefficients_*.csv

Note:
   PCA all-features train/test is large (~155MB). We still load it with pandas
   from disk. This is fine in a python run, but opening in the editor may fail.
"""

from __future__ import annotations

import json
import os
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import seaborn as sns
from sklearn.feature_selection import RFE
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


warnings.filterwarnings("ignore")
sns.set_style("whitegrid")


PROJECT_ROOT = "/Users/ismet/Desktop/411_Project"
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "cda_outputs")

DATA_ORIGINAL = os.path.join(PROJECT_ROOT, "dataset_knn_plus_genre_onehot_clean.csv")

PCA_CONT_TRAIN = os.path.join(OUTPUTS_DIR, "pca_continuous_only_train.csv")
PCA_CONT_TEST = os.path.join(OUTPUTS_DIR, "pca_continuous_only_test.csv")

PCA_ALL_TRAIN = os.path.join(OUTPUTS_DIR, "pca_all_features_train.csv")
PCA_ALL_TEST = os.path.join(OUTPUTS_DIR, "pca_all_features_test.csv")

TARGET = "popularity"

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_SPLITS = 5

RFE_K = 30

OUT_COMPARISON = os.path.join(OUTPUTS_DIR, "model_evaluation_comparison.csv")
OUT_PARAMS = os.path.join(OUTPUTS_DIR, "model_evaluation_best_params.json")


@dataclass(frozen=True)
class SplitData:
	X_train: pd.DataFrame
	X_test: pd.DataFrame
	y_train: pd.Series
	y_test: pd.Series


def ensure_dirs() -> None:
	os.makedirs(OUTPUTS_DIR, exist_ok=True)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
	return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
	return {
		"rmse": rmse(y_true, y_pred),
		"mae": float(mean_absolute_error(y_true, y_pred)),
		"r2": float(r2_score(y_true, y_pred)),
	}


def load_original_all_features(path: str) -> Tuple[pd.DataFrame, pd.Series]:
	df = pd.read_csv(path)
	if TARGET not in df.columns:
		raise ValueError(f"Expected target column '{TARGET}' not found in {path}.")

	drop_cols = [c for c in ["track_id", "artists", "album_name", "track_name"] if c in df.columns]
	X = df.drop(columns=[TARGET] + drop_cols, errors="ignore")
	y = df[TARGET]

	X = X.apply(pd.to_numeric, errors="coerce")
	y = pd.to_numeric(y, errors="coerce")

	mask = ~X.isna().any(axis=1) & ~y.isna()
	X = X.loc[mask].copy()
	y = y.loc[mask].copy()

	return X, y


def make_split(X: pd.DataFrame, y: pd.Series) -> SplitData:
	X_train, X_test, y_train, y_test = train_test_split(
		X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
	)
	return SplitData(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)


def load_pca_train_test(train_path: str, test_path: str) -> SplitData:
	train = pd.read_csv(train_path)
	test = pd.read_csv(test_path)

	if TARGET not in train.columns or TARGET not in test.columns:
		raise ValueError("PCA train/test CSVs must include target column 'popularity'.")

	y_train = train[TARGET]
	y_test = test[TARGET]
	X_train = train.drop(columns=[TARGET])
	X_test = test.drop(columns=[TARGET])

	# Ensure numeric
	X_train = X_train.apply(pd.to_numeric, errors="coerce")
	X_test = X_test.apply(pd.to_numeric, errors="coerce")
	y_train = pd.to_numeric(y_train, errors="coerce")
	y_test = pd.to_numeric(y_test, errors="coerce")

	return SplitData(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)


def tune_and_test(
	*,
	model_name: str,
	pipe: Pipeline,
	grid: Optional[Dict[str, List[Any]]],
	split: SplitData,
	cv: Any,
) -> Tuple[Dict[str, Any], Dict[str, float], Any]:
	"""Return (best_params, test_metrics, fitted_best_estimator)."""
	if grid is None:
		pipe.fit(split.X_train, split.y_train)
		pred = pipe.predict(split.X_test)
		return {}, regression_metrics(split.y_test.to_numpy(), np.asarray(pred)), pipe

	search = GridSearchCV(
		estimator=pipe,
		param_grid=grid,
		scoring="neg_root_mean_squared_error",
		cv=cv,
		n_jobs=-1,
	)
	search.fit(split.X_train, split.y_train)
	best = search.best_estimator_
	pred = best.predict(split.X_test)
	return dict(search.best_params_), regression_metrics(split.y_test.to_numpy(), np.asarray(pred)), best


def extract_coefficients(estimator: Pipeline, feature_names: List[str]) -> pd.DataFrame:
	# Pipeline steps: scaler -> model
	model = estimator.named_steps.get("model")
	if model is None or not hasattr(model, "coef_"):
		raise ValueError("Estimator does not expose coefficients.")

	coef = np.asarray(getattr(model, "coef_"))
	coef = coef.reshape(-1)
	if len(coef) != len(feature_names):
		# Fallback: just index them; should not happen for our linear models
		feature_names = [f"x{i}" for i in range(len(coef))]

	out = pd.DataFrame({"feature": feature_names, "coef": coef})
	out["abs_coef"] = out["coef"].abs()
	return out.sort_values("abs_coef", ascending=False)


def main() -> None:
	ensure_dirs()
	cv = KFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)

	results: List[Dict[str, Any]] = []
	best_params: Dict[str, Dict[str, Any]] = {}

	# -------------------------------
	# Variant A: All features (original)
	# -------------------------------
	X_all, y_all = load_original_all_features(DATA_ORIGINAL)
	split_all = make_split(X_all, y_all)
	feature_names_all = list(split_all.X_train.columns)

	# Feature selection: fit RFE on training only, then transform train/test
	rfe_k = min(RFE_K, split_all.X_train.shape[1])
	rfe = RFE(estimator=LinearRegression(), n_features_to_select=rfe_k)
	rfe.fit(split_all.X_train, split_all.y_train)
	selected = list(split_all.X_train.columns[rfe.support_])
	split_fs = SplitData(
		X_train=split_all.X_train[selected].copy(),
		X_test=split_all.X_test[selected].copy(),
		y_train=split_all.y_train,
		y_test=split_all.y_test,
	)

	variants: List[Tuple[str, SplitData, List[str]]] = [
		("all_features", split_all, feature_names_all),
		(f"feature_selection_rfe_{len(selected)}", split_fs, selected),
	]

	# -------------------------------
	# Variant C: PCA (pre-exported)
	# -------------------------------
	# Note: PCA train/test already includes y and is leakage-safe.
	# For PCA components, scaling is typically not needed (they're already scaled-ish)
	# but we keep a StandardScaler in the pipeline for consistency.
	split_pca_cont = load_pca_train_test(PCA_CONT_TRAIN, PCA_CONT_TEST)
	pca_cont_features = list(split_pca_cont.X_train.columns)
	variants.append(("pca_continuous_only", split_pca_cont, pca_cont_features))

	# PCA all features is large; we still load it from disk
	split_pca_all = load_pca_train_test(PCA_ALL_TRAIN, PCA_ALL_TEST)
	pca_all_features = list(split_pca_all.X_train.columns)
	variants.append(("pca_all_features", split_pca_all, pca_all_features))

	# -------------------------------
	# Models to compare (statistical)
	# -------------------------------
	model_specs: List[Tuple[str, Pipeline, Optional[Dict[str, List[Any]]]]] = []

	ols = Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())])
	model_specs.append(("ols", ols, None))

	ridge = Pipeline([("scaler", StandardScaler()), ("model", Ridge(random_state=RANDOM_STATE))])
	ridge_grid = {"model__alpha": [0.01, 0.1, 1.0, 10.0, 100.0]}
	model_specs.append(("ridge", ridge, ridge_grid))

	lasso = Pipeline([
		("scaler", StandardScaler()),
		("model", Lasso(max_iter=20000, random_state=RANDOM_STATE)),
	])
	lasso_grid = {"model__alpha": [0.0005, 0.001, 0.01, 0.1, 1.0]}
	model_specs.append(("lasso", lasso, lasso_grid))

	enet = Pipeline([
		("scaler", StandardScaler()),
		("model", ElasticNet(max_iter=20000, random_state=RANDOM_STATE)),
	])
	enet_grid = {"model__alpha": [0.001, 0.01, 0.1, 1.0], "model__l1_ratio": [0.2, 0.5, 0.8]}
	model_specs.append(("elasticnet", enet, enet_grid))

	# -------------------------------
	# Run evaluation
	# -------------------------------
	for variant_name, split, feat_names in variants:
		for model_name, pipe, grid in model_specs:
			params, metrics, fitted = tune_and_test(
				model_name=model_name,
				pipe=pipe,
				grid=grid,
				split=split,
				cv=cv,
			)

			key = f"{variant_name}::{model_name}"
			if len(params) > 0:
				best_params[key] = params

			results.append(
				{
					"variant": variant_name,
					"model": model_name,
					**metrics,
				}
			)

			# Save coefficient tables for interpretability (where it makes sense)
			if model_name in {"ols", "ridge", "lasso", "elasticnet"}:
				try:
					coef_df = extract_coefficients(fitted, feat_names)
					out_coef = os.path.join(
						OUTPUTS_DIR, f"model_evaluation_coefficients_{variant_name}_{model_name}.csv"
					)
					coef_df.to_csv(out_coef, index=False)
				except Exception:
					# PCA / other mismatch shouldn't crash the evaluation
					pass

	comp = pd.DataFrame(results).sort_values(["variant", "rmse"], ascending=[True, True])
	comp.to_csv(OUT_COMPARISON, index=False)

	with open(OUT_PARAMS, "w", encoding="utf-8") as f:
		json.dump(best_params, f, indent=2)

	print("\nSaved:")
	print(" -", OUT_COMPARISON)
	print(" -", OUT_PARAMS)
	print("\nTop results by variant (lowest RMSE):")
	print(comp.groupby("variant").head(3))


if __name__ == "__main__":
	main()
