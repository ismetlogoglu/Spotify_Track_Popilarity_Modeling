"""STAT 411 Project — Section F: Statistical Modeling (Regression).

Implements a Recitation-8-style workflow:
- Train/Test split with a fixed seed
- Cross-validation using leakage-safe Pipelines
- Baseline Linear Regression + assumption diagnostics
- Ridge / Lasso / Elastic Net with hyperparameter tuning (CV)

Target:
	popularity (continuous regression).

Outputs:
	- Metrics CSV: cda_outputs/modeling_metrics.csv
	- Best params JSON: cda_outputs/modeling_best_params.json
	- Assumption plots: plots/modeling_residuals_vs_fitted.png, plots/modeling_qq.png

Notes:
	- This is the *regression* track. For the binarized-popularity classification track
	  (thresholded popularity), see modeling_classification.py.
"""

from __future__ import annotations

import json
import os
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import TransformedTargetRegressor
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


# -------------------------------
# Configuration
# -------------------------------

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_SPLITS = 5

PROJECT_ROOT = "/Users/ismet/Desktop/411_Project"

# Default to your modeling-ready dataset (KNN-imputed + genre one-hot)
DATA_PATH = os.path.join(PROJECT_ROOT, "dataset_knn_plus_genre_onehot_clean.csv")

TARGET = "popularity"

PLOTS_DIR = os.path.join(PROJECT_ROOT, "plots")
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "cda_outputs")

METRICS_OUT = os.path.join(OUTPUTS_DIR, "modeling_metrics.csv")
BEST_PARAMS_OUT = os.path.join(OUTPUTS_DIR, "modeling_best_params.json")

RESID_VS_FITTED_PLOT = os.path.join(PLOTS_DIR, "modeling_residuals_vs_fitted.png")
QQ_PLOT = os.path.join(PLOTS_DIR, "modeling_qq.png")


# -------------------------------
# Small “contract” helpers
# -------------------------------


@dataclass(frozen=True)
class SplitData:
	X_train: pd.DataFrame
	X_test: pd.DataFrame
	y_train: pd.Series
	y_test: pd.Series


def ensure_dirs() -> None:
	os.makedirs(PLOTS_DIR, exist_ok=True)
	os.makedirs(OUTPUTS_DIR, exist_ok=True)


def load_dataset(path: str) -> Tuple[pd.DataFrame, pd.Series]:
	df = pd.read_csv(path)
	if TARGET not in df.columns:
		raise ValueError(f"Expected target column '{TARGET}' not found in {path}.")

	# Drop ID-ish columns if present
	drop_cols = [c for c in ["track_id", "artists", "album_name", "track_name"] if c in df.columns]

	X = df.drop(columns=[TARGET] + drop_cols, errors="ignore")
	y = df[TARGET]

	# Ensure numeric
	X = X.apply(pd.to_numeric, errors="coerce")
	y = pd.to_numeric(y, errors="coerce")

	# Defensive NA handling
	mask = ~X.isna().any(axis=1) & ~y.isna()
	X = X.loc[mask].copy()
	y = y.loc[mask].copy()

	return X, y


def make_split(X: pd.DataFrame, y: pd.Series) -> SplitData:
	X_train, X_test, y_train, y_test = train_test_split(
		X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
	)
	return SplitData(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
	rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
	return {
		"rmse": rmse,
		"mae": float(mean_absolute_error(y_true, y_pred)),
		"r2": float(r2_score(y_true, y_pred)),
	}


def plot_residual_diagnostics(y_true: np.ndarray, y_pred: np.ndarray) -> None:
	resid = y_true - y_pred

	# Residuals vs Fitted
	plt.figure(figsize=(7, 5))
	sns.scatterplot(x=y_pred, y=resid, s=18)
	plt.axhline(0, color="black", linewidth=1)
	plt.xlabel("Fitted values")
	plt.ylabel("Residuals")
	plt.title("Residuals vs Fitted")
	plt.tight_layout()
	plt.savefig(RESID_VS_FITTED_PLOT, dpi=200)
	plt.close()

	# QQ plot (normality-ish check)
	# We avoid adding a hard dependency; use scipy if installed, else approximate.
	plt.figure(figsize=(6, 6))
	try:
		from scipy import stats  # type: ignore

		stats.probplot(resid, dist="norm", plot=plt)
		plt.title("QQ Plot of Residuals")
	except Exception:
		# Fallback: compare sorted residuals to normal quantiles via numpy
		z = np.sort((resid - resid.mean()) / (resid.std(ddof=1) + 1e-12))
		n = len(z)
		theo = np.sort(np.random.default_rng(RANDOM_STATE).normal(size=n))
		plt.scatter(theo, z, s=12)
		plt.xlabel("Theoretical quantiles (approx)")
		plt.ylabel("Standardized residual quantiles")
		plt.title("QQ Plot (approx; install scipy for exact)")
		lim = max(abs(theo).max(), abs(z).max())
		plt.plot([-lim, lim], [-lim, lim], color="black", linewidth=1)

	plt.tight_layout()
	plt.savefig(QQ_PLOT, dpi=200)
	plt.close()


def cv_and_test_evaluate(
	name: str,
	estimator: Any,
	split: SplitData,
	cv: Any,
	*,
	param_grid: Dict[str, List[Any]] | None = None,
) -> Tuple[Dict[str, Any], Dict[str, float]]:
	"""Fit model (with optional tuning) and return best params + test metrics."""

	if param_grid is None:
		model = estimator
		model.fit(split.X_train, split.y_train)
		y_pred = model.predict(split.X_test)
		metrics = regression_metrics(split.y_test.to_numpy(), np.asarray(y_pred))
		return {"model": name, "best_params": {}}, metrics

	search = GridSearchCV(
		estimator=estimator,
		param_grid=param_grid,
		scoring="neg_root_mean_squared_error",
		cv=cv,
		n_jobs=-1,
	)
	search.fit(split.X_train, split.y_train)
	best = search.best_estimator_
	y_pred = best.predict(split.X_test)
	metrics = regression_metrics(split.y_test.to_numpy(), np.asarray(y_pred))
	return {"model": name, "best_params": dict(search.best_params_)}, metrics


def main() -> None:
	ensure_dirs()

	print("\n=============================")
	print("SECTION F — STATISTICAL MODELING")
	print("=============================\n")

	print("DATA:", DATA_PATH)
	X, y = load_dataset(DATA_PATH)
	print("X shape:", X.shape, "| y shape:", y.shape)

	split = make_split(X, y)
	print("Train:", split.X_train.shape, "Test:", split.X_test.shape)

	# Recitation-8-style CV object
	cv = KFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)

	# -------------------------------
	# Baseline OLS
	# -------------------------------
	# Scaling isn't required for OLS mathematically, but we keep the pipeline pattern
	# consistent across models and leakage-safe for CV.
	ols = Pipeline(
		steps=[
			("scaler", StandardScaler()),
			("model", LinearRegression()),
		]
	)

	print("\n--- Baseline: Linear Regression (OLS) ---")
	_, ols_metrics = cv_and_test_evaluate("ols", ols, split, cv)
	print("Test metrics:", ols_metrics)

	# Assumption diagnostics on test residuals (visual)
	ols.fit(split.X_train, split.y_train)
	y_pred_ols = ols.predict(split.X_test)
	plot_residual_diagnostics(split.y_test.to_numpy(), np.asarray(y_pred_ols))
	print("Saved plots:")
	print(" -", RESID_VS_FITTED_PLOT)
	print(" -", QQ_PLOT)

	# Optional simple remedy hook: log-transform target if needed.
	# (We keep it lightweight and only run it if y is strictly positive.)
	remedy_metrics: Dict[str, float] | None = None
	if float(y.min()) > 0:
		print("\n--- Remedy candidate: log1p(y) via TransformedTargetRegressor ---")
		log_model = Pipeline(
			steps=[
				("scaler", StandardScaler()),
				(
					"model",
					TransformedTargetRegressor(
						regressor=LinearRegression(),
						func=np.log1p,
						inverse_func=np.expm1,
					),
				),
			]
		)
		_, remedy_metrics = cv_and_test_evaluate("ols_log1p_y", log_model, split, cv)
		print("Test metrics:", remedy_metrics)

	# -------------------------------
	# Regularization (tuned)
	# -------------------------------
	print("\n--- Regularization: Ridge / Lasso / ElasticNet (tuned with CV) ---")

	ridge_pipe = Pipeline([
		("scaler", StandardScaler()),
		("model", Ridge(random_state=RANDOM_STATE)),
	])
	ridge_grid = {"model__alpha": [0.01, 0.1, 1.0, 10.0, 100.0]}

	lasso_pipe = Pipeline([
		("scaler", StandardScaler()),
		("model", Lasso(max_iter=20000, random_state=RANDOM_STATE)),
	])
	lasso_grid = {"model__alpha": [0.0005, 0.001, 0.01, 0.1, 1.0]}

	enet_pipe = Pipeline([
		("scaler", StandardScaler()),
		("model", ElasticNet(max_iter=20000, random_state=RANDOM_STATE)),
	])
	enet_grid = {
		"model__alpha": [0.001, 0.01, 0.1, 1.0],
		"model__l1_ratio": [0.2, 0.5, 0.8],
	}

	best_params: Dict[str, Dict[str, Any]] = {}
	rows: List[Dict[str, Any]] = []

	# Collect baseline rows
	rows.append({"model": "ols", **ols_metrics})
	if remedy_metrics is not None:
		rows.append({"model": "ols_log1p_y", **remedy_metrics})

	# Regularized models
	ridge_bp, ridge_metrics = cv_and_test_evaluate("ridge", ridge_pipe, split, cv, param_grid=ridge_grid)
	best_params["ridge"] = ridge_bp["best_params"]
	rows.append({"model": "ridge", **ridge_metrics})

	lasso_bp, lasso_metrics = cv_and_test_evaluate("lasso", lasso_pipe, split, cv, param_grid=lasso_grid)
	best_params["lasso"] = lasso_bp["best_params"]
	rows.append({"model": "lasso", **lasso_metrics})

	enet_bp, enet_metrics = cv_and_test_evaluate(
		"elasticnet", enet_pipe, split, cv, param_grid=enet_grid
	)
	best_params["elasticnet"] = enet_bp["best_params"]
	rows.append({"model": "elasticnet", **enet_metrics})

	metrics_df = pd.DataFrame(rows).sort_values(by="rmse", ascending=True)
	metrics_df.to_csv(METRICS_OUT, index=False)
	with open(BEST_PARAMS_OUT, "w", encoding="utf-8") as f:
		json.dump(best_params, f, indent=2)

	print("\nSaved:")
	print(" -", METRICS_OUT)
	print(" -", BEST_PARAMS_OUT)
	print("\nModel comparison (sorted by RMSE):")
	print(metrics_df)


if __name__ == "__main__":
	main()
