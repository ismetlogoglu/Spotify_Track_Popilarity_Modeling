"""STAT 411 Project — Section F + Model Evaluation (Classification; statistical models only)

We binarize popularity into a meaningful "popular" label using the top 25% threshold:

    popular = 1 if popularity >= 49  (q75 in this dataset)

This script follows Recitation-8 logic:
- Train/Test split with a fixed seed + stratify
- StratifiedKFold cross-validation
- Leakage-safe Pipelines (scaler inside CV)
- Logistic Regression baseline + regularized variants (L2/L1/ElasticNet via C + penalty)
- Imbalanced handling comparison:
    A) Unbalanced baseline
    B) class_weight='balanced'
    C) SMOTE (pipeline)  [requires imbalanced-learn]

Outputs (written under cda_outputs/):
- classification_metrics.csv (comparison table)
- classification_best_params.json

Notes (rubric):
- For imbalanced classification, we do NOT rely on accuracy alone. We report
    Sensitivity, Specificity, Kappa, ROC-AUC and F1 (plus Precision and PR-AUC for extra context).
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

from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


warnings.filterwarnings("ignore")

PROJECT_ROOT = "/Users/ismet/Desktop/411_Project"
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "cda_outputs")
PLOTS_DIR = os.path.join(PROJECT_ROOT, "plots")
DATA_PATH = os.path.join(PROJECT_ROOT, "dataset_knn_plus_genre_onehot_clean.csv")

# Leak-safe PCA train/test datasets exported earlier (fit on train only)
PCA_CONT_TRAIN = os.path.join(OUTPUTS_DIR, "pca_continuous_only_train.csv")
PCA_CONT_TEST = os.path.join(OUTPUTS_DIR, "pca_continuous_only_test.csv")
PCA_ALL_TRAIN = os.path.join(OUTPUTS_DIR, "pca_all_features_train.csv")
PCA_ALL_TEST = os.path.join(OUTPUTS_DIR, "pca_all_features_test.csv")

TARGET_REG = "popularity"
THRESHOLD = 49  # q75
TARGET_BIN = "popular_bin"

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_SPLITS = 5

OUT_METRICS = os.path.join(OUTPUTS_DIR, "classification_metrics.csv")
OUT_PARAMS = os.path.join(OUTPUTS_DIR, "classification_best_params.json")

OUT_CM_PLOT = os.path.join(PLOTS_DIR, "classification_best_confusion_matrix.png")

RFE_K = 30


@dataclass(frozen=True)
class SplitData:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series


def ensure_dirs() -> None:
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)


def load_data(path: str) -> Tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(path)
    if TARGET_REG not in df.columns:
        raise ValueError(f"Expected '{TARGET_REG}' not found in {path}")

    # Binarize
    df[TARGET_BIN] = (df[TARGET_REG].astype(float) >= THRESHOLD).astype(int)

    # IMPORTANT: drop BOTH the original continuous target AND any derived/duplicate targets
    # to avoid target leakage.
    leak_cols = [c for c in df.columns if c.lower() in {"popularity", "popular", "popular_bin", "is_popular"}]
    drop_cols = [c for c in ["track_id", "artists", "album_name", "track_name"] if c in df.columns]
    drop_cols = list(dict.fromkeys(drop_cols + leak_cols))  # keep order, de-dupe
    X = df.drop(columns=drop_cols, errors="ignore")
    y = df[TARGET_BIN]

    X = X.apply(pd.to_numeric, errors="coerce")
    y = pd.to_numeric(y, errors="coerce")

    mask = ~X.isna().any(axis=1) & ~y.isna()
    X = X.loc[mask].copy()
    y = y.loc[mask].astype(int).copy()

    return X, y


def load_pca_train_test(train_path: str, test_path: str) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Load PCA train/test splits from disk and binarize popularity consistently."""
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    if TARGET_REG not in train.columns or TARGET_REG not in test.columns:
        raise ValueError("PCA train/test CSVs must include the original 'popularity' column.")

    y_train = (train[TARGET_REG].astype(float) >= THRESHOLD).astype(int)
    y_test = (test[TARGET_REG].astype(float) >= THRESHOLD).astype(int)
    X_train = train.drop(columns=[TARGET_REG], errors="ignore")
    X_test = test.drop(columns=[TARGET_REG], errors="ignore")

    X_train = X_train.apply(pd.to_numeric, errors="coerce")
    X_test = X_test.apply(pd.to_numeric, errors="coerce")
    mask_tr = ~X_train.isna().any(axis=1) & ~y_train.isna()
    mask_te = ~X_test.isna().any(axis=1) & ~y_test.isna()

    return (
        X_train.loc[mask_tr].copy(),
        y_train.loc[mask_tr].astype(int).copy(),
        X_test.loc[mask_te].copy(),
        y_test.loc[mask_te].astype(int).copy(),
    )


def make_split(X: pd.DataFrame, y: pd.Series) -> SplitData:
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    return SplitData(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> Dict[str, float]:
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan

    out = {
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "pr_auc": float(average_precision_score(y_true, y_proba)),
        "f1": float(f1_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "kappa": float(cohen_kappa_score(y_true, y_pred)),
    }
    return out


def evaluate_model(
    *,
    name: str,
    estimator: Any,
    param_grid: Dict[str, List[Any]] | None,
    split: SplitData,
    cv: Any,
) -> Tuple[Dict[str, Any], Dict[str, float]]:
    if param_grid is None:
        estimator.fit(split.X_train, split.y_train)
        proba = estimator.predict_proba(split.X_test)[:, 1]
        pred = (proba >= 0.5).astype(int)
        return {"model": name, "best_params": {}}, classification_metrics(split.y_test.to_numpy(), pred, proba)

    search = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1,
    )
    search.fit(split.X_train, split.y_train)
    best = search.best_estimator_

    proba = best.predict_proba(split.X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)

    return {"model": name, "best_params": dict(search.best_params_)}, classification_metrics(split.y_test.to_numpy(), pred, proba)


def make_split_from_matrices(
    X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series
) -> SplitData:
    return SplitData(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)


def main() -> None:
    ensure_dirs()

    print("\n=============================")
    print("CLASSIFICATION MODELING (Popularity binarized)")
    print("=============================\n")

    X, y = load_data(DATA_PATH)
    pos_rate = float(y.mean())
    print("DATA:", DATA_PATH)
    print("Threshold:", THRESHOLD, "=> positive rate:", f"{pos_rate*100:.2f}%")
    print("X shape:", X.shape)

    split_all = make_split(X, y)
    print("All-features Train/Test:", split_all.X_train.shape, split_all.X_test.shape)

    # Feature-selection split (RFE on train only)
    rfe_k = min(RFE_K, split_all.X_train.shape[1])
    rfe = RFE(
        estimator=LogisticRegression(max_iter=5000, solver="liblinear"),
        n_features_to_select=rfe_k,
    )
    rfe.fit(split_all.X_train, split_all.y_train)
    selected = list(split_all.X_train.columns[rfe.support_])
    split_fs = SplitData(
        X_train=split_all.X_train[selected].copy(),
        X_test=split_all.X_test[selected].copy(),
        y_train=split_all.y_train,
        y_test=split_all.y_test,
    )
    print(f"Feature-selection (RFE) kept {len(selected)} features")

    # PCA splits (pre-exported) — already leak-safe
    Xtr_c, ytr_c, Xte_c, yte_c = load_pca_train_test(PCA_CONT_TRAIN, PCA_CONT_TEST)
    split_pca_cont = make_split_from_matrices(Xtr_c, ytr_c, Xte_c, yte_c)
    print("PCA continuous-only Train/Test:", split_pca_cont.X_train.shape, split_pca_cont.X_test.shape)

    Xtr_a, ytr_a, Xte_a, yte_a = load_pca_train_test(PCA_ALL_TRAIN, PCA_ALL_TEST)
    split_pca_all = make_split_from_matrices(Xtr_a, ytr_a, Xte_a, yte_a)
    print("PCA all-features Train/Test:", split_pca_all.X_train.shape, split_pca_all.X_test.shape)

    variants: List[Tuple[str, SplitData]] = [
        ("all_features", split_all),
        ("feature_selection_rfe_30", split_fs),
        ("pca_continuous_only", split_pca_cont),
        ("pca_all_features", split_pca_all),
    ]

    cv = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    rows: List[Dict[str, Any]] = []
    best_params: Dict[str, Dict[str, Any]] = {}

    # Compare a minimal, rubric-friendly set:
    # - Unbalanced logistic
    # - class_weight balanced logistic
    model_specs: List[Tuple[str, Any, Dict[str, List[Any]]]] = []
    model_specs.append(
        (
            "logreg_unbalanced_l2",
            Pipeline([
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=5000, solver="liblinear")),
            ]),
            {"model__C": [0.01, 0.1, 1.0, 10.0]},
        )
    )
    model_specs.append(
        (
            "logreg_classweight_l2",
            Pipeline([
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=5000, solver="liblinear", class_weight="balanced")),
            ]),
            {"model__C": [0.01, 0.1, 1.0, 10.0]},
        )
    )

    # Optional SMOTE version (if compatible)
    smote_spec: Tuple[str, Any, Dict[str, List[Any]]] | None = None
    try:
        from imblearn.over_sampling import SMOTE
        from imblearn.pipeline import Pipeline as ImbPipeline

        smote_spec = (
            "logreg_smote_l2",
            ImbPipeline([
                ("smote", SMOTE(random_state=RANDOM_STATE)),
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=5000, solver="liblinear")),
            ]),
            {"model__C": [0.01, 0.1, 1.0, 10.0]},
        )
    except Exception as e:
        print("SMOTE spec unavailable:", e)

    for variant_name, split in variants:
        for model_name, estimator, grid in model_specs:
            full_name = f"{variant_name}::{model_name}"
            bp, m = evaluate_model(name=full_name, estimator=estimator, param_grid=grid, split=split, cv=cv)
            if len(bp["best_params"]) > 0:
                best_params[bp["model"]] = bp["best_params"]
            rows.append({"variant": variant_name, "model": model_name, **m})

        if smote_spec is not None:
            model_name, estimator, grid = smote_spec
            full_name = f"{variant_name}::{model_name}"
            try:
                bp, m = evaluate_model(name=full_name, estimator=estimator, param_grid=grid, split=split, cv=cv)
                if len(bp["best_params"]) > 0:
                    best_params[bp["model"]] = bp["best_params"]
                rows.append({"variant": variant_name, "model": model_name, **m})
            except Exception as e:
                print(f"SMOTE skipped for {variant_name}:", e)

    out = pd.DataFrame(rows).sort_values(by=["variant", "roc_auc"], ascending=[True, False])
    out.to_csv(OUT_METRICS, index=False)
    with open(OUT_PARAMS, "w", encoding="utf-8") as f:
        json.dump(best_params, f, indent=2)

    print("\nSaved:")
    print(" -", OUT_METRICS)
    print(" -", OUT_PARAMS)
    print("\nTop results by variant (ROC-AUC):")
    print(out.groupby("variant").head(3))

    # -------------------------------------------------
    # Confusion matrix for BEST overall model
    #   Choose by ROC-AUC, tie-break by F1.
    # -------------------------------------------------
    best_row = out.sort_values(by=["roc_auc", "f1"], ascending=False).iloc[0]
    best_variant = str(best_row["variant"])
    best_model = str(best_row["model"])
    print("\nBest overall model:", f"{best_variant}::{best_model}")

    # Find the split for this variant
    split_map = {
        "all_features": split_all,
        "feature_selection_rfe_30": split_fs,
        "pca_continuous_only": split_pca_cont,
        "pca_all_features": split_pca_all,
    }
    split_best = split_map[best_variant]

    # Recreate the estimator with the best C (saved in best_params)
    key = f"{best_variant}::{best_model}"
    C_best = float(best_params.get(key, {}).get("model__C", 1.0))

    if best_model == "logreg_classweight_l2":
        estimator_best = Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=5000, solver="liblinear", class_weight="balanced", C=C_best)),
        ])
    elif best_model == "logreg_unbalanced_l2":
        estimator_best = Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=5000, solver="liblinear", C=C_best)),
        ])
    else:
        # Fallback to unbalanced L2
        estimator_best = Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=5000, solver="liblinear", C=C_best)),
        ])

    estimator_best.fit(split_best.X_train, split_best.y_train)
    proba = estimator_best.predict_proba(split_best.X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)

    cm = confusion_matrix(split_best.y_test.to_numpy(), pred)
    tn, fp, fn, tp = cm.ravel()
    print("Confusion matrix [ [TN FP] [FN TP] ]:")
    print(cm)
    print(f"TN={tn} FP={fp} FN={fn} TP={tp}")

    plt.figure(figsize=(5.5, 4.8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["Pred 0", "Pred 1"],
        yticklabels=["True 0", "True 1"],
    )
    plt.title(f"Best model CM: {best_variant}::{best_model} (thr={THRESHOLD})")
    plt.tight_layout()
    plt.savefig(OUT_CM_PLOT, dpi=200)
    plt.close()
    print("Saved confusion matrix plot:")
    print(" -", OUT_CM_PLOT)


if __name__ == "__main__":
    main()
