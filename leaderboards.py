"""Leaderboards for STAT 411 Project models.

Purpose
-------
Create small, report-friendly summary CSVs that answer:
- "Kaç model yarattık?" (how many model variants were evaluated)
- "Hangisini seçtik?" (which model is selected as best)

Inputs (expected to already exist)
---------------------------------
- cda_outputs/modeling_metrics.csv  (from modeling.py)
- cda_outputs/model_evaluation_comparison.csv (from model_eveluation.py)
- cda_outputs/classification_metrics.csv (from modeling_classification.py)

Outputs
-------
- cda_outputs/leaderboard_regression.csv
- cda_outputs/leaderboard_classification.csv

Selection rules
---------------
Regression:
- Within each variant: choose lowest RMSE.
- Overall best: lowest RMSE.

Classification:
- Within each variant: choose highest F1 (tie-break ROC-AUC, then Kappa).
- Overall best: highest F1 (tie-break ROC-AUC, then Kappa).

Note: ROC-AUC is threshold-independent; F1/precision depend on default threshold 0.5.
"""

from __future__ import annotations

import os

import pandas as pd


PROJECT_ROOT = "/Users/ismet/Desktop/411_Project"
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "cda_outputs")

REG_METRICS = os.path.join(OUTPUTS_DIR, "modeling_metrics.csv")
REG_EVAL = os.path.join(OUTPUTS_DIR, "model_evaluation_comparison.csv")
CLF_METRICS = os.path.join(OUTPUTS_DIR, "classification_metrics.csv")

OUT_REG = os.path.join(OUTPUTS_DIR, "leaderboard_regression.csv")
OUT_CLF = os.path.join(OUTPUTS_DIR, "leaderboard_classification.csv")


def leaderboard_regression() -> pd.DataFrame:
    # Prefer the richer comparison file if it exists.
    if os.path.exists(REG_EVAL):
        df = pd.read_csv(REG_EVAL)
        # expected columns: variant, model, rmse, mae, r2
        best_by_variant = (
            df.sort_values(["variant", "rmse", "mae"], ascending=[True, True, True])
            .groupby("variant", as_index=False)
            .head(1)
            .reset_index(drop=True)
        )
        best_by_variant["selected"] = best_by_variant["rmse"] == best_by_variant["rmse"].min()
        return best_by_variant

    if os.path.exists(REG_METRICS):
        df = pd.read_csv(REG_METRICS)
        df = df.copy()
        df.insert(0, "variant", "all_features")
        best = df.sort_values(["rmse", "mae"], ascending=[True, True]).head(1).copy()
        df["selected"] = False
        df.loc[best.index, "selected"] = True
        return df

    raise FileNotFoundError("No regression metrics found. Run modeling.py and/or model_eveluation.py first.")


def leaderboard_classification() -> pd.DataFrame:
    if not os.path.exists(CLF_METRICS):
        raise FileNotFoundError("No classification metrics found. Run modeling_classification.py first.")

    df = pd.read_csv(CLF_METRICS)
    # expected columns include: variant, model, roc_auc, f1, precision, ...
    # Rubric-friendly choice for imbalanced classification:
    # prioritize F1 (balance of precision/recall), then ROC-AUC (ranking quality),
    # then Kappa (agreement beyond chance).
    best_by_variant = (
        df.sort_values(["variant", "f1", "roc_auc", "kappa"], ascending=[True, False, False, False])
        .groupby("variant", as_index=False)
        .head(1)
        .reset_index(drop=True)
    )

    # Mark overall selection (same rule but across all rows)
    df_best = df.sort_values(["f1", "roc_auc", "kappa"], ascending=[False, False, False]).head(1)
    best_variant = str(df_best.iloc[0]["variant"])
    best_model = str(df_best.iloc[0]["model"])
    best_by_variant["selected"] = (best_by_variant["variant"] == best_variant) & (best_by_variant["model"] == best_model)
    return best_by_variant


def main() -> None:
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    reg = leaderboard_regression()
    reg.to_csv(OUT_REG, index=False)

    clf = leaderboard_classification()
    clf.to_csv(OUT_CLF, index=False)

    print("Saved leaderboards:")
    print(" -", OUT_REG)
    print(" -", OUT_CLF)

    # Helpful console summary
    print("\nCounts:")
    if os.path.exists(REG_EVAL):
        print("Regression models evaluated:", len(pd.read_csv(REG_EVAL)))
    elif os.path.exists(REG_METRICS):
        print("Regression models evaluated:", len(pd.read_csv(REG_METRICS)))
    print("Classification models evaluated:", len(pd.read_csv(CLF_METRICS)))

    print("\nSelected regression model:")
    print(reg[reg["selected"]].to_string(index=False))

    print("\nSelected classification model:")
    print(clf[clf["selected"]].to_string(index=False))


if __name__ == "__main__":
    main()
