"""Re-add popularity into a dataset with missing values.

Goal (user request):
- `dataset.csv` has NA values in `popularity`.
- `dataset1.csv` is the NA-free original version.
- We want to restore/fill ONLY the `popularity` column in `dataset.csv` using `dataset1.csv`.

Approach:
- Join key: (track_id, artists, album_name, track_name)
  * `track_id` alone is NOT unique in dataset1 (duplicates exist and in rare cases popularity differs).
  * Using the composite key makes matching safer.
- Fill policy:
  * For each key, take the median popularity from dataset1 (robust if duplicates).
  * Only fill rows where `dataset.csv.popularity` is NA.

Output:
- Writes `dataset_popularity_restored.csv` next to the inputs.
"""

from __future__ import annotations

import os
import pandas as pd


BASE_DIR = "/Users/ismet/Desktop/411_Project"
DATASET_WITH_NA = os.path.join(BASE_DIR, "dataset.csv")
DATASET_CLEAN = os.path.join(BASE_DIR, "dataset1.csv")
OUT_PATH = os.path.join(BASE_DIR, "dataset_popularity_restored.csv")

KEY_COLS = ["track_id", "artists", "album_name", "track_name"]
TARGET_COL = "popularity"


def main() -> None:
    df0 = pd.read_csv(DATASET_WITH_NA)
    df1 = pd.read_csv(DATASET_CLEAN, usecols=KEY_COLS + [TARGET_COL])

    if TARGET_COL not in df0.columns:
        raise ValueError(f"Expected '{TARGET_COL}' in {DATASET_WITH_NA}")

    missing_before = int(df0[TARGET_COL].isna().sum())
    print("Input:", DATASET_WITH_NA)
    print("Reference:", DATASET_CLEAN)
    print("Rows:", len(df0))
    print("Missing popularity (before):", missing_before)

    # Build a robust lookup table from dataset1
    # If duplicates exist, aggregate popularity by median per key.
    ref = (
        df1.groupby(KEY_COLS, dropna=False)[TARGET_COL]
        .median()
        .reset_index()
        .rename(columns={TARGET_COL: "popularity_ref"})
    )

    # Merge reference popularity
    merged = df0.merge(ref, on=KEY_COLS, how="left")

    # Fill only missing popularity
    merged[TARGET_COL] = merged[TARGET_COL].fillna(merged["popularity_ref"])
    merged = merged.drop(columns=["popularity_ref"])

    missing_after = int(merged[TARGET_COL].isna().sum())
    filled = missing_before - missing_after

    print("Filled popularity count:", filled)
    print("Missing popularity (after):", missing_after)

    merged.to_csv(OUT_PATH, index=False)
    print("Saved:", OUT_PATH)


if __name__ == "__main__":
    main()
