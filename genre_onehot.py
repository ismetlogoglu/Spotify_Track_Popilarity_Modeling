import pandas as pd
from sklearn.model_selection import train_test_split

# This script does NOT rerun KNN.
# It uses the already-created numeric KNN-imputed dataset and adds:
# 1) track_genre_merged (rare -> Other, based on TRAIN frequency)
# 2) one-hot columns for track_genre_merged

NUM_PATH = "/Users/ismet/Desktop/411_Project/dataset_tidy_knn_imputed_trainfit.csv"
RAW_PATH = "/Users/ismet/Desktop/411_Project/dataset_tidy.csv"

OUT_MERGED = "/Users/ismet/Desktop/411_Project/dataset_knn_plus_genre_merged.csv"
OUT_ONEHOT = "/Users/ismet/Desktop/411_Project/dataset_knn_plus_genre_onehot.csv"

min_count = 200

num = pd.read_csv(NUM_PATH)
raw = pd.read_csv(RAW_PATH, na_values=["NA", ""])
raw = raw[raw["popularity"].notna()].copy()  # align with num rows

# Sanity check
if len(num) != len(raw):
    raise ValueError("Row counts do not match between numeric-imputed and raw tidy data")

# Build train split ONLY to decide which genres are rare
X = raw.drop(columns=["popularity"])
y = raw["popularity"].astype(float)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=411)

vc = X_train["track_genre"].value_counts(dropna=False)
rare = set(vc[vc < min_count].index.astype(str).tolist())

track_genre_merged = (
    raw["track_genre"]
    .fillna("Missing")
    .astype(str)
    .apply(lambda g: "Other" if g in rare else g)
)

print("Unique track_genre:", int(raw["track_genre"].nunique(dropna=True)))
print("Unique track_genre_merged:", int(track_genre_merged.nunique()))
print("Other count:", int((track_genre_merged == "Other").sum()))

# (1) Save readable merged version
merged = num.copy()
merged["track_genre_merged"] = track_genre_merged.values
merged.to_csv(OUT_MERGED, index=False)
print("Saved:", OUT_MERGED)

# (2) One-hot version
onehot = pd.get_dummies(track_genre_merged, prefix="genre", dtype=int)
onehot_df = pd.concat([num.reset_index(drop=True), onehot.reset_index(drop=True)], axis=1)
onehot_df.to_csv(OUT_ONEHOT, index=False)
print("Saved:", OUT_ONEHOT)

print("Done.")
