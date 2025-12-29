import numpy as np
import pandas as pd

# Simple tidy script (no imputation). Existing missing values stay missing.

df = pd.read_csv("/Users/ismet/Desktop/411_Project/dataset.csv", na_values=["NA", ""])
print("Rows, Cols:", df.shape)
print("Total missing (initial):", int(df.isna().sum().sum()))

# 1) Duplicates
print("Duplicate rows (exact):", int(df.duplicated().sum()))
print("Duplicate track_id:", int(df.duplicated(subset=["track_id"]).sum()))
df = df.drop_duplicates(subset=["track_id"], keep="first")
print("Rows after dedup:", df.shape[0])

# 2) Inconsistency checks (set invalid values to NA)

invalid_popularity = ((df["popularity"] < 0) | (df["popularity"] > 100)).sum()
df.loc[(df["popularity"] < 0) | (df["popularity"] > 100), "popularity"] = np.nan

invalid_duration = ((df["duration_ms"] < 5000) | (df["duration_ms"] > 3600000)).sum()
df.loc[(df["duration_ms"] < 5000) | (df["duration_ms"] > 3600000), "duration_ms"] = np.nan

audio_cols = [
    "danceability",
    "energy",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
]
invalid_audio = 0
for c in audio_cols:
    m = (df[c] < 0) | (df[c] > 1)
    invalid_audio += int(m.sum())
    df.loc[m, c] = np.nan

invalid_key = ((df["key"] < 0) | (df["key"] > 11)).sum()
df.loc[(df["key"] < 0) | (df["key"] > 11), "key"] = np.nan

invalid_mode = (~df["mode"].isin([0, 1]) & df["mode"].notna()).sum()
df.loc[~df["mode"].isin([0, 1]), "mode"] = np.nan

invalid_loudness = ((df["loudness"] < -60) | (df["loudness"] > 5)).sum()
df.loc[(df["loudness"] < -60) | (df["loudness"] > 5), "loudness"] = np.nan

invalid_tempo = ((df["tempo"] < 20) | (df["tempo"] > 250)).sum()
df.loc[(df["tempo"] < 20) | (df["tempo"] > 250), "tempo"] = np.nan

valid_ts = [3, 4, 5, 7]
invalid_ts = (~df["time_signature"].isin(valid_ts) & df["time_signature"].notna()).sum()
df.loc[~df["time_signature"].isin(valid_ts), "time_signature"] = np.nan

print("Invalid popularity -> NA:", int(invalid_popularity))
print("Invalid duration_ms -> NA:", int(invalid_duration))
print("Invalid audio features -> NA (total):", int(invalid_audio))
print("Invalid key -> NA:", int(invalid_key))
print("Invalid mode -> NA:", int(invalid_mode))
print("Invalid loudness -> NA:", int(invalid_loudness))
print("Invalid tempo -> NA:", int(invalid_tempo))
print("Invalid time_signature -> NA:", int(invalid_ts))

# 3) Drop rows with missing critical text fields
before = df.shape[0]
df = df[df["track_id"].notna() & df["track_name"].notna() & df["artists"].notna()]
print("Dropped rows (missing track_id/track_name/artists):", int(before - df.shape[0]))

out = "/Users/ismet/Desktop/411_Project/dataset_tidy.csv"
df.to_csv(out, index=False, na_rep="NA")
print("Total missing (final):", int(df.isna().sum().sum()))
print("Saved:", out)
