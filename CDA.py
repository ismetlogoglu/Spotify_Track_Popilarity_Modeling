import os

import numpy as np
import pandas as pd

from scipy import stats


# ============================================================
# Confirmatory Data Analysis (CDA)
# Research question from EDA:
# "Which audio features and track genre are associated with popularity?"
#
# In CDA we confirm with statistical tests + assumption checks:
# 1) Numeric features vs popularity
#    - Pearson correlation test if (approx) normal + linear
#    - Otherwise Spearman rank correlation
#
# 2) Popularity differences across genres (top N genres)
#    - One-way ANOVA if assumptions are reasonable
#    - Otherwise Kruskal–Wallis H-test
#    - Post-hoc: Tukey HSD (parametric) or pairwise Mann–Whitney with Holm correction
#
# Dataset used: dataset_EDA.csv (built earlier, no missing values)
# ============================================================


# ---------------------------
# Paths
# ---------------------------
BASE_DIR = "/Users/ismet/Desktop/411_Project"
DATA_PATH = os.path.join(BASE_DIR, "dataset_EDA.csv")
OUT_DIR = os.path.join(BASE_DIR, "cda_outputs")
os.makedirs(OUT_DIR, exist_ok=True)

# CDA output policy
# User request: do NOT write CDA results to CSV; print everything to terminal only.
# Keep this hard-disabled to avoid accidental file writes.
SAVE_OUTPUTS = False

print("DATA_PATH:", DATA_PATH)
print("OUT_DIR:", OUT_DIR)


# ---------------------------
# Load
# ---------------------------
df = pd.read_csv(DATA_PATH)
print("\nSHAPE:", df.shape)

if "popularity" not in df.columns:
	raise ValueError("dataset_EDA.csv must contain 'popularity'.")


# ---------------------------
# Helpers (small + practical)
# ---------------------------
def _normality_pvalue(series: pd.Series) -> float:
	"""Shapiro-Wilk p-value. Uses sampling for very large N (Shapiro is sensitive and has limits)."""
	s = series.dropna().astype(float)
	if len(s) < 3:
		return np.nan
	# Shapiro is not recommended for huge N; sample to 5000 to avoid warnings/slowdowns.
	if len(s) > 5000:
		s = s.sample(5000, random_state=411)
	try:
		_, p = stats.shapiro(s)
		return float(p)
	except Exception:
		return np.nan


def _effect_size_eta_squared_anova(groups: list[np.ndarray]) -> float:
	"""Eta-squared effect size for one-way ANOVA."""
	all_vals = np.concatenate(groups)
	overall_mean = np.mean(all_vals)
	ss_total = np.sum((all_vals - overall_mean) ** 2)
	ss_between = 0.0
	for g in groups:
		ss_between += len(g) * (np.mean(g) - overall_mean) ** 2
	return float(ss_between / ss_total) if ss_total > 0 else np.nan


def _effect_size_epsilon_squared_kw(H: float, n: int, k: int) -> float:
	"""Epsilon-squared for Kruskal–Wallis."""
	if n <= k or n <= 1:
		return np.nan
	return float((H - k + 1) / (n - k))


def _holm_adjust(pvals: np.ndarray) -> np.ndarray:
	"""Holm-Bonferroni adjusted p-values."""
	pvals = np.asarray(pvals, dtype=float)
	m = len(pvals)
	order = np.argsort(pvals)
	adj = np.empty(m, dtype=float)
	for rank, idx in enumerate(order):
		adj[idx] = (m - rank) * pvals[idx]
	# enforce monotonicity
	adj_sorted = adj[order]
	adj_sorted = np.maximum.accumulate(adj_sorted)
	adj_sorted = np.clip(adj_sorted, 0, 1)
	adj[order] = adj_sorted
	return adj


# ============================================================
# 1) Numeric features vs popularity
# ============================================================

print("\n" + "=" * 60)
print("1) NUMERIC FEATURES vs POPULARITY")
print("=" * 60)

num_cols = df.select_dtypes(include="number").columns.tolist()
num_cols = [c for c in num_cols if c != "popularity"]

results_corr = []

pop_p = _normality_pvalue(df["popularity"])
print(f"Normality (Shapiro, sampled) popularity p={pop_p:.4g} (small p => not normal)")

for col in num_cols:
	x = df[col].astype(float)
	y = df["popularity"].astype(float)

	# Normality check (rough). With large N, almost always 'not normal';
	# we use it mainly to decide Pearson vs Spearman.
	p_x = _normality_pvalue(x)
	use_pearson = (not np.isnan(p_x)) and (not np.isnan(pop_p)) and (p_x > 0.05) and (pop_p > 0.05)

	if use_pearson:
		r, p = stats.pearsonr(x, y)
		method = "Pearson"
	else:
		r, p = stats.spearmanr(x, y)
		method = "Spearman"

	results_corr.append(
		{
			"feature": col,
			"method": method,
			"corr": float(r),
			"p_value": float(p),
			"normality_p_feature": float(p_x) if not np.isnan(p_x) else np.nan,
		}
	)

df_corr = pd.DataFrame(results_corr)
df_corr["abs_corr"] = df_corr["corr"].abs()
df_corr = df_corr.sort_values("abs_corr", ascending=False)

print("\nTop 10 associations with popularity (by |corr|):")
print(df_corr[["feature", "method", "corr", "p_value"]].head(10).to_string(index=False))

print("\nFull correlation table (sorted by |corr|):")
print(df_corr[["feature", "method", "corr", "p_value", "normality_p_feature"]].to_string(index=False))

if SAVE_OUTPUTS:
	df_corr.to_csv(os.path.join(OUT_DIR, "cda_numeric_popularity_correlations.csv"), index=False)
	print("Saved: cda_numeric_popularity_correlations.csv")


# ============================================================
# 2) Popularity differences across genres
# ============================================================

print("\n" + "=" * 60)
print("2) POPULARITY DIFFERENCES ACROSS GENRES")
print("=" * 60)

if "track_genre" not in df.columns:
	print("No 'track_genre' column found. Skipping genre CDA.")
	print("Done.")
	raise SystemExit(0)

# We only test top genres to keep assumptions + interpretation reasonable
TOP_N_GENRES = 10
top_genres = df["track_genre"].value_counts().head(TOP_N_GENRES).index.tolist()
df_g = df[df["track_genre"].isin(top_genres)].copy()

# Group arrays
groups = [df_g.loc[df_g["track_genre"] == g, "popularity"].astype(float).to_numpy() for g in top_genres]

print(f"Using top {TOP_N_GENRES} genres (N={len(df_g)} rows)")

# Assumption checks
# - Normality per group: Shapiro (sampled)
# - Homogeneity of variance: Levene
group_norm_p = {g: _normality_pvalue(df_g.loc[df_g["track_genre"] == g, "popularity"]) for g in top_genres}
levene_stat, levene_p = stats.levene(*groups, center="median")

print("\nNormality (Shapiro, sampled) per genre: (p>0.05 ~ approx normal)")
for g in top_genres:
	print(f"  {g:20s} p={group_norm_p[g]:.4g}")
print(f"\nLevene test (homogeneity of variance): stat={levene_stat:.4g}, p={levene_p:.4g}")

all_normal = all((p > 0.05) for p in group_norm_p.values() if not np.isnan(p))
equal_var = (levene_p > 0.05) if not np.isnan(levene_p) else False

use_anova = all_normal and equal_var

if use_anova:
	print("\nDecision: assumptions look OK -> using one-way ANOVA")
	F, p = stats.f_oneway(*groups)
	eta2 = _effect_size_eta_squared_anova(groups)
	print(f"ANOVA: F={F:.4g}, p={p:.4g}, eta^2={eta2:.4g}")

	# Post-hoc Tukey HSD
	try:
		from statsmodels.stats.multicomp import pairwise_tukeyhsd

		tuk = pairwise_tukeyhsd(endog=df_g["popularity"], groups=df_g["track_genre"], alpha=0.05)
		tuk_df = pd.DataFrame(data=tuk.summary().data[1:], columns=tuk.summary().data[0])
		print("\nPost-hoc (Tukey HSD):")
		print(tuk_df.to_string(index=False))
		if SAVE_OUTPUTS:
			tuk_df.to_csv(os.path.join(OUT_DIR, "cda_genre_posthoc_tukey.csv"), index=False)
			print("Saved: cda_genre_posthoc_tukey.csv")
	except Exception as e:
		print("Tukey post-hoc skipped (statsmodels missing or error):", e)

else:
	print("\nDecision: assumptions not met (common with large N) -> using Kruskal–Wallis")
	H, p = stats.kruskal(*groups)
	eps2 = _effect_size_epsilon_squared_kw(H, n=len(df_g), k=len(top_genres))
	print(f"Kruskal–Wallis: H={H:.4g}, p={p:.4g}, epsilon^2={eps2:.4g}")

	# Post-hoc pairwise Mann–Whitney U with Holm correction
	pairs = []
	pvals = []
	for i in range(len(top_genres)):
		for j in range(i + 1, len(top_genres)):
			g1, g2 = top_genres[i], top_genres[j]
			x = df_g.loc[df_g["track_genre"] == g1, "popularity"].astype(float)
			y = df_g.loc[df_g["track_genre"] == g2, "popularity"].astype(float)
			U, p_u = stats.mannwhitneyu(x, y, alternative="two-sided")
			pairs.append((g1, g2, float(U), float(p_u)))
			pvals.append(float(p_u))

	adj = _holm_adjust(np.array(pvals))
	post = pd.DataFrame(pairs, columns=["genre_1", "genre_2", "U_stat", "p_value_raw"])
	post["p_value_holm"] = adj
	post = post.sort_values("p_value_holm")
	print("\nPost-hoc (pairwise Mann–Whitney U with Holm correction) — top 25 most significant:")
	print(post.head(25).to_string(index=False))
	if SAVE_OUTPUTS:
		post.to_csv(os.path.join(OUT_DIR, "cda_genre_posthoc_mannwhitney_holm.csv"), index=False)
		print("Saved: cda_genre_posthoc_mannwhitney_holm.csv")


# Save group summary (means/medians)
summary = (
	df_g.groupby("track_genre")["popularity"]
	.agg(n="count", mean="mean", median="median", std="std")
	.sort_values("mean", ascending=False)
)
print("\nTop genre popularity summary (by mean):")
print(summary.head(10).to_string())

if SAVE_OUTPUTS:
	summary.to_csv(os.path.join(OUT_DIR, "cda_genre_popularity_summary.csv"))
	print("\nSaved: cda_genre_popularity_summary.csv")


print("\nDone.")

