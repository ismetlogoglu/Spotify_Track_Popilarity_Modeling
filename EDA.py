import os

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats
from scipy.stats import chi2_contingency
from statsmodels.graphics.mosaicplot import mosaic




# =========================
# DATA
# =========================
path = "/Users/ismet/Desktop/411_Project/dataset_EDA.csv"
df = pd.read_csv(path)

PLOTS_DIR = "/Users/ismet/Desktop/411_Project/plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

# Modern look
sns.set_theme(style="whitegrid", context="talk")
sns.set_palette("bright")

print("PATH:", path)
print("PLOTS_DIR:", PLOTS_DIR)


# =========================
# BASIC CHECKS
# =========================
print("\nSHAPE:")
print(df.shape)

print("\nHEAD (first 5 rows):")
print(df.head())

print("\nSTRUCTURE (dtypes):")
print(df.dtypes)

print("\nSUMMARY (describe, include all):")
print(df.describe(include="all").T)

print("\nMISSING VALUES (per column):")
print(df.isna().sum())


# =========================
# NUMERIC / CATEGORICAL
# =========================
num_df = df.select_dtypes(include="number")
cat_df = df.select_dtypes(include="object")

# -------------------------
# Plot-only outlier handling (NO standardization)
# -------------------------
df_plot = df.copy()

# Duration outlier removal for visualization (IQR rule). This does NOT change dataset_EDA.csv.
if "duration_ms" in df_plot.columns:
	q1 = df_plot["duration_ms"].quantile(0.25)
	q3 = df_plot["duration_ms"].quantile(0.75)
	iqr = q3 - q1
	lo = q1 - 1.5 * iqr
	hi = q3 + 1.5 * iqr
	before = len(df_plot)
	df_plot = df_plot[(df_plot["duration_ms"] >= lo) & (df_plot["duration_ms"] <= hi)].copy()
	after = len(df_plot)
	print(f"\n[Plot-only] duration_ms IQR filter: kept {after}/{before} rows ({before - after} removed)")

num_df_plot = df_plot.select_dtypes(include="number")

print("\nNUMERIC COLS:", list(num_df.columns))
print("CATEGORICAL COLS:", list(cat_df.columns))


# =========================
# UNIVARIATE: NUMERIC
# =========================
print("\nUNIVARIATE: numeric (skewness + kurtosis)")
sk = num_df.apply(lambda s: stats.skew(s, nan_policy="omit"))
ku = num_df.apply(lambda s: stats.kurtosis(s, nan_policy="omit"))
print(
	pd.DataFrame({"skew": sk, "kurtosis": ku}).sort_values(
		"skew", key=lambda s: s.abs(), ascending=False
	)
)

plt.figure(figsize=(7, 4))
sns.histplot(df_plot["popularity"], bins=30, kde=True)
plt.title("Popularity distribution")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "hist_popularity.png"), dpi=200)
plt.close()

# Hist + KDE for numeric columns (excluding popularity)
cols = [c for c in num_df_plot.columns if c != "popularity"]
if len(cols):
	ncols = 4
	nrows = int(np.ceil(len(cols) / ncols))
	fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
	axes = np.array(axes).reshape(-1)

	for i, c in enumerate(cols):
		sns.histplot(df_plot[c], bins=30, kde=True, ax=axes[i])
		axes[i].set_title(c)

	for j in range(i + 1, len(axes)):
		axes[j].axis("off")

	fig.suptitle("Numeric distributions (hist + KDE)", y=1.02)
	fig.tight_layout()
	fig.savefig(os.path.join(PLOTS_DIR, "hist_all_numeric.png"), dpi=200)
	plt.close(fig)

plt.figure(figsize=(10, 6))
sns.boxplot(
	data=num_df_plot.drop(columns=["popularity"], errors="ignore"),
	orient="h",
	showfliers=False,
	linewidth=1,
)
plt.title("Box plots (numeric features; outliers hidden; duration filtered)")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "box_numeric_modern.png"), dpi=200)
plt.close()


# =========================
# UNIVARIATE: CATEGORICAL
# =========================
print("\nUNIVARIATE: categorical (top counts)")
if cat_df.shape[1] == 0:
	print("No categorical columns.")
else:
	for c in cat_df.columns:
		vc = df[c].value_counts(dropna=False)
		print(f"\n{c}: unique={vc.size}")
		print(vc.head(15))

	if "track_genre" in df.columns:
		topk = 20
		vc = df["track_genre"].value_counts().head(topk)
		fig, ax = plt.subplots(figsize=(11, 7))
		sns.barplot(x=vc.values, y=vc.index, color="#2563eb", ax=ax)
		ax.set_title(f"Track genre counts (top {topk})")
		ax.set_xlabel("count")
		ax.set_ylabel("track_genre")
		ax.tick_params(axis="y", labelsize=10)
		ax.tick_params(axis="x", labelsize=10)

		# Add small count labels on each bar
		xmax = float(vc.values.max())
		pad = 0.01 * xmax
		for i, v in enumerate(vc.values):
			ax.text(float(v) + pad, i, f"{int(v)}", va="center", ha="left", fontsize=9, color="#111827")
		ax.set_xlim(0, xmax * 1.10)

		fig.tight_layout()
		fig.savefig(os.path.join(PLOTS_DIR, "bar_track_genre_top20.png"), dpi=200)
		plt.close(fig)


# =========================
# BIVARIATE/MULTIVARIATE: CONTINUOUS
# =========================
print("\nBIVARIATE: correlations (Pearson + Spearman)")
pearson = num_df_plot.corr(method="pearson")
spearman = num_df_plot.corr(method="spearman")

mask_upper = np.triu(np.ones_like(pearson, dtype=bool), k=1)

plt.figure(figsize=(11, 9))
sns.heatmap(
	pearson,
	mask=mask_upper,
	cmap="coolwarm",
	center=0,
	annot=True,
	fmt=".2f",
	linewidths=0.5,
	annot_kws={"size": 9},
	cbar_kws={"label": "corr"},
)
plt.title("Correlation matrix (Pearson)")
plt.xticks(rotation=45, ha="right", fontsize=10)
plt.yticks(rotation=0, fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "heatmap_corr_pearson_triangle_annot.png"), dpi=200)
plt.close()

plt.figure(figsize=(11, 9))
sns.heatmap(
	spearman,
	mask=mask_upper,
	cmap="coolwarm",
	center=0,
	annot=True,
	fmt=".2f",
	linewidths=0.5,
	annot_kws={"size": 9},
	cbar_kws={"label": "corr"},
)
plt.title("Correlation matrix (Spearman)")
plt.xticks(rotation=45, ha="right", fontsize=10)
plt.yticks(rotation=0, fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "heatmap_corr_spearman_triangle_annot.png"), dpi=200)
plt.close()

print("\nCorrelation with popularity (Pearson, top 15 abs):")
c = pearson["popularity"].dropna().sort_values(key=lambda s: s.abs(), ascending=False)
print(c.head(15))

top_feats = [x for x in c.index if x != "popularity"][:5]

def nice_scatter(xcol: str) -> None:
	fig, ax = plt.subplots(figsize=(7.2, 4.6))
	sns.scatterplot(
		data=df_plot,
		x=xcol,
		y="popularity",
		alpha=0.28,
		s=16,
		color="#2563eb",
		edgecolor="none",
		ax=ax,
	)
	sns.regplot(
		data=df_plot,
		x=xcol,
		y="popularity",
		scatter=False,
		color="#ef4444",
		line_kws={"linewidth": 2},
		ax=ax,
	)
	ax.set_title(f"Popularity vs {xcol}")
	ax.set_xlabel(xcol)
	ax.set_ylabel("popularity")
	fig.tight_layout()
	fig.savefig(os.path.join(PLOTS_DIR, f"scatter_popularity_vs_{xcol}_modern.png"), dpi=200)
	plt.close(fig)

for feat in top_feats:
	nice_scatter(feat)


# popularity ~ track_genre: box + jitter
if "track_genre" in df.columns:
	top_genres = df["track_genre"].value_counts().head(12).index
	tmp = df_plot[df_plot["track_genre"].isin(top_genres)].copy()
	plt.figure(figsize=(12, 6))
	sns.boxplot(data=tmp, x="track_genre", y="popularity", hue="track_genre", showfliers=False, palette="magma", dodge=False, legend=False)
	sns.stripplot(
		data=tmp,
		x="track_genre",
		y="popularity",
		color="#1b1b1b",
		alpha=0.18,
		size=2.2,
		jitter=0.25,
	)
	plt.xticks(rotation=45, ha="right")
	plt.title("Popularity by track_genre (top 12) — box + jitter (modern)")
	plt.tight_layout()
	plt.savefig(os.path.join(PLOTS_DIR, "box_jitter_popularity_by_genre_top12_modern.png"), dpi=200)
	plt.close()


# =========================
# CATEGORICAL ASSOCIATION: Chi-square, Cramer's V, Odds Ratio
# =========================
def cramers_v(table: pd.DataFrame) -> float:
	chi2, _, _, _ = chi2_contingency(table)
	n = table.to_numpy().sum()
	r, k = table.shape
	if n == 0:
		return np.nan
	phi2 = chi2 / n
	phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
	rcorr = r - ((r - 1) ** 2) / (n - 1)
	kcorr = k - ((k - 1) ** 2) / (n - 1)
	denom = min((kcorr - 1), (rcorr - 1))
	return float(np.sqrt(phi2corr / denom)) if denom > 0 else np.nan


print("\nCATEGORICAL ASSOCIATION (Chi-square + Cramer's V)")
if "popularity" in df.columns:
	df["popularity_bin"] = pd.qcut(df["popularity"], q=4, labels=["Q1", "Q2", "Q3", "Q4"])

pairs = []
if "track_genre" in df.columns and "popularity_bin" in df.columns:
	pairs.append(("track_genre", "popularity_bin"))
if not pairs and cat_df.shape[1] >= 2:
	pairs.append((cat_df.columns[0], cat_df.columns[1]))

if not pairs:
	print("No suitable categorical pairs found.")

for a, b in pairs:
	table = pd.crosstab(df[a], df[b])
	chi2, p, dof, _ = chi2_contingency(table)
	v = cramers_v(table)
	print(f"\n{a} vs {b}")
	print("table shape:", table.shape, "chi2:", round(chi2, 3), "dof:", dof, "p-value:", p, "Cramer's V:", round(v, 4))

	a_top = df[a].value_counts().head(10).index
	b_top = df[b].value_counts().head(6).index
	# For mosaic: NO "Other". Keep only top-N categories for clearer comparison.
	m = df[df[a].isin(a_top) & df[b].isin(b_top)].copy()

	# Mosaic: zoomed out + readable labels + legend
	palette = sns.color_palette("Set2", n_colors=len(b_top))
	color_map = {cat: palette[i] for i, cat in enumerate(b_top)}

	def props(key):
		# key is a tuple like (a_value, b_value)
		b_val = key[1] if isinstance(key, tuple) and len(key) > 1 else None
		return {"facecolor": color_map.get(b_val, (0.8, 0.8, 0.8)), "edgecolor": "white"}

	fig, ax = plt.subplots(figsize=(20, 10))
	mosaic(
		m,
		[a, b],
		properties=props,
		labelizer=lambda k: "",
		ax=ax,
	)
	ax.set_title(f"Mosaic: {a} vs {b} (top categories)")

	# Make x labels readable: show fewer, rotate, smaller font
	for label in ax.get_xticklabels():
		label.set_rotation(30)
		label.set_ha("right")
		label.set_fontsize(10)
	for label in ax.get_yticklabels():
		label.set_fontsize(10)

	# Legend: color corresponds to the second variable categories
	from matplotlib.patches import Patch
	legend_handles = [Patch(facecolor=color_map[cat], edgecolor="white", label=str(cat)) for cat in b_top]
	leg = ax.legend(
		handles=legend_handles,
		title=str(b),
		loc="center left",
		bbox_to_anchor=(1.01, 0.5),
		frameon=True,
	)
	plt.setp(leg.get_title(), fontsize=11)
	for txt in leg.get_texts():
		txt.set_fontsize(10)

	fig.tight_layout()
	fig.savefig(os.path.join(PLOTS_DIR, f"mosaic_{a}_vs_{b}_toponly_modern.png"), dpi=200)
	plt.close(fig)

	if table.shape == (2, 2):
		t = table.to_numpy().astype(float) + 0.5
		odds_ratio = (t[0, 0] * t[1, 1]) / (t[0, 1] * t[1, 0])
		print("odds ratio (2x2, +0.5 correction):", float(odds_ratio))


# =========================
# INTERACTIVE PLOTS (Plotly) -> HTML
# =========================

print("\nINTERACTIVE plot (Plotly) -> saved as HTML")
try:
	import plotly.express as px

	# One useful interactive plot: popularity vs key feature, colored by genre (top genres only)
	feat = top_feats[0] if top_feats else "instrumentalness"
	if "track_genre" in df.columns:
		topg = df["track_genre"].value_counts().head(12).index
		dfi = df[df["track_genre"].isin(topg)].copy()
	else:
		dfi = df.copy()

	fig = px.scatter(
		dfi,
		x=feat,
		y="popularity",
		color="track_genre" if "track_genre" in dfi.columns else None,
		hover_data=[c for c in ["track_name", "artists", "album_name"] if c in dfi.columns],
		opacity=0.55,
		title=f"Popularity vs {feat} (interactive, top genres)",
	)
	fig.update_layout(template="plotly_white", legend_title_text="track_genre")
	fig.write_html(os.path.join(PLOTS_DIR, "interactive_popularity_scatter_by_genre.html"))

	print("Saved interactive HTML file in plots/.")
except Exception as e:
	print("Interactive plotting skipped. Reason:", e)


print("\nDone. Plots saved under:", PLOTS_DIR)

