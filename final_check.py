import pandas as pd

# Clean, minimal check script (no functions) for the cleaned FULL KNN-imputed dataset
path = "/Users/ismet/Desktop/411_Project/dataset_tidy_knn_imputed_full_clean.csv"
df = pd.read_csv(path)

print("PATH:", path)
print("\nSHAPE:")
print(df.shape)

print("\nHEAD:")
print(df.head())

print("\nSTRUCTURE (dtypes):")
print(df.dtypes)

print("\nSUMMARY (describe):")
print(df.describe().T)

print("\nMISSING VALUES (per column, nonzero only):")
miss = df.isna().sum()
miss = miss[miss > 0].sort_values(ascending=False)
print(miss if len(miss) else "No missing values.")
