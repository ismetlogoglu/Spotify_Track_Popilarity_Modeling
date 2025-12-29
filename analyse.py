
import pandas as pd

# Read the dataset
df = pd.read_csv("/Users/ismet/Desktop/411_Project/dataset.csv", na_values=["NA", ""])

# First 5 rows (head)
print(df.head())
print()

# Data structure (sütun isimleri ve veri tipleri)
print("=== DATA STRUCTURE ===")
print(f"Satır sayısı: {df.shape[0]}")
print(f"Sütun sayısı: {df.shape[1]}")
print()
print("Sütunlar ve Veri Tipleri:")
print(df.dtypes)
print()

# Data summary (özet istatistikler)
print("=== DATA SUMMARY (Sayısal Sütunlar) ===")
print(df.describe())
print()

# Eksik veri bilgisi
print("=== EKSİK VERİ BİLGİSİ ===")
missing = df.isnull().sum()
missing_pct = (df.isnull().sum() / len(df)) * 100
missing_df = pd.DataFrame({"Eksik Sayısı": missing, "Yüzde (%)": missing_pct})
print(missing_df[missing_df["Eksik Sayısı"] > 0])