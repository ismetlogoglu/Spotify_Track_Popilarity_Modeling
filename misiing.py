import csv
import json
from pathlib import Path

root = Path(__file__).resolve().parent
dataset = root / "dataset.csv"
indices = root / "na_indices.json"

# Column-wise NA counts (and optional percentages)
with dataset.open("r", newline="", encoding="utf-8") as fh:
    reader = csv.DictReader(fh)
    fieldnames = reader.fieldnames or []
    rows = list(reader)

total_rows = len(rows)
with indices.open("r", encoding="utf-8") as fh:
    na_map = json.load(fh)

report = []
for column, ids in na_map.items():
    count = len(ids)
    pct = (count / total_rows) * 100 if total_rows else 0
    report.append((column or "<index>", count, pct))

for column, count, pct in sorted(report):
    print(f"{column:18s} -> {count:6d} rows missing ({pct:5.2f}%)")