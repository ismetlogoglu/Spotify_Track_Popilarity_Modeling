import csv
import json
import random
from pathlib import Path

RANDOM_SEED = 411
MIN_FRACTION = 0.05
MAX_FRACTION = 0.10
DATASET_NAME = 'dataset.csv'
OUTPUT_INDEX_FILE = 'na_indices.json'


def is_numeric_value(value: str) -> bool:
    try:
        float(value)
    except (TypeError, ValueError):
        return False
    return True


def main() -> None:
    random.seed(RANDOM_SEED)

    dataset_path = Path(__file__).resolve().parent / DATASET_NAME
    if not dataset_path.exists():
        raise FileNotFoundError(dataset_path)

    with dataset_path.open('r', newline='', encoding='utf-8') as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        rows = list(reader)

    if not rows:
        raise ValueError('dataset appears to be empty')

    numeric_fields: list[str] = []
    for field in fieldnames:
        if field is None:
            continue
        values = [row[field] for row in rows if row.get(field) not in (None, '', 'NA')]
        if not values:
            continue
        if all(is_numeric_value(value) for value in values):
            numeric_fields.append(field)

    num_rows = len(rows)
    index_field = fieldnames[0] if fieldnames else None
    na_indices: dict[str, list[str]] = {}

    for field in numeric_fields:
        fraction = random.uniform(MIN_FRACTION, MAX_FRACTION)
        count = max(1, int(round(num_rows * fraction)))
        selected_indices = random.sample(range(num_rows), count)
        selected_indices.sort()

        for row_index in selected_indices:
            rows[row_index][field] = 'NA'

        key_name = field if field else '__index__'
        if index_field and index_field in rows[0]:
            recorded_indices = [rows[row_index][index_field] or str(row_index) for row_index in selected_indices]
        else:
            recorded_indices = [str(row_index) for row_index in selected_indices]
        na_indices[key_name] = recorded_indices

    with dataset_path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    indices_path = dataset_path.with_name(OUTPUT_INDEX_FILE)
    with indices_path.open('w', encoding='utf-8') as handle:
        json.dump(na_indices, handle, indent=2)

    print(f'Processed {num_rows} rows, numeric columns: {numeric_fields}')


if __name__ == '__main__':
    main()
