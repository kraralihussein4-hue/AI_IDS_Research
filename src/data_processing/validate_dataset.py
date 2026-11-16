
from pathlib import Path
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data" / "processed"

CHUNK_SIZE = 100_000

EXPECTED_COLUMNS = 79

def validate_file(file_path):

    print("\n" + "=" * 65)
    print(f"Checking: {file_path.name}")
    print("=" * 65)

    total_rows = 0
    total_columns = None
    total_infinite = 0
    total_missing = 0
    total_non_numeric = 0
    labels = {}

    for chunk in pd.read_csv(
        file_path,
        chunksize=CHUNK_SIZE,
        low_memory=False
    ):

        chunk.columns = chunk.columns.str.strip()

        total_rows += len(chunk)

        if total_columns is None:
            total_columns = len(chunk.columns)

        if "Label" not in chunk.columns:
            raise ValueError("Label column is missing!")

        features = chunk.drop(columns=["Label"])

        for column in features.columns:

            numeric = pd.to_numeric(
                features[column],
                errors="coerce"
            )

            invalid = (
                features[column].notna() & numeric.isna()
            )

            total_non_numeric += int(invalid.sum())

            values = numeric.to_numpy(dtype=float)

            total_infinite += int(
                np.isinf(values).sum()
            )

            total_missing += int(
                numeric.isna().sum()
            )

        labels_chunk = chunk["Label"].value_counts(
            dropna=False
        ).to_dict()

        for label, count in labels_chunk.items():
            labels[label] = labels.get(label, 0) + count

    print(f"Rows: {total_rows:,}")
    print(f"Columns: {total_columns}")
    print(f"Missing feature values: {total_missing:,}")
    print(f"Infinite values: {total_infinite:,}")
    print(f"Non-numeric feature values: {total_non_numeric:,}")

    print("\nLabels:")
    for label, count in labels.items():
        print(f"  {label}: {count:,}")

    checks = {
        "Column count is correct": total_columns == EXPECTED_COLUMNS,
        "No infinite values": total_infinite == 0,
        "No unexpected non-numeric values": total_non_numeric == 0,
        "Label column exists": True,
    }

    print("\nValidation results:")

    for check, passed in checks.items():
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {check}")

    return {
        "rows": total_rows,
        "columns": total_columns,
        "missing": total_missing,
        "infinite": total_infinite,
        "non_numeric": total_non_numeric,
        "checks": checks,
    }


def main():

    if not DATA_DIR.exists():
        raise FileNotFoundError(
            f"Processed data folder not found: {DATA_DIR}"
        )

    files = sorted(DATA_DIR.glob("*.csv"))

    if len(files) != 8:
        print(
            f"WARNING: Expected 8 CSV files, found {len(files)}"
        )

    if not files:
        raise FileNotFoundError("No processed CSV files found!")

    print("=" * 65)
    print("CICIDS2017 DATASET VALIDATION")
    print("=" * 65)

    total_rows = 0
    all_passed = True

    for file_path in files:

        result = validate_file(file_path)

        total_rows += result["rows"]

        if not all(result["checks"].values()):
            all_passed = False

    print("\n" + "=" * 65)
    print("FINAL VALIDATION SUMMARY")
    print("=" * 65)

    print(f"Files checked: {len(files)}")
    print(f"Total rows checked: {total_rows:,}")

    if all_passed:
        print("\nALL STRUCTURAL CHECKS PASSED")
    else:
        print("\nSOME CHECKS FAILED - REVIEW REQUIRED")


if __name__ == "__main__":
    main()
