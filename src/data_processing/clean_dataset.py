
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd


# Project paths
BASE_DIR = Path(__file__).resolve().parents[2]

RAW_DIR = (
    BASE_DIR
    / "data"
    / "raw"
    / "MachineLearningCSV"
    / "MachineLearningCVE"
)

PROCESSED_DIR = BASE_DIR / "data" / "processed"

CHUNK_SIZE = 100_000


def clean_file(file_path: Path) -> dict:
    """
    Clean one CICIDS2017 CSV file in chunks.

    The original file is never modified.
    All rows and labels are preserved.
    Infinite values are replaced with NaN.
    """

    output_path = PROCESSED_DIR / file_path.name

    total_rows = 0
    total_missing = Counter()
    total_infinite = Counter()
    total_converted = Counter()
    label_counts = Counter()

    # Start a fresh output file on every run
    first_chunk = True

    print(f"\nProcessing: {file_path.name}")

    for chunk_number, chunk in enumerate(
        pd.read_csv(
            file_path,
            chunksize=CHUNK_SIZE,
            low_memory=False
        ),
        start=1
    ):

        # Remove accidental spaces from column names
        chunk.columns = chunk.columns.str.strip()

        total_rows += len(chunk)

        # Preserve the original label values
        labels = chunk["Label"].copy()

        feature_columns = [
            column for column in chunk.columns
            if column != "Label"
        ]

        # Convert feature columns to numeric values
        for column in feature_columns:

            original = chunk[column]

            numeric_values = pd.to_numeric(
                original,
                errors="coerce"
            )

            # Count non-empty values that could not be converted
            invalid_values = (
                original.notna() & numeric_values.isna()
            )

            total_converted[column] += int(
                invalid_values.sum()
            )

            # Count existing missing values
            total_missing[column] += int(
                numeric_values.isna().sum()
            )

            # Count infinite values before replacing them
            infinite_mask = np.isinf(
                numeric_values.to_numpy(dtype=float)
            )

            total_infinite[column] += int(
                infinite_mask.sum()
            )

            # Replace positive and negative infinity with NaN
            chunk[column] = numeric_values.replace(
                [np.inf, -np.inf],
                np.nan
            )

        # Restore labels without changing their text
        chunk["Label"] = labels

        label_counts.update(
            chunk["Label"].value_counts(dropna=False).to_dict()
        )

        # Save the cleaned chunk
        chunk.to_csv(
            output_path,
            mode="w" if first_chunk else "a",
            header=first_chunk,
            index=False,
            encoding="utf-8"
        )

        first_chunk = False

        print(
            f"  Chunk {chunk_number}: "
            f"{len(chunk):,} rows processed"
        )

    return {
        "file": file_path.name,
        "rows": total_rows,
        "output": str(output_path),
        "missing": total_missing,
        "infinite": total_infinite,
        "converted": total_converted,
        "labels": label_counts,
    }


def main():

    if not RAW_DIR.exists():
        raise FileNotFoundError(
            f"Dataset folder not found: {RAW_DIR}"
        )

    files = sorted(RAW_DIR.glob("*.csv"))

    if not files:
        raise FileNotFoundError(
            f"No CSV files found in: {RAW_DIR}"
        )

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("=" * 65)
    print("CICIDS2017 DATA CLEANING")
    print("=" * 65)

    print(f"Input files: {len(files)}")
    print(f"Output folder: {PROCESSED_DIR}")

    all_results = []

    for file_path in files:

        result = clean_file(file_path)
        all_results.append(result)

        print("\nFile summary:")
        print(f"Rows: {result['rows']:,}")
        print(f"Saved to: {result['output']}")

        print("\nLabels:")
        for label, count in result["labels"].items():
            print(f"  {label}: {count:,}")

    print("\n" + "=" * 65)
    print("CLEANING COMPLETED")
    print("=" * 65)

    total_rows = sum(
        result["rows"] for result in all_results
    )

    print(f"Total processed rows: {total_rows:,}")
    print(f"Processed files: {len(all_results)}")
    print(f"Output folder: {PROCESSED_DIR}")

    print("\nOriginal dataset files were not modified.")
    print("No rows or attack classes were intentionally removed.")


if __name__ == "__main__":
    main()
