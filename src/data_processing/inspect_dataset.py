
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd


# Project paths
DATA_DIR = Path(
    "data/raw/MachineLearningCSV/MachineLearningCVE"
)

CHUNK_SIZE = 100_000

TARGET_COLUMNS = [
    "Flow Bytes/s",
    "Flow Packets/s",
]


def inspect_file(file_path):
    total_rows = 0
    missing_counts = Counter()
    infinite_counts = Counter()
    affected_labels = Counter()
    label_counts = Counter()

    for chunk in pd.read_csv(
        file_path,
        chunksize=CHUNK_SIZE,
        low_memory=False
    ):
        chunk.columns = chunk.columns.str.strip()
        total_rows += len(chunk)

        for column in TARGET_COLUMNS:
            values = pd.to_numeric(
                chunk[column],
                errors="coerce"
            )

            missing_counts[column] += int(
                values.isna().sum()
            )

            infinite_counts[column] += int(
                np.isinf(values.to_numpy()).sum()
            )

            chunk[column] = values.replace(
                [np.inf, -np.inf],
                np.nan
            )

        labels = chunk["Label"].astype(str).str.strip()
        label_counts.update(labels.value_counts().to_dict())

        affected = chunk[TARGET_COLUMNS].isna().any(axis=1)

        affected_labels.update(
            labels[affected].value_counts().to_dict()
        )

    return (
        total_rows,
        missing_counts,
        infinite_counts,
        label_counts,
        affected_labels
    )


def main():
    if not DATA_DIR.exists():
        raise FileNotFoundError(
            f"Dataset directory not found: {DATA_DIR}"
        )

    files = sorted(DATA_DIR.glob("*.csv"))

    if not files:
        raise FileNotFoundError(
            "No CSV files found in dataset directory."
        )

    print("=" * 70)
    print("AI-BASED IDS - DATASET QUALITY INSPECTION")
    print("=" * 70)

    for file_path in files:
        (
            rows,
            missing,
            infinite,
            labels,
            affected
        ) = inspect_file(file_path)

        print("\nFILE:", file_path.name)
        print("TOTAL ROWS:", f"{rows:,}")

        print("\nMISSING VALUES:")
        for column in TARGET_COLUMNS:
            print(f"{column}: {missing[column]:,}")

        print("\nINFINITE VALUES:")
        for column in TARGET_COLUMNS:
            print(f"{column}: {infinite[column]:,}")

        print("\nLABEL DISTRIBUTION:")
        for label, count in sorted(labels.items()):
            print(f"{label}: {count:,}")

        print("\nAFFECTED RECORDS BY LABEL:")
        for label, count in sorted(affected.items()):
            print(f"{label}: {count:,}")

        print("-" * 70)


if __name__ == "__main__":
    main()
