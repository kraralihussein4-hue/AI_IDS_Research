
from pathlib import Path
from collections import Counter

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = BASE_DIR / "data" / "processed"

CHUNK_SIZE = 100_000


def main():

    if not DATA_DIR.exists():
        raise FileNotFoundError(
            f"Processed data folder not found: {DATA_DIR}"
        )

    files = sorted(DATA_DIR.glob("*.csv"))

    if not files:
        raise FileNotFoundError("No CSV files found!")

    # Maps each feature hash to its first observed label
    first_labels = {}

    total_rows = 0
    duplicate_rows = 0
    same_label_duplicates = 0
    conflicting_label_duplicates = 0

    file_rows = Counter()
    file_duplicates = Counter()

    print("=" * 70)
    print("CICIDS2017 DUPLICATE RECORDS ANALYSIS")
    print("=" * 70)

    print("Comparing feature columns across all CSV files.")
    print("The Label column is excluded from duplicate detection.")

    for file_path in files:

        print(f"\nProcessing: {file_path.name}")

        for chunk_number, chunk in enumerate(
            pd.read_csv(
                file_path,
                chunksize=CHUNK_SIZE,
                low_memory=False
            ),
            start=1
        ):

            chunk.columns = chunk.columns.str.strip()

            if "Label" not in chunk.columns:
                raise ValueError(
                    f"Label column missing in {file_path.name}"
                )

            labels = chunk["Label"].astype(str).tolist()

            features = chunk.drop(columns=["Label"])

            # Generate a 64-bit hash for each feature row
            hashes = pd.util.hash_pandas_object(
                features,
                index=False
            ).to_numpy()

            for row_hash, label in zip(hashes, labels):

                total_rows += 1
                file_rows[file_path.name] += 1

                key = int(row_hash)

                if key in first_labels:

                    duplicate_rows += 1
                    file_duplicates[file_path.name] += 1

                    if first_labels[key] == label:
                        same_label_duplicates += 1
                    else:
                        conflicting_label_duplicates += 1

                else:
                    first_labels[key] = label

            print(
                f"  Chunk {chunk_number}: "
                f"{len(chunk):,} rows checked"
            )

    unique_feature_rows = len(first_labels)

    print("\n" + "=" * 70)
    print("DUPLICATE ANALYSIS RESULTS")
    print("=" * 70)

    print(f"Total rows checked: {total_rows:,}")
    print(f"Unique feature hashes: {unique_feature_rows:,}")
    print(f"Repeated rows: {duplicate_rows:,}")

    print(
        f"Repeated rows with same label: "
        f"{same_label_duplicates:,}"
    )

    print(
        f"Repeated rows with conflicting labels: "
        f"{conflicting_label_duplicates:,}"
    )

    if total_rows:
        duplicate_percentage = (
            duplicate_rows / total_rows
        ) * 100

        print(
            f"Duplicate percentage: "
            f"{duplicate_percentage:.4f}%"
        )

    print("\nDuplicates by file:")

    for file_name in file_rows:
        print(
            f"  {file_name}: "
            f"{file_duplicates[file_name]:,} "
            f"repeated rows out of "
            f"{file_rows[file_name]:,}"
        )

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETED")
    print("=" * 70)

    print(
        "Note: Duplicate detection uses 64-bit row hashes. "
        "Potential hash collisions require exact verification."
    )

    print(
        "No source or processed dataset files were modified."
    )


if __name__ == "__main__":
    main()
