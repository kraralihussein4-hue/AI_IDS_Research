
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "processed"
REPORTS_DIR = ROOT / "reports"

MANIFEST_FILE = REPORTS_DIR / "conflict_exclusion_manifest.csv"
OUTPUT_FILE = REPORTS_DIR / "retained_dataset_class_counts.csv"

CHUNK_SIZE = 50_000


def row_key(values):
    values = np.asarray(values, dtype=np.float64).copy()

    # Normalize negative zero and missing values
    values[values == 0] = 0.0
    values[np.isnan(values)] = np.nan

    return values.tobytes()


def main():
    if not MANIFEST_FILE.exists():
        raise FileNotFoundError(MANIFEST_FILE)

    # Process original CSV files only; exclude the generated ML dataset
    csv_files = sorted(
        file for file in DATA_DIR.glob("*.csv")
        if file.name != "ml_dataset.csv"
    )

    if not csv_files:
        raise FileNotFoundError(
            f"No original CSV files found in {DATA_DIR}"
        )

    manifest = pd.read_csv(MANIFEST_FILE)

    metadata_columns = {
        "group_id",
        "handling_status",
        "total_occurrences",
        "number_of_labels",
        "label_counts",
        "example_locations",
    }

    feature_columns = [
        col for col in manifest.columns
        if col not in metadata_columns
    ]

    if not feature_columns:
        raise ValueError("No feature columns found in manifest")

    exclusion_keys = {
        row_key(row)
        for row in manifest[feature_columns].to_numpy()
    }

    original_counts = Counter()
    excluded_counts = Counter()

    for file_path in csv_files:
        print(f"Processing: {file_path.name}")

        for chunk in pd.read_csv(
            file_path,
            chunksize=CHUNK_SIZE,
            low_memory=False
        ):
            if "Label" not in chunk.columns:
                raise ValueError(
                    f"Label column missing in {file_path.name}"
                )

            if set(feature_columns) != (
                set(chunk.columns) - {"Label"}
            ):
                raise ValueError(
                    f"Feature columns do not match in {file_path.name}"
                )

            features = chunk[feature_columns].apply(
                pd.to_numeric,
                errors="coerce"
            ).to_numpy(dtype=np.float64)

            labels = chunk["Label"].astype(str).to_numpy()

            for values, label in zip(features, labels):
                original_counts[label] += 1

                if row_key(values) in exclusion_keys:
                    excluded_counts[label] += 1

    all_labels = sorted(original_counts)

    results = []

    for label in all_labels:
        original = original_counts[label]
        excluded = excluded_counts[label]

        results.append({
            "Label": label,
            "Original_Rows": original,
            "Excluded_Rows": excluded,
            "Retained_Rows": original - excluded,
        })

    report = pd.DataFrame(results)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    report.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    print("\nDataset exclusion analysis completed.")
    print(f"Original rows: {sum(original_counts.values()):,}")
    print(f"Excluded rows: {sum(excluded_counts.values()):,}")
    print(
        f"Retained rows: "
        f"{sum(original_counts.values()) - sum(excluded_counts.values()):,}"
    )
    print(f"\nReport saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
