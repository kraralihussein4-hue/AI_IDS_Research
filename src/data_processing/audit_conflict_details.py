
from pathlib import Path
from collections import Counter
import json

import numpy as np
import pandas as pd


# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"

CHUNK_SIZE = 100_000
MAX_LOCATIONS_PER_GROUP = 10


def get_csv_files():
    """Return all processed CSV files."""
    files = sorted(DATA_DIR.glob("*.csv"))

    if not files:
        raise FileNotFoundError(
            f"No CSV files found in: {DATA_DIR}"
        )

    return files


def get_feature_columns(file_path):
    """Read the CSV header and identify feature columns."""
    columns = pd.read_csv(file_path, nrows=0).columns.tolist()

    if "Label" not in columns:
        raise ValueError(
            f"'Label' column not found in {file_path.name}"
        )

    return [column for column in columns if column != "Label"]


def row_hashes(features):
    """Create a hash for every feature row."""
    return pd.util.hash_pandas_object(
        features,
        index=False
    ).to_numpy(dtype=np.uint64)


def find_candidate_hashes(files, feature_columns):
    """
    First pass:
    Find hashes that appear with more than one label.
    Hashes are only candidates; exact equality is checked later.
    """
    hash_labels = {}

    print("\n[1/2] Searching for candidate conflicting rows...")

    for file_path in files:
        print(f"Reading: {file_path.name}")

        for chunk in pd.read_csv(
            file_path,
            chunksize=CHUNK_SIZE,
            low_memory=False
        ):
            features = chunk[feature_columns]
            labels = chunk["Label"].fillna("UNKNOWN").astype(str)

            hashes = row_hashes(features)

            for row_hash, label in zip(hashes, labels):
                if row_hash not in hash_labels:
                    hash_labels[row_hash] = set()

                hash_labels[row_hash].add(label)

    candidates = {
        row_hash
        for row_hash, labels in hash_labels.items()
        if len(labels) > 1
    }

    print(f"Candidate hashes found: {len(candidates):,}")

    return candidates


def make_exact_key(row):
    """
    Convert a numeric feature row into a stable exact key.
    NaN values and positive/negative zero are normalized.
    """
    values = np.asarray(row, dtype=np.float64).copy()

    values[np.isnan(values)] = np.nan
    values[values == 0] = 0.0

    return values.tobytes(), values


def audit_exact_conflicts(files, feature_columns, candidate_hashes):
    """
    Second pass:
    Verify exact feature equality and count occurrences per label.
    """
    groups = {}

    print("\n[2/2] Verifying exact feature matches...")

    for file_path in files:
        print(f"Auditing: {file_path.name}")

        csv_row_number = 2  # First data row; row 1 is the header.

        for chunk in pd.read_csv(
            file_path,
            chunksize=CHUNK_SIZE,
            low_memory=False
        ):
            features = chunk[feature_columns]
            labels = chunk["Label"].fillna("UNKNOWN").astype(str)

            hashes = row_hashes(features)

            for position, (row_hash, label) in enumerate(
                zip(hashes, labels)
            ):
                current_csv_row = csv_row_number + position

                if row_hash not in candidate_hashes:
                    continue

                exact_key, values = make_exact_key(
                    features.iloc[position].to_numpy()
                )

                if exact_key not in groups:
                    groups[exact_key] = {
                        "features": values,
                        "label_counts": Counter(),
                        "locations": []
                    }

                group = groups[exact_key]
                group["label_counts"][label] += 1

                if len(group["locations"]) < MAX_LOCATIONS_PER_GROUP:
                    group["locations"].append(
                        f"{file_path.name}: row {current_csv_row}"
                    )

            csv_row_number += len(chunk)

    # Keep only exact groups that genuinely have multiple labels.
    exact_conflicts = {
        key: group
        for key, group in groups.items()
        if len(group["label_counts"]) > 1
    }

    return exact_conflicts


def save_reports(conflicts, feature_columns):
    """Save a summary report and a separate feature-values report."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    feature_rows = []

    ordered_groups = sorted(
        conflicts.values(),
        key=lambda group: sum(group["label_counts"].values()),
        reverse=True
    )

    for group_number, group in enumerate(ordered_groups, start=1):
        group_id = f"CONFLICT_{group_number:04d}"

        label_counts = dict(
            sorted(group["label_counts"].items())
        )

        total_occurrences = sum(label_counts.values())

        summary_rows.append({
            "group_id": group_id,
            "total_occurrences": total_occurrences,
            "number_of_labels": len(label_counts),
            "label_counts": json.dumps(
                label_counts,
                ensure_ascii=False
            ),
            "example_locations": " | ".join(group["locations"])
        })

        feature_record = {
            "group_id": group_id
        }

        feature_record.update(
            dict(zip(feature_columns, group["features"]))
        )

        feature_rows.append(feature_record)

    summary_path = REPORTS_DIR / "conflict_group_summary.csv"
    features_path = REPORTS_DIR / "conflict_group_features.csv"

    pd.DataFrame(summary_rows).to_csv(
        summary_path,
        index=False,
        encoding="utf-8-sig"
    )

    pd.DataFrame(
        feature_rows,
        columns=["group_id"] + feature_columns
    ).to_csv(
        features_path,
        index=False,
        encoding="utf-8-sig"
    )

    print("\nReports saved successfully:")
    print(summary_path)
    print(features_path)

    total_occurrences = sum(
        sum(group["label_counts"].values())
        for group in conflicts.values()
    )

    print("\n========== AUDIT SUMMARY ==========")
    print(f"Exact conflicting groups: {len(conflicts):,}")
    print(f"Total occurrences in these groups: {total_occurrences:,}")
    print(f"Feature columns: {len(feature_columns)}")
    print("Original and processed datasets were not modified.")
    print("===================================")


def main():
    print("AI-Based IDS - Duplicate Conflict Audit")
    print("----------------------------------------")

    files = get_csv_files()
    feature_columns = get_feature_columns(files[0])

    print(f"CSV files found: {len(files)}")
    print(f"Feature columns: {len(feature_columns)}")

    candidates = find_candidate_hashes(
        files,
        feature_columns
    )

    if not candidates:
        print("\nNo candidate conflicts found.")
        return

    conflicts = audit_exact_conflicts(
        files,
        feature_columns,
        candidates
    )

    save_reports(
        conflicts,
        feature_columns
    )


if __name__ == "__main__":
    main()
