
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data" / "processed"
REPORT_DIR = BASE_DIR / "reports"

CHUNK_SIZE = 100_000


def get_hashes(features):
    return pd.util.hash_pandas_object(
        features,
        index=False
    ).to_numpy()


def first_pass(files):
    """
    Identify hashes associated with more than one label.
    """

    first_labels = {}
    candidate_labels = defaultdict(set)

    total_rows = 0

    print("\nPASS 1: Finding candidate conflicts")

    for file_path in files:

        print(f"\nScanning: {file_path.name}")
        file_row_number = 0

        for chunk in pd.read_csv(
            file_path,
            chunksize=CHUNK_SIZE,
            low_memory=False
        ):

            chunk.columns = chunk.columns.str.strip()

            features = chunk.drop(columns=["Label"])
            labels = chunk["Label"].astype(str).to_numpy()

            hashes = get_hashes(features)

            for row_hash, label in zip(hashes, labels):

                key = int(row_hash)

                if key not in first_labels:
                    first_labels[key] = label

                elif first_labels[key] != label:
                    candidate_labels[key].add(first_labels[key])
                    candidate_labels[key].add(label)

            total_rows += len(chunk)
            file_row_number += len(chunk)

            print(
                f"  Rows scanned in file: "
                f"{file_row_number:,}"
            )

    print(f"\nTotal rows scanned: {total_rows:,}")
    print(f"Candidate hashes: {len(candidate_labels):,}")

    return candidate_labels


def second_pass(files, candidate_labels):
    """
    Verify candidate hashes by comparing actual feature arrays.
    """

    verified_groups = defaultdict(list)

    print("\nPASS 2: Verifying exact feature equality")

    for file_path in files:

        print(f"\nVerifying: {file_path.name}")

        row_offset = 0

        for chunk in pd.read_csv(
            file_path,
            chunksize=CHUNK_SIZE,
            low_memory=False
        ):

            chunk.columns = chunk.columns.str.strip()

            features = chunk.drop(columns=["Label"])
            labels = chunk["Label"].astype(str).to_numpy()

            hashes = get_hashes(features)
            values = features.to_numpy(dtype=np.float64)

            for i, row_hash in enumerate(hashes):

                key = int(row_hash)

                if key not in candidate_labels:
                    continue

                label = labels[i]
                row_values = values[i]

                groups = verified_groups[key]

                matched_group = None

                for group in groups:

                    if np.array_equal(
                        group["features"],
                        row_values,
                        equal_nan=True
                    ):
                        matched_group = group
                        break

                location = (
                    file_path.name,
                    row_offset + i + 1
                )

                if matched_group is None:

                    groups.append({
                        "features": row_values.copy(),
                        "labels": {label},
                        "locations": [location],
                        "count": 1
                    })

                else:

                    matched_group["labels"].add(label)
                    matched_group["count"] += 1

                    if len(matched_group["locations"]) < 10:
                        matched_group["locations"].append(location)

            row_offset += len(chunk)

            print(
                f"  Rows checked in file: "
                f"{row_offset:,}"
            )

    return verified_groups


def save_report(verified_groups):

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    report_path = (
        REPORT_DIR / "verified_duplicate_conflicts.csv"
    )

    records = []

    for groups in verified_groups.values():

        for group in groups:

            if len(group["labels"]) <= 1:
                continue

            locations = [
                f"{name} (row {row_number})"
                for name, row_number in group["locations"]
            ]

            records.append({
                "labels": " | ".join(
                    sorted(group["labels"])
                ),
                "occurrences": group["count"],
                "different_labels": len(group["labels"]),
                "example_locations": " ; ".join(locations)
            })

    columns = [
        "labels",
        "occurrences",
        "different_labels",
        "example_locations"
    ]

    report = pd.DataFrame(
        records,
        columns=columns
    )

    report.to_csv(
        report_path,
        index=False,
        encoding="utf-8-sig"
    )

    return report, report_path


def main():

    if not DATA_DIR.exists():
        raise FileNotFoundError(
            f"Processed folder not found: {DATA_DIR}"
        )

    files = sorted(DATA_DIR.glob("*.csv"))

    if not files:
        raise FileNotFoundError(
            "No processed CSV files found."
        )

    print("=" * 70)
    print("EXACT DUPLICATE CONFLICT VERIFICATION")
    print("=" * 70)

    candidate_labels = first_pass(files)

    if not candidate_labels:

        print("\nNo candidate conflicts found.")
        return

    verified_groups = second_pass(
        files,
        candidate_labels
    )

    report, report_path = save_report(
        verified_groups
    )

    total_conflicting_groups = len(report)

    total_conflicting_occurrences = (
        int(report["occurrences"].sum())
        if not report.empty
        else 0
    )

    print("\n" + "=" * 70)
    print("EXACT VERIFICATION RESULTS")
    print("=" * 70)

    print(
        f"Candidate hashes: "
        f"{len(candidate_labels):,}"
    )

    print(
        f"Verified conflicting feature groups: "
        f"{total_conflicting_groups:,}"
    )

    print(
        f"Occurrences in verified conflicting groups: "
        f"{total_conflicting_occurrences:,}"
    )

    print(f"\nReport saved to:\n{report_path}")

    print("\nNo dataset records were deleted or modified.")

    print("=" * 70)
    print("VERIFICATION COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()
