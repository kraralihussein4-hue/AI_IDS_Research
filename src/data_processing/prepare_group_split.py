
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedGroupKFold


# =========================================================
# 1. PROJECT PATHS AND SETTINGS
# =========================================================

ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data" / "processed"
REPORTS_DIR = ROOT / "reports"

MANIFEST_FILE = (
    REPORTS_DIR / "conflict_exclusion_manifest.csv"
)

OUTPUT_FILE = (
    REPORTS_DIR / "group_split_indices.csv"
)

SUMMARY_FILE = (
    REPORTS_DIR / "group_split_summary.csv"
)

CHUNK_SIZE = 50_000
N_SPLITS = 5
RANDOM_STATE = 42


# =========================================================
# 2. FEATURE FINGERPRINT
# =========================================================

def row_key(values):
    """
    Create a stable fingerprint for a feature row.

    Negative zero and missing values are normalized.
    """

    values = np.asarray(
        values,
        dtype=np.float64
    ).copy()

    values[values == 0] = 0.0
    values[np.isnan(values)] = np.nan

    return values.tobytes()


# =========================================================
# 3. LOAD CONFLICT EXCLUSION KEYS
# =========================================================

def load_exclusion_keys(feature_columns):

    if not MANIFEST_FILE.exists():
        raise FileNotFoundError(MANIFEST_FILE)

    manifest = pd.read_csv(
        MANIFEST_FILE,
        low_memory=False
    )

    missing_columns = [
        col
        for col in feature_columns
        if col not in manifest.columns
    ]

    if missing_columns:
        raise ValueError(
            "Manifest is missing feature columns: "
            f"{missing_columns}"
        )

    features = manifest[feature_columns].apply(
        pd.to_numeric,
        errors="coerce"
    ).to_numpy(dtype=np.float64)

    exclusion_keys = {
        row_key(row)
        for row in features
    }

    print(
        f"Conflict groups loaded: "
        f"{len(exclusion_keys):,}"
    )

    return exclusion_keys


# =========================================================
# 4. PREPARE GROUP-BASED DATA
# =========================================================

def prepare_data():

    if not DATA_DIR.exists():
        raise FileNotFoundError(DATA_DIR)

    csv_files = sorted(
        DATA_DIR.glob("*.csv")
    )

    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found in {DATA_DIR}"
        )

    first_chunk = pd.read_csv(
        csv_files[0],
        nrows=5,
        low_memory=False
    )

    if "Label" not in first_chunk.columns:
        raise ValueError(
            "Label column is missing"
        )

    feature_columns = [
        col
        for col in first_chunk.columns
        if col != "Label"
    ]

    exclusion_keys = load_exclusion_keys(
        feature_columns
    )

    group_lookup = {}

    group_ids = []
    row_labels = []
    row_files = []
    row_numbers = []

    global_row_number = 0
    excluded_rows = 0

    expected_columns = set(
        feature_columns + ["Label"]
    )

    # -----------------------------------------------------
    # Read CSV files in chunks
    # -----------------------------------------------------

    for file_path in csv_files:

        print(
            f"\nProcessing: {file_path.name}"
        )

        file_row_number = 0

        for chunk in pd.read_csv(
            file_path,
            chunksize=CHUNK_SIZE,
            low_memory=False
        ):

            if set(chunk.columns) != expected_columns:
                raise ValueError(
                    f"Column mismatch in {file_path.name}"
                )

            features = chunk[
                feature_columns
            ].apply(
                pd.to_numeric,
                errors="coerce"
            ).to_numpy(dtype=np.float64)

            labels = chunk[
                "Label"
            ].astype(str).to_numpy()

            for values, label in zip(
                features,
                labels
            ):

                current_file_row = file_row_number

                file_row_number += 1
                global_row_number += 1

                key = row_key(values)

                # Exclude conflicting feature patterns
                if key in exclusion_keys:

                    excluded_rows += 1
                    continue

                # Assign the same group ID to identical rows
                if key not in group_lookup:

                    group_lookup[key] = len(
                        group_lookup
                    )

                group_id = group_lookup[key]

                group_ids.append(group_id)
                row_labels.append(label)
                row_files.append(file_path.name)
                row_numbers.append(current_file_row)

    if not group_ids:
        raise ValueError(
            "No retained rows available for splitting"
        )

    print("\nData preparation completed.")

    return (
        np.asarray(group_ids, dtype=np.int64),
        np.asarray(row_labels, dtype=str),
        row_files,
        row_numbers,
        global_row_number,
        excluded_rows,
        len(group_lookup)
    )


# =========================================================
# 5. STRATIFIED GROUP SPLIT
# =========================================================

def create_group_split(
    groups,
    labels
):

    unique_group_count = len(
        np.unique(groups)
    )

    if unique_group_count < N_SPLITS:
        raise ValueError(
            "Not enough unique groups for 5-fold splitting"
        )

    splitter = StratifiedGroupKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    train_indices, test_indices = next(
        splitter.split(
            X=np.zeros(len(groups)),
            y=labels,
            groups=groups
        )
    )

    train_groups = set(
        groups[train_indices]
    )

    test_groups = set(
        groups[test_indices]
    )

    overlap = train_groups.intersection(
        test_groups
    )

    if overlap:
        raise RuntimeError(
            "Data leakage detected: "
            "groups appear in both sets"
        )

    return train_indices, test_indices


# =========================================================
# 6. SAVE SPLIT FILES AND SUMMARY
# =========================================================

def save_results(
    groups,
    labels,
    row_files,
    row_numbers,
    train_indices,
    test_indices,
    original_rows,
    excluded_rows,
    unique_groups
):

    split_labels = np.full(
        len(groups),
        "train",
        dtype=object
    )

    split_labels[test_indices] = "test"

    output = pd.DataFrame({

        "Source_File": row_files,

        "Row_Number": row_numbers,

        "Group_ID": groups,

        "Label": labels,

        "Split": split_labels

    })

    output.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    summary = (
        output.groupby(
            ["Split", "Label"]
        )
        .size()
        .reset_index(
            name="Rows"
        )
    )

    summary.to_csv(
        SUMMARY_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    # -----------------------------------------------------
    # Print final validation
    # -----------------------------------------------------

    train_groups = set(
        groups[train_indices]
    )

    test_groups = set(
        groups[test_indices]
    )

    overlap = train_groups.intersection(
        test_groups
    )

    print("\n" + "=" * 55)
    print("GROUP-BASED DATA SPLIT RESULTS")
    print("=" * 55)

    print(
        f"Original rows scanned: "
        f"{original_rows:,}"
    )

    print(
        f"Excluded conflict rows: "
        f"{excluded_rows:,}"
    )

    print(
        f"Retained rows: "
        f"{len(groups):,}"
    )

    print(
        f"Unique feature groups: "
        f"{unique_groups:,}"
    )

    print(
        f"Training rows: "
        f"{len(train_indices):,}"
    )

    print(
        f"Testing rows: "
        f"{len(test_indices):,}"
    )

    print(
        f"Training percentage: "
        f"{len(train_indices) / len(groups) * 100:.2f}%"
    )

    print(
        f"Testing percentage: "
        f"{len(test_indices) / len(groups) * 100:.2f}%"
    )

    print(
        f"Shared groups: "
        f"{len(overlap)}"
    )

    if len(overlap) == 0:
        print(
            "\nPASS: No identical feature groups "
            "are shared between train and test."
        )

    print(
        f"\nSplit file: {OUTPUT_FILE}"
    )

    print(
        f"Summary file: {SUMMARY_FILE}"
    )

    print("=" * 55)


# =========================================================
# 7. MAIN
# =========================================================

def main():

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    (
        groups,
        labels,
        row_files,
        row_numbers,
        original_rows,
        excluded_rows,
        unique_groups
    ) = prepare_data()

    train_indices, test_indices = create_group_split(
        groups,
        labels
    )

    save_results(
        groups,
        labels,
        row_files,
        row_numbers,
        train_indices,
        test_indices,
        original_rows,
        excluded_rows,
        unique_groups
    )


if __name__ == "__main__":
    main()
