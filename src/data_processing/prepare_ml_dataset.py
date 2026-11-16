
from pathlib import Path
from collections import defaultdict

import pandas as pd


# =====================================================
# 1. PROJECT SETTINGS
# =====================================================

ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data" / "processed"
REPORTS_DIR = ROOT / "reports"

SPLIT_FILE = REPORTS_DIR / "group_split_indices.csv"

OUTPUT_FILE = DATA_DIR / "ml_dataset.csv"
TEMP_FILE = DATA_DIR / "ml_dataset.tmp"

SUMMARY_FILE = REPORTS_DIR / "ml_dataset_summary.csv"

CHUNK_SIZE = 50_000
TRAIN_SAMPLE_SIZE = 200_000
RANDOM_STATE = 42


# =====================================================
# 2. STRATIFIED SAMPLE ALLOCATION
# =====================================================

def allocate_samples(counts, target_size):

    total = int(counts.sum())

    if target_size > total:
        raise ValueError(
            "Requested sample exceeds available training rows."
        )

    proportions = counts / total
    quotas = (proportions * target_size).astype(int)

    for label in counts.index:
        if counts[label] > 0 and quotas[label] == 0:
            quotas[label] = 1

    for label in counts.index:
        quotas[label] = min(
            int(quotas[label]),
            int(counts[label])
        )

    while int(quotas.sum()) < target_size:

        candidates = [
            label for label in counts.index
            if quotas[label] < counts[label]
        ]

        if not candidates:
            raise ValueError(
                "Unable to allocate requested sample size."
            )

        candidates.sort(
            key=lambda label: (
                proportions[label] * target_size
                - quotas[label]
            ),
            reverse=True
        )

        for label in candidates:
            if int(quotas.sum()) >= target_size:
                break

            quotas[label] += 1

    return quotas.astype(int)


# =====================================================
# 3. SELECT TRAINING AND TESTING ROWS
# =====================================================

def select_rows():

    if not SPLIT_FILE.exists():
        raise FileNotFoundError(SPLIT_FILE)

    print("Loading split information...")

    metadata = pd.read_csv(
        SPLIT_FILE,
        usecols=[
            "Source_File",
            "Row_Number",
            "Label",
            "Split"
        ],
        low_memory=False
    )

    if metadata["Row_Number"].isna().any():
        raise ValueError("Missing row numbers found.")

    if metadata["Label"].isna().any():
        raise ValueError("Missing labels found.")

    if metadata.duplicated(
        ["Source_File", "Row_Number"]
    ).any():
        raise ValueError("Duplicate source row references found.")

    train_metadata = metadata[
        metadata["Split"] == "train"
    ]

    test_metadata = metadata[
        metadata["Split"] == "test"
    ]

    if train_metadata.empty or test_metadata.empty:
        raise ValueError("Training or testing data is empty.")

    print("Selecting stratified training sample...")

    class_counts = train_metadata["Label"].value_counts()

    quotas = allocate_samples(
        class_counts,
        TRAIN_SAMPLE_SIZE
    )

    selected_parts = []

    for label, group in train_metadata.groupby("Label"):

        selected_parts.append(
            group.sample(
                n=int(quotas[label]),
                random_state=RANDOM_STATE
            )
        )

    selected_train = pd.concat(
        selected_parts,
        ignore_index=True
    )

    print(
        f"Selected training rows: {len(selected_train):,}"
    )

    print(
        f"All testing rows retained: {len(test_metadata):,}"
    )

    train_rows = defaultdict(set)
    test_rows = defaultdict(set)

    for file_name, row_number in zip(
        selected_train["Source_File"],
        selected_train["Row_Number"]
    ):
        train_rows[file_name].add(int(row_number))

    for file_name, row_number in zip(
        test_metadata["Source_File"],
        test_metadata["Row_Number"]
    ):
        test_rows[file_name].add(int(row_number))

    for file_name in set(train_rows) | set(test_rows):

        overlap = (
            train_rows[file_name]
            & test_rows[file_name]
        )

        if overlap:
            raise RuntimeError(
                f"Train/test row overlap in {file_name}"
            )

    del metadata
    del train_metadata
    del test_metadata
    del selected_train

    return train_rows, test_rows


# =====================================================
# 4. EXTRACT DATA FROM SOURCE FILES
# =====================================================

def build_dataset(train_rows, test_rows):

    csv_files = sorted(
        path
        for path in DATA_DIR.glob("*.csv")
        if path.name != OUTPUT_FILE.name
    )

    if not csv_files:
        raise FileNotFoundError(
            f"No source CSV files found in {DATA_DIR}"
        )

    if OUTPUT_FILE.exists():
        raise FileExistsError(
            f"Output already exists: {OUTPUT_FILE}"
        )

    if TEMP_FILE.exists():
        raise FileExistsError(
            f"Temporary file already exists: {TEMP_FILE}"
        )

    first_write = True
    total_train = 0
    total_test = 0
    summary_parts = []

    try:

        for file_path in csv_files:

            print(f"\nProcessing: {file_path.name}")

            train_set = train_rows.get(
                file_path.name, set()
            )

            test_set = test_rows.get(
                file_path.name, set()
            )

            if train_set.intersection(test_set):
                raise RuntimeError(
                    f"Row overlap found in {file_path.name}"
                )

            file_train_count = 0
            file_test_count = 0

            for chunk in pd.read_csv(
                file_path,
                chunksize=CHUNK_SIZE,
                low_memory=False
            ):

                # Use the original row indexes supplied by Pandas.
                train_mask = chunk.index.isin(train_set)
                test_mask = chunk.index.isin(test_set)

                selected_mask = train_mask | test_mask

                if not selected_mask.any():
                    continue

                selected = chunk.loc[selected_mask].copy()

                selected["Split"] = "test"

                selected.loc[
                    train_mask[selected_mask],
                    "Split"
                ] = "train"

                selected["Binary_Label"] = (
                    selected["Label"]
                    .ne("BENIGN")
                    .map({
                        True: "ATTACK",
                        False: "BENIGN"
                    })
                )

                file_train_count += int(
                    (selected["Split"] == "train").sum()
                )

                file_test_count += int(
                    (selected["Split"] == "test").sum()
                )

                summary_parts.append(
                    selected.groupby(
                        ["Split", "Label"]
                    ).size().reset_index(name="Rows")
                )

                selected.to_csv(
                    TEMP_FILE,
                    mode="w" if first_write else "a",
                    header=first_write,
                    index=False,
                    encoding="utf-8-sig"
                )

                first_write = False

            # Validate extracted row counts for this file.
            if file_train_count != len(train_set):
                raise RuntimeError(
                    f"Training row mismatch in {file_path.name}: "
                    f"expected {len(train_set):,}, "
                    f"found {file_train_count:,}"
                )

            if file_test_count != len(test_set):
                raise RuntimeError(
                    f"Testing row mismatch in {file_path.name}: "
                    f"expected {len(test_set):,}, "
                    f"found {file_test_count:,}"
                )

            total_train += file_train_count
            total_test += file_test_count

            print(f"Training rows: {file_train_count:,}")
            print(f"Testing rows: {file_test_count:,}")

            train_set.clear()
            test_set.clear()

        if first_write:
            raise ValueError("No rows were extracted.")

        if total_train != TRAIN_SAMPLE_SIZE:
            raise RuntimeError(
                f"Expected {TRAIN_SAMPLE_SIZE:,} training rows, "
                f"but extracted {total_train:,}."
            )

        if total_test == 0:
            raise RuntimeError("No testing rows were extracted.")

        summary = (
            pd.concat(summary_parts, ignore_index=True)
            .groupby(
                ["Split", "Label"],
                as_index=False
            )["Rows"]
            .sum()
        )

        # Publish the dataset after successful validation.
        TEMP_FILE.replace(OUTPUT_FILE)

        summary.to_csv(
            SUMMARY_FILE,
            index=False,
            encoding="utf-8-sig"
        )

        print("\n" + "=" * 55)
        print("ML DATASET PREPARATION COMPLETED")
        print("=" * 55)

        print(f"Training rows: {total_train:,}")
        print(f"Testing rows: {total_test:,}")
        print(f"Total rows: {total_train + total_test:,}")

        print(f"\nDataset: {OUTPUT_FILE}")
        print(f"Summary: {SUMMARY_FILE}")

        print("=" * 55)

    except Exception:

        print(
            "\nDataset preparation failed. "
            "Original source files were not modified."
        )

        raise


# =====================================================
# 5. MAIN
# =====================================================

def main():

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    train_rows, test_rows = select_rows()

    build_dataset(
        train_rows,
        test_rows
    )


if __name__ == "__main__":
    main()
