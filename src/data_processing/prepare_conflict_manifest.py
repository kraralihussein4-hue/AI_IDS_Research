
from pathlib import Path
import pandas as pd

# Project paths
ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"

SUMMARY_FILE = REPORTS / "conflict_group_summary.csv"
FEATURES_FILE = REPORTS / "conflict_group_features.csv"

OUTPUT_FILE = REPORTS / "conflict_exclusion_manifest.csv"
README_FILE = REPORTS / "conflict_handling_policy.txt"


def main():
    if not SUMMARY_FILE.exists():
        raise FileNotFoundError(f"Missing file: {SUMMARY_FILE}")

    if not FEATURES_FILE.exists():
        raise FileNotFoundError(f"Missing file: {FEATURES_FILE}")

    summary = pd.read_csv(SUMMARY_FILE)
    features = pd.read_csv(FEATURES_FILE)

    if "group_id" not in summary.columns:
        raise ValueError("group_id is missing from summary file")

    if "group_id" not in features.columns:
        raise ValueError("group_id is missing from features file")

    if summary["group_id"].duplicated().any():
        raise ValueError("Duplicate group_id values in summary")

    if features["group_id"].duplicated().any():
        raise ValueError("Duplicate group_id values in features")

    manifest = summary.merge(
        features,
        on="group_id",
        how="inner",
        validate="one_to_one"
    )

    if len(manifest) != len(summary) or len(manifest) != len(features):
        raise ValueError("Some conflict groups did not match between files")

    manifest.insert(1, "handling_status", "EXCLUDE_FROM_PRIMARY_EVALUATION")
    manifest.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    policy = """AI-Based IDS - Conflict Handling Policy

1. Original raw and processed datasets must remain unchanged.
2. Conflicting feature groups are documented in the exclusion manifest.
3. These groups are excluded from the primary supervised training
   and evaluation dataset until their labels can be independently verified.
4. No majority-vote relabeling is performed.
5. The excluded records must be reported transparently in the research.
6. Any future verification using PCAP evidence must be documented.

This policy is a research safeguard, not a claim that any conflicting
label is necessarily incorrect.
"""

    README_FILE.write_text(policy, encoding="utf-8")

    print("Conflict manifest created successfully.")
    print(f"Conflict groups documented: {len(manifest)}")
    print(f"Manifest: {OUTPUT_FILE}")
    print(f"Policy: {README_FILE}")
    print("Original datasets were not modified.")


if __name__ == "__main__":
    main()
