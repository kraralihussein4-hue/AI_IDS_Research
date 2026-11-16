
from pathlib import Path
import json

import joblib
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[2]

DATA_FILE = ROOT_DIR / "data" / "processed" / "ml_dataset.csv"

MODEL_FILE = (
    ROOT_DIR
    / "experiments"
    / "random_forest_binary"
    / "random_forest_binary.joblib"
)

OUTPUT_DIR = (
    ROOT_DIR
    / "experiments"
    / "random_forest_binary"
)

REPORT_FILE = OUTPUT_DIR / "attack_type_detection.csv"
METRICS_FILE = OUTPUT_DIR / "attack_type_metrics.json"


def main():

    print("=" * 60)
    print("ATTACK TYPE DETECTION EVALUATION")
    print("=" * 60)

    if not DATA_FILE.exists():
        raise FileNotFoundError(DATA_FILE)

    if not MODEL_FILE.exists():
        raise FileNotFoundError(MODEL_FILE)

    print("\nLoading dataset...")
    data = pd.read_csv(DATA_FILE, low_memory=False)

    test_data = data[data["Split"] == "test"].copy()

    excluded_columns = [
        "Label",
        "Split",
        "Binary_Label",
    ]

    feature_columns = [
        col for col in data.columns
        if col not in excluded_columns
    ]

    X_test = test_data[feature_columns].copy()

    for column in feature_columns:
        X_test[column] = pd.to_numeric(
            X_test[column],
            errors="coerce"
        )

    print("Loading trained model...")
    model = joblib.load(MODEL_FILE)

    print("\nPredicting test records...")
    predictions = model.predict(X_test)

    test_data["Predicted_Binary_Label"] = predictions

    # --------------------------------------------------
    # Overall false positive rate
    # --------------------------------------------------

    benign = test_data[
        test_data["Binary_Label"] == "BENIGN"
    ]

    false_positives = (
        benign["Predicted_Binary_Label"] == "ATTACK"
    ).sum()

    false_positive_rate = (
        false_positives / len(benign)
        if len(benign) else 0
    )

    # --------------------------------------------------
    # Detection rate for each attack type
    # --------------------------------------------------

    attack_data = test_data[
        test_data["Binary_Label"] == "ATTACK"
    ]

    results = []

    for label, group in attack_data.groupby("Label"):

        detected = (
            group["Predicted_Binary_Label"] == "ATTACK"
        ).sum()

        missed = len(group) - detected

        detection_rate = detected / len(group)

        results.append({
            "Attack_Type": label,
            "Total_Records": int(len(group)),
            "Detected": int(detected),
            "Missed": int(missed),
            "Detection_Rate": float(detection_rate),
        })

    report = pd.DataFrame(results)

    report = report.sort_values(
        "Detection_Rate",
        ascending=True
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    report.to_csv(
        REPORT_FILE,
        index=False
    )

    summary = {
        "test_records": int(len(test_data)),
        "attack_types": int(len(report)),
        "benign_records": int(len(benign)),
        "false_positives": int(false_positives),
        "false_positive_rate": float(false_positive_rate),
        "lowest_attack_detection_rate": float(
            report["Detection_Rate"].min()
        ),
        "highest_attack_detection_rate": float(
            report["Detection_Rate"].max()
        ),
    }

    with open(
        METRICS_FILE,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            summary,
            file,
            indent=4
        )

    print("\nATTACK TYPE RESULTS")
    print("=" * 60)

    print(
        report.to_string(
            index=False,
            formatters={
                "Detection_Rate": "{:.4%}".format
            }
        )
    )

    print("\nOVERALL FALSE POSITIVE RATE:")
    print(f"{false_positive_rate:.4%}")

    print("\nSaved report:")
    print(REPORT_FILE)

    print("\nSaved summary:")
    print(METRICS_FILE)

    print("\nEVALUATION COMPLETED")


if __name__ == "__main__":
    main()
