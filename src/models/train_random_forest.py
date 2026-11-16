
from pathlib import Path
import json
import time

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
)
from sklearn.pipeline import Pipeline


# --------------------------------------------------
# 1. Project paths
# --------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parents[2]

DATA_FILE = ROOT_DIR / "data" / "processed" / "ml_dataset.csv"

OUTPUT_DIR = ROOT_DIR / "experiments" / "random_forest_binary"

MODEL_FILE = OUTPUT_DIR / "random_forest_binary.joblib"
METRICS_FILE = OUTPUT_DIR / "metrics.json"
REPORT_FILE = OUTPUT_DIR / "classification_report.csv"
CONFUSION_FILE = OUTPUT_DIR / "confusion_matrix.csv"
IMPORTANCE_FILE = OUTPUT_DIR / "feature_importance.csv"


# --------------------------------------------------
# 2. Load dataset
# --------------------------------------------------

def main():

    start_time = time.time()

    print("=" * 60)
    print("RANDOM FOREST - BINARY IDS CLASSIFICATION")
    print("=" * 60)

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_FILE}"
        )

    print("\nLoading dataset...")

    data = pd.read_csv(DATA_FILE, low_memory=False)

    required_columns = {
        "Label",
        "Split",
        "Binary_Label",
    }

    missing_columns = required_columns - set(data.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    train_data = data[data["Split"] == "train"].copy()
    test_data = data[data["Split"] == "test"].copy()

    print(f"Training rows: {len(train_data):,}")
    print(f"Testing rows: {len(test_data):,}")

    # --------------------------------------------------
    # 3. Prepare features and labels
    # --------------------------------------------------

    excluded_columns = [
        "Label",
        "Split",
        "Binary_Label",
    ]

    feature_columns = [
        column
        for column in data.columns
        if column not in excluded_columns
    ]

    X_train = train_data[feature_columns].copy()
    X_test = test_data[feature_columns].copy()

    y_train = train_data["Binary_Label"]
    y_test = test_data["Binary_Label"]

    allowed_labels = {"BENIGN", "ATTACK"}

    if not set(y_train.unique()).issubset(allowed_labels):
        raise ValueError("Unexpected training labels found.")

    if not set(y_test.unique()).issubset(allowed_labels):
        raise ValueError("Unexpected testing labels found.")

    if y_train.isna().any() or y_test.isna().any():
        raise ValueError("Missing target labels detected.")

    # Ensure all features are numeric.
    for column in feature_columns:
        X_train[column] = pd.to_numeric(
            X_train[column],
            errors="coerce"
        )

        X_test[column] = pd.to_numeric(
            X_test[column],
            errors="coerce"
        )

    # Replace infinite values with missing values.
    X_train = X_train.replace(
        [np.inf, -np.inf],
        np.nan
    )

    X_test = X_test.replace(
        [np.inf, -np.inf],
        np.nan
    )

    print(f"\nNumber of features: {len(feature_columns)}")

    print("\nTraining class distribution:")
    print(y_train.value_counts())

    print("\nTesting class distribution:")
    print(y_test.value_counts())

    # --------------------------------------------------
    # 4. Build machine learning pipeline
    # --------------------------------------------------

    model = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median")
            ),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=100,
                    max_depth=24,
                    min_samples_leaf=2,
                    class_weight="balanced_subsample",
                    random_state=42,
                    n_jobs=-1,
                    verbose=1,
                )
            ),
        ]
    )

    # --------------------------------------------------
    # 5. Train model
    # --------------------------------------------------

    print("\nStarting model training...")
    print("Please wait. Training may take some time.")

    model.fit(X_train, y_train)

    print("\nTraining completed successfully.")

    # --------------------------------------------------
    # 6. Evaluate model
    # --------------------------------------------------

    print("\nEvaluating model on the test dataset...")

    predictions = model.predict(X_test)

    probabilities = model.predict_proba(X_test)

    attack_index = list(
        model.named_steps["classifier"].classes_
    ).index("ATTACK")

    attack_probabilities = probabilities[:, attack_index]

    metrics = {
        "model": "Random Forest",
        "task": "Binary Classification",
        "training_rows": int(len(X_train)),
        "testing_rows": int(len(X_test)),
        "features": int(len(feature_columns)),
        "accuracy": float(
            accuracy_score(y_test, predictions)
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_test, predictions)
        ),
        "precision_attack": float(
            precision_score(
                y_test,
                predictions,
                pos_label="ATTACK",
                zero_division=0
            )
        ),
        "recall_attack": float(
            recall_score(
                y_test,
                predictions,
                pos_label="ATTACK",
                zero_division=0
            )
        ),
        "f1_attack": float(
            f1_score(
                y_test,
                predictions,
                pos_label="ATTACK",
                zero_division=0
            )
        ),
        "macro_f1": float(
            f1_score(
                y_test,
                predictions,
                average="macro",
                zero_division=0
            )
        ),
        "roc_auc": float(
            roc_auc_score(
                (y_test == "ATTACK").astype(int),
                attack_probabilities
            )
        ),
        "average_precision": float(
            average_precision_score(
                (y_test == "ATTACK").astype(int),
                attack_probabilities
            )
        ),
    }

    # --------------------------------------------------
    # 7. Save evaluation reports
    # --------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    report = classification_report(
        y_test,
        predictions,
        labels=["BENIGN", "ATTACK"],
        output_dict=True,
        zero_division=0
    )

    pd.DataFrame(report).transpose().to_csv(
        REPORT_FILE,
        index=True
    )

    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=["BENIGN", "ATTACK"]
    )

    confusion_df = pd.DataFrame(
        matrix,
        index=["Actual_BENIGN", "Actual_ATTACK"],
        columns=["Predicted_BENIGN", "Predicted_ATTACK"]
    )

    confusion_df.to_csv(CONFUSION_FILE)

    # --------------------------------------------------
    # 8. Save feature importance
    # --------------------------------------------------

    classifier = model.named_steps["classifier"]

    importance_df = pd.DataFrame({
        "Feature": feature_columns,
        "Importance": classifier.feature_importances_,
    })

    importance_df = importance_df.sort_values(
        by="Importance",
        ascending=False
    )

    importance_df.to_csv(
        IMPORTANCE_FILE,
        index=False
    )

    # --------------------------------------------------
    # 9. Save model and metrics
    # --------------------------------------------------

    joblib.dump(model, MODEL_FILE)

    elapsed_time = time.time() - start_time

    metrics["training_and_evaluation_seconds"] = round(
        elapsed_time,
        2
    )

    with open(
        METRICS_FILE,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            metrics,
            file,
            indent=4
        )

    # --------------------------------------------------
    # 10. Display results
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("MODEL EVALUATION RESULTS")
    print("=" * 60)

    for name, value in metrics.items():
        print(f"{name}: {value}")

    print("\nConfusion Matrix:")
    print(confusion_df)

    print("\nTop 10 Important Features:")
    print(importance_df.head(10).to_string(index=False))

    print("\nSaved files:")
    print(MODEL_FILE)
    print(METRICS_FILE)
    print(REPORT_FILE)
    print(CONFUSION_FILE)
    print(IMPORTANCE_FILE)

    print("\n" + "=" * 60)
    print("RANDOM FOREST EXPERIMENT COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()
