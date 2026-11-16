
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


# =====================================================
# PATHS
# =====================================================

ROOT_DIR = Path(__file__).resolve().parents[2]

MODEL_FILE = (
    ROOT_DIR
    / "experiments"
    / "random_forest_binary"
    / "random_forest_binary.joblib"
)


# =====================================================
# MODEL LOADING
# =====================================================

def load_model():

    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            f"Trained model not found: {MODEL_FILE}"
        )

    model = joblib.load(MODEL_FILE)

    if not hasattr(model, "feature_names_in_"):
        raise ValueError(
            "The model does not contain feature names."
        )

    return model


# =====================================================
# INPUT VALIDATION
# =====================================================

def prepare_input(data: pd.DataFrame, model):

    if not isinstance(data, pd.DataFrame):
        raise TypeError(
            "Input must be a pandas DataFrame."
        )

    if data.empty:
        raise ValueError(
            "The uploaded dataset is empty."
        )

    if data.columns.duplicated().any():
        raise ValueError(
            "Duplicate column names were detected."
        )

    feature_columns = list(model.feature_names_in_)

    # Ignore dataset metadata if present.
    metadata_columns = [
        "Label",
        "Split",
        "Binary_Label",
    ]

    input_data = data.drop(
        columns=[
            column
            for column in metadata_columns
            if column in data.columns
        ],
        errors="ignore",
    )

    missing_features = [
        column
        for column in feature_columns
        if column not in input_data.columns
    ]

    if missing_features:
        raise ValueError(
            "Missing required network features: "
            + ", ".join(missing_features)
        )

    # Keep the exact training feature order.
    X = input_data[feature_columns].copy()

    for column in feature_columns:

        X[column] = pd.to_numeric(
            X[column],
            errors="coerce",
        )

    X = X.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    return X


# =====================================================
# PREDICTION ENGINE
# =====================================================

def predict_traffic(data: pd.DataFrame):

    model = load_model()

    X = prepare_input(
        data,
        model,
    )

    predictions = model.predict(X)

    probabilities = model.predict_proba(X)

    classes = list(
        model.named_steps["classifier"].classes_
    )

    attack_index = classes.index("ATTACK")
    benign_index = classes.index("BENIGN")

    attack_probability = probabilities[:, attack_index]
    benign_probability = probabilities[:, benign_index]

    results = pd.DataFrame({
        "Prediction": predictions,
        "Attack_Probability": attack_probability,
        "Benign_Probability": benign_probability,
        "Confidence": np.maximum(
            attack_probability,
            benign_probability,
        ),
    })

    results["Prediction"] = results["Prediction"].astype(str)

    results["Threat_Level"] = np.where(
        results["Prediction"] == "ATTACK",
        "MALICIOUS",
        "BENIGN",
    )

    return results
