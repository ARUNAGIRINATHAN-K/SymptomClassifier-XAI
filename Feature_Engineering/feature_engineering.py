import os
from typing import Tuple, Dict, List, Optional

import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder
import joblib


def load_data(csv_path: str) -> pd.DataFrame:
    """Load CSV into a pandas DataFrame."""
    return pd.read_csv(csv_path)


def preprocess_features(
    df: pd.DataFrame,
    fit: bool = True,
    mlb: Optional[MultiLabelBinarizer] = None,
) -> Tuple[pd.DataFrame, Optional[np.ndarray], Dict]:
    """
    Convert raw symptom text and patient attributes into model-ready features.

    Steps performed:
    - Split `Symptoms` into lists
    - Multi-hot encode symptoms with `MultiLabelBinarizer`
    - Create `Symptom_Count` feature
    - Encode `Gender` as binary (Male=0, Female=1)
    - Drop `Patient_ID` and keep other numeric features

    Returns:
      X: DataFrame of features
      y: numpy array of labels if `Disease` column present, otherwise None
      artifacts: dict with encoders and feature column list
    """
    df = df.copy()

    # Ensure Symptoms column exists
    if "Symptoms" not in df.columns:
        df["Symptoms"] = [[] for _ in range(len(df))]

    # Convert to list of strings
    df["Symptoms"] = df["Symptoms"].fillna("").apply(lambda x: [s.strip() for s in x.split(",")] if x else [])

    # Multi-hot encode symptoms
    if fit or mlb is None:
        mlb = MultiLabelBinarizer()
        symptom_features = mlb.fit_transform(df["Symptoms"])
    else:
        symptom_features = mlb.transform(df["Symptoms"])

    # Column names: replace spaces with underscores for safety
    symptom_cols = [c.replace(" ", "_") for c in mlb.classes_]
    symptom_df = pd.DataFrame(symptom_features, columns=symptom_cols, index=df.index)

    # Engineered feature: Symptom_Count
    df["Symptom_Count"] = df["Symptoms"].apply(len)

    # Encode gender (simple binary). If values are other than Male/Female, attempt mapping.
    if "Gender" in df.columns:
        df["Gender_bin"] = df["Gender"].map({"Male": 0, "Female": 1})
        # If mapping produced NaNs, try label encoding fallback
        if df["Gender_bin"].isna().any():
            le = LabelEncoder()
            df["Gender_bin"] = le.fit_transform(df["Gender"].astype(str))
    else:
        df["Gender_bin"] = 0

    # Build final feature frame
    drop_cols = [c for c in ["Patient_ID", "Symptoms", "Gender"] if c in df.columns]
    base_df = df.drop(columns=drop_cols, errors="ignore")

    X = pd.concat([base_df.reset_index(drop=True), symptom_df.reset_index(drop=True)], axis=1)

    # Prepare target if present
    y = None
    label_encoder = None
    if "Disease" in df.columns:
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(df["Disease"].astype(str))

    artifacts = {
        "mlb": mlb,
        "label_encoder": label_encoder,
        "feature_columns": X.columns.tolist(),
        "symptom_columns": symptom_cols,
    }

    return X, y, artifacts


def save_artifacts(artifacts: Dict, out_dir: str) -> None:
    """Save encoders and feature column list to `out_dir` using joblib."""
    os.makedirs(out_dir, exist_ok=True)
    if artifacts.get("mlb") is not None:
        joblib.dump(artifacts["mlb"], os.path.join(out_dir, "symptom_binarizer.pkl"))
    if artifacts.get("label_encoder") is not None:
        joblib.dump(artifacts["label_encoder"], os.path.join(out_dir, "label_encoder.pkl"))
    joblib.dump(artifacts.get("feature_columns", []), os.path.join(out_dir, "feature_columns.pkl"))


def example_run(data_csv: str, out_dir: Optional[str] = None) -> None:
    """Example entrypoint: loads data, runs preprocessing, saves processed CSV + artifacts."""
    df = load_data(data_csv)
    X, y, artifacts = preprocess_features(df, fit=True)

    # Default output folder: Feature_Engineering/saved_models (next to this file)
    if out_dir is None:
        base_dir = os.path.dirname(__file__)
        out_dir = os.path.join(base_dir, "saved_models")

    # Save processed features
    os.makedirs(out_dir, exist_ok=True)
    X.to_csv(os.path.join(out_dir, "processed_features.csv"), index=False)

    # Save artifacts
    save_artifacts(artifacts, out_dir)

    print("Processed features saved to:", os.path.join(out_dir, "processed_features.csv"))
    print("Encoders and feature list saved to:", out_dir)


if __name__ == "__main__":
    # Default path from project structure
    default_path = os.path.join("..", "Data", "Healthcare.csv")
    if not os.path.exists(default_path):
        # Try absolute Windows path (workspace layout)
        default_path = r"a:\My project\SymptomClassifier-XAI\Data\Healthcare.csv"
    # Use saved_models folder inside the Feature_Engineering package
    feature_saved_models = os.path.join(os.path.dirname(__file__), "saved_models")
    example_run(default_path, out_dir=feature_saved_models)
