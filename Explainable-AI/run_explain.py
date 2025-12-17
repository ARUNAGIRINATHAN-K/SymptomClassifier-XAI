import os
import joblib
import pandas as pd
from xai_explainer import explain_and_save

# Paths
proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
feature_saved = os.path.join(proj_root, "Feature_Engineering", "saved_models")
classification_saved = os.path.join(proj_root, "Classification-Model", "saved_models")

# Prefer model in Classification-Model/saved_models
model_path = os.path.join(classification_saved, "xgboost_model.joblib")
if not os.path.exists(model_path):
    # fallback to disease_model.pkl
    model_path = os.path.join(classification_saved, "disease_model.pkl")

if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model not found at {model_path}")

processed_features = os.path.join(feature_saved, "processed_features.csv")
if not os.path.exists(processed_features):
    # try Classification-Model processed features
    processed_features = os.path.join(classification_saved, "processed_features.csv")

if not os.path.exists(processed_features):
    raise FileNotFoundError(f"Processed features not found at {processed_features}")

print("Loading model:", model_path)
model = joblib.load(model_path)
print("Loading features:", processed_features)
X = pd.read_csv(processed_features)

# Load feature column list
feature_cols_path = os.path.join(feature_saved, "feature_columns.pkl")
feature_cols = None
if os.path.exists(feature_cols_path):
    try:
        feature_cols = joblib.load(feature_cols_path)
    except Exception:
        feature_cols = None

# Derive model's expected feature names (try booster first)
target_order = None
try:
    booster = model.get_booster()
    target_order = list(booster.feature_names) if booster.feature_names is not None else None
except Exception:
    target_order = None
if target_order is None and hasattr(model, "feature_names_in_"):
    try:
        target_order = list(model.feature_names_in_)
    except Exception:
        target_order = None

# Reconcile columns
if feature_cols:
    # remove target column if present
    reconciled = [c for c in feature_cols if c.lower() not in ("disease", "target")]
    # handle common rename: Gender_bin -> Gender
    if "Gender_bin" in reconciled and target_order and "Gender" in target_order:
        reconciled = [("Gender" if c == "Gender_bin" else c) for c in reconciled]
    # If model provides an order, ensure we match it exactly
    if target_order:
        # ensure all model features exist in reconciled/X; add missing as zeros
        missing = [f for f in target_order if f not in X.columns]
        if missing:
            print("Adding missing feature columns with zeros:", missing)
            for m in missing:
                X[m] = 0
        # Now reorder according to target_order and coerce
        X = X[target_order].copy()
        for c in X.columns:
            X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0)
        reconciled = target_order
    else:
        # No target order: select intersection and coerce
        cols = [c for c in reconciled if c in X.columns]
        X = X[cols].copy()
        for c in X.columns:
            X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0)
        reconciled = cols
    # save reconciled list back
    try:
        joblib.dump(reconciled, feature_cols_path)
        print(f"Saved reconciled feature list to {feature_cols_path}")
    except Exception as e:
        print("Warning: could not save reconciled feature list:", e)
else:
    # If no feature_cols available but model provides names, attempt to align
    if target_order:
        missing = [f for f in target_order if f not in X.columns]
        if missing:
            print("Adding missing feature columns with zeros:", missing)
            for m in missing:
                X[m] = 0
        X = X[target_order].copy()
        for c in X.columns:
            X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0)
    else:
        # Best-effort: coerce all numeric-like columns
        for c in X.columns:
            X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0)

# Explain instance 0 and save
out_dir = os.path.join(os.path.dirname(__file__), "saved_explanations")
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "inst0_expl.json")

explanation_file = explain_and_save(model, X, instance_idx=0, out_path=out_path)
print("Saved explanation to:", explanation_file)
