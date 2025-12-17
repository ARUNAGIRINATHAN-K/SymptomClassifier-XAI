"""XAI explainer utilities using SHAP with a LIME fallback.

Functions:
- explain_instance_shap(model, X, instance_idx, background=None, nsamples=100)
- explain_instance_lime(model, X, instance_idx, feature_names=None)
- explain_and_save(model, X, instance_idx, out_path, feature_names=None)

Notes:
- Install: `pip install shap lime pandas numpy` (SHAP preferred)
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import numbers


def _is_tree_model(model: Any) -> bool:
    name = type(model).__name__.lower()
    return any(k in name for k in ("xgboost", "lightgbm", "catboost", "randomforest", "decisiontree"))


def explain_instance_shap(
    model: Any,
    X: pd.DataFrame,
    instance_idx: int = 0,
    background: Optional[pd.DataFrame] = None,
    nsamples: int = 100,
) -> Dict:
    """Return SHAP explanation dict for a single instance.

    Tries TreeExplainer for tree models, otherwise uses KernelExplainer with a small background.
    """
    try:
        import shap
    except Exception as e:
        raise RuntimeError("SHAP is not installed: pip install shap") from e

    X = pd.DataFrame(X)
    instance = X.iloc[[instance_idx]]

    if background is None:
        # use a small sample as background to speed up KernelExplainer
        background = X.sample(n=min(50, len(X)), random_state=0)

    if _is_tree_model(model):
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(instance)
    else:
        # KernelExplainer expects a predict_proba or predict function depending on model
        try:
            predict_fn = model.predict_proba
        except AttributeError:
            predict_fn = model.predict
        explainer = shap.KernelExplainer(predict_fn, background.values)
        shap_values = explainer.shap_values(instance.values, nsamples=nsamples)

    # Handle multi-class: shap_values may be list per class
    if isinstance(shap_values, list):
        # choose top predicted class
        try:
            probs = model.predict_proba(instance) if hasattr(model, "predict_proba") else None
            top_class = int(np.argmax(probs)) if probs is not None else 0
        except Exception:
            top_class = 0
        vals = np.array(shap_values)[top_class].reshape(-1)
    else:
        vals = np.array(shap_values).reshape(-1)

    contributions = list(zip(X.columns.tolist(), vals.tolist()))
    contributions_sorted = sorted(contributions, key=lambda x: abs(x[1]), reverse=True)

    return {
        "method": "shap",
        "instance_index": int(instance_idx),
        "prediction": (model.predict(instance)[0] if hasattr(model, "predict") else None),
        "contributions": [
            {"feature": f, "contribution": float(v)} for f, v in contributions_sorted
        ],
    }


def explain_instance_lime(model: Any, X: pd.DataFrame, instance_idx: int = 0, feature_names: Optional[List[str]] = None) -> Dict:
    """Return LIME explanation for a single instance (fallback)."""
    try:
        from lime.lime_tabular import LimeTabularExplainer
    except Exception as e:
        raise RuntimeError("LIME is not installed: pip install lime") from e

    X = pd.DataFrame(X)
    instance = X.iloc[instance_idx].values
    feature_names = feature_names or X.columns.tolist()

    explainer = LimeTabularExplainer(
        training_data=X.values,
        feature_names=feature_names,
        discretize_continuous=True,
    )

    if hasattr(model, "predict_proba"):
        exp = explainer.explain_instance(instance, model.predict_proba, num_features=len(feature_names))
        # pick top class explanation
        class_expl = exp.as_map()[exp.available_labels()[0]]
        contributions = [(feature_names[i], w) for i, w in class_expl]
    else:
        exp = explainer.explain_instance(instance, model.predict, num_features=len(feature_names))
        contributions = exp.as_list()

    contributions_sorted = sorted(contributions, key=lambda x: abs(x[1]), reverse=True)
    return {
        "method": "lime",
        "instance_index": int(instance_idx),
        "prediction": (model.predict(X.iloc[[instance_idx]])[0] if hasattr(model, "predict") else None),
        "contributions": [{"feature": f, "contribution": float(v)} for f, v in contributions_sorted],
    }


def explain_and_save(
    model: Any,
    X: pd.DataFrame,
    instance_idx: int,
    out_path: str,
    feature_names: Optional[List[str]] = None,
    prefer_shap: bool = True,
) -> str:
    """Compute explanation and save JSON to out_path. Returns path."""
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    explanation = None
    if prefer_shap:
        try:
            explanation = explain_instance_shap(model, X, instance_idx)
        except Exception:
            # fallback to LIME
            explanation = explain_instance_lime(model, X, instance_idx, feature_names=feature_names)
    else:
        try:
            explanation = explain_instance_lime(model, X, instance_idx, feature_names=feature_names)
        except Exception:
            explanation = explain_instance_shap(model, X, instance_idx)


    def _make_json_serializable(obj: Any) -> Any:
        """Recursively convert numpy/pandas types to native Python types for JSON."""
        if obj is None:
            return None
        if isinstance(obj, (str, bool)):
            return obj
        if isinstance(obj, numbers.Number):
            # numpy numbers subclass Python numbers.Number but may not be JSON serializable
            try:
                return obj.item() if hasattr(obj, "item") else obj
            except Exception:
                return float(obj)
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, pd.Series):
            return obj.tolist()
        if isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient="records")
        if isinstance(obj, dict):
            return {str(k): _make_json_serializable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_make_json_serializable(v) for v in obj]
        # fallback to string
        try:
            return str(obj)
        except Exception:
            return None

    sanitized = _make_json_serializable(explanation)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(sanitized, f, indent=2, ensure_ascii=False)

    return out_path
