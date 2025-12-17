"""Triage & Risk Model

This module provides a simple rule-based triage classifier that maps
predicted disease names to an urgency class:
- Self-Care (Low Risk)
- Consult GP (Medium Risk)
- Seek Emergency Care (High Risk)

The mapping is configurable via a JSON file or by passing a custom dict.
"""
from __future__ import annotations

import json
import os
from typing import Dict, Iterable, List, Optional

import numpy as np


# Default mapping (example). Update this mapping to match your disease labels.
# Keys are disease names (strings), values are output classes.
DEFAULT_SEVERITY_MAPPING: Dict[str, str] = {
    "common cold": "Self-Care (Low Risk)",
    "allergic rhinitis": "Self-Care (Low Risk)",
    "influenza": "Consult GP (Medium Risk)",
    "flu": "Consult GP (Medium Risk)",
    "bronchitis": "Consult GP (Medium Risk)",
    "pneumonia": "Seek Emergency Care (High Risk)",
    "myocardial infarction": "Seek Emergency Care (High Risk)",
    "stroke": "Seek Emergency Care (High Risk)",
}


def load_mapping(path: str) -> Dict[str, str]:
    """Load a custom severity mapping from a JSON file.

    JSON format: {"disease name": "Risk Label", ...}
    """
    with open(path, "r", encoding="utf-8") as f:
        mapping = json.load(f)
    return {k.strip(): v.strip() for k, v in mapping.items()}


def save_mapping(mapping: Dict[str, str], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2, ensure_ascii=False)


def get_risk_label_from_disease(disease: str, mapping: Optional[Dict[str, str]] = None) -> str:
    """Return triage label for a single disease name.

    - Performs case-insensitive match against mapping keys.
    - If not found, returns a conservative default: `Consult GP (Medium Risk)`.
    """
    if not disease:
        return "Consult GP (Medium Risk)"
    mapping = mapping or DEFAULT_SEVERITY_MAPPING
    key = disease.strip()
    # direct
    if key in mapping:
        return mapping[key]
    # case-insensitive
    lower_map = {k.lower(): v for k, v in mapping.items()}
    if key.lower() in lower_map:
        return lower_map[key.lower()]
    # fallback
    return "Consult GP (Medium Risk)"


def predict_risk_from_prediction(predicted_disease: str, mapping: Optional[Dict[str, str]] = None) -> Dict[str, object]:
    """Return a dict with disease and triage label for a single predicted disease name."""
    label = get_risk_label_from_disease(predicted_disease, mapping)
    return {"disease": predicted_disease, "triage": label}


def predict_risk_from_proba(probas: Iterable[float], class_names: List[str], mapping: Optional[Dict[str, str]] = None) -> Dict[str, object]:
    """Given predicted probabilities and corresponding class names, return the top prediction and triage.

    Args:
      probas: iterable of probabilities (same order as class_names)
      class_names: list of class labels (disease names)
      mapping: optional custom mapping

    Returns:
      dict with keys: `disease`, `confidence`, `triage`
    """
    probs = np.asarray(list(probas), dtype=float)
    if probs.size == 0:
        raise ValueError("`probas` must be non-empty")
    if len(probs) != len(class_names):
        raise ValueError("Length of `probas` must match `class_names`")

    idx = int(np.argmax(probs))
    disease = class_names[idx]
    confidence = float(probs[idx])
    triage = get_risk_label_from_disease(disease, mapping)
    return {"disease": disease, "confidence": confidence, "triage": triage}


def bulk_predict_risk(predicted_diseases: Iterable[str], mapping: Optional[Dict[str, str]] = None) -> List[Dict[str, str]]:
    """Map a list of predicted disease names to triage labels."""
    return [{"disease": d, "triage": get_risk_label_from_disease(d, mapping)} for d in predicted_diseases]


if __name__ == "__main__":
    # Quick demo when running the script directly
    demo_classes = ["common cold", "influenza", "pneumonia"]
    demo_probas = [0.1, 0.6, 0.3]
    out = predict_risk_from_proba(demo_probas, demo_classes)
    print("Demo prediction ->", out)
    # Save default mapping for easy editing
    mapping_path = os.path.join(os.path.dirname(__file__), "severity_mapping.json")
    if not os.path.exists(mapping_path):
        save_mapping(DEFAULT_SEVERITY_MAPPING, mapping_path)
        print("Saved default mapping to:", mapping_path)
