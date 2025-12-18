<div align="center">

![](symptom-Classifier.png)

### AI-Powered Medical Diagnosis with Explainable Intelligence

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-Machine%20Learning-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Classifier-FF6600?style=for-the-badge)](https://xgboost.readthedocs.io/)
[![SHAP](https://img.shields.io/badge/SHAP-Explainability-00BFFF?style=for-the-badge)](https://shap.readthedocs.io/)
[![LIME](https://img.shields.io/badge/LIME-Interpretability-32CD32?style=for-the-badge)](https://github.com/marcotcr/lime)
[![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![NumPy](https://img.shields.io/badge/NumPy-Scientific%20Computing-013243?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org/)

[![License](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](LICENSE)
[![Stars](https://img.shields.io/github/stars/ARUNAGIRINATHAN-K/SymptomClassifier-XAI?style=for-the-badge)](https://github.com/ARUNAGIRINATHAN-K/SymptomClassifier-XAI/stargazers)
[![Issues](https://img.shields.io/github/issues/ARUNAGIRINATHAN-K/SymptomClassifier-XAI?style=for-the-badge)](https://github.com/ARUNAGIRINATHAN-K/SymptomClassifier-XAI/issues)
[![Last Commit](https://img.shields.io/github/last-commit/ARUNAGIRINATHAN-K/SymptomClassifier-XAI?style=for-the-badge)](https://github.com/ARUNAGIRINATHAN-K/SymptomClassifier-XAI/commits/main)

</div>

---

<div align="center">

## Overview

*SymptomClassifier-XAI is an advanced machine learning system that predicts diseases from patient symptoms while providing transparent, explainable AI insights. The project combines state-of-the-art classification algorithms with interpretability frameworks to deliver trustworthy medical predictions with clear reasoning.*

--- 

## Technical Architecture

### Data Flow

```mermaid
graph LR
    A[Raw Patient Data] --> B[Feature Engineering]
    B --> C[XGBoost Classifier]
    C --> D[Disease Prediction]
    D --> E[Triage Risk Model]
    D --> F[SHAP/LIME Explainer]
    E --> G[Urgency Classification]
    F --> H[Explanation Report]
```
</div>

---

### Model Pipeline

1. **Input Processing**: Multi-hot encoding of symptoms + demographic features
2. **Classification**: XGBoost multi-class prediction
3. **Interpretation**: SHAP values for feature importance
4. **Risk Assessment**: Rule-based triage mapping
5. **Output**: Disease prediction + risk level + explanation

---

<div align="center">

### ⭐ Star this repository if you find it helpful!

[![GitHub stars](https://img.shields.io/github/stars/ARUNAGIRINATHAN-K/SymptomClassifier-XAI?style=social)](https://github.com/ARUNAGIRINATHAN-K/SymptomClassifier-XAI/stargazers)

**Made with ❤️ for transparent healthcare AI**

</div>
