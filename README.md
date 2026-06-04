# Breast Cancer Classification

Machine learning models to classify breast tumors as **benign or malignant** from digitized fine needle aspirate (FNA) image features, with emphasis on clinical evaluation metrics.

## Overview

Using the **Breast Cancer Wisconsin (Diagnostic) Dataset**, this project asks: *can geometric and texture features of cell nuclei predict whether a tumor is malignant?* Performance is evaluated using clinically meaningful metrics — sensitivity, specificity, and ROC-AUC — since the cost of a missed malignant diagnosis far outweighs a false positive.

## Dataset

- **Source:** Breast Cancer Wisconsin (Diagnostic) — UCI / scikit-learn built-in
- **Samples:** 569 tumors (357 benign, 212 malignant)
- **Features:** 30 numeric features derived from digitized FNA images
- **Feature groups:** mean, standard error, and "worst" (largest) values of:
  - Radius, Texture, Perimeter, Area, Smoothness
  - Compactness, Concavity, Concave Points, Symmetry, Fractal Dimension

## Methods

### Models
- **Logistic Regression** — interpretable linear classifier
- **Random Forest** — ensemble method, robust to overfitting
- **SVM (RBF kernel)** — strong performance on high-dimensional biological data

### Evaluation Metrics
All three models are assessed on:

| Metric | Clinical Meaning |
|---|---|
| Sensitivity (Recall for Malignant) | Of all actual malignant tumors, how many did we catch? |
| Specificity (Recall for Benign) | Of all benign tumors, how many did we correctly spare? |
| ROC-AUC | Overall discrimination ability across all thresholds |
| Precision | Of predicted malignant, how many were truly malignant? |
| Accuracy | Overall correct classifications |

> **Why sensitivity matters most:** In cancer screening, a missed malignant tumor (false negative) is far more harmful than an unnecessary biopsy (false positive). A clinically useful model must prioritize high sensitivity.

## Results

| Model | ROC-AUC | Sensitivity | Specificity | Accuracy |
|---|---|---|---|---|
| Logistic Regression | ~0.99 | ~0.97 | ~0.98 | ~0.97 |
| Random Forest | ~0.99 | ~0.96 | ~0.98 | ~0.97 |
| SVM | ~0.99 | ~0.97 | ~0.99 | ~0.98 |

*Results may vary slightly due to train/test split randomness.*

## Key Findings

- All three models achieve **>0.99 ROC-AUC**, reflecting the high discriminability of FNA features
- **Mean concave points**, **mean perimeter**, and **worst radius** are consistently the most important features
- SVM and Logistic Regression achieve comparable performance to Random Forest on this dataset, suggesting strong linear separability in the feature space
- Features from the "worst" group (largest measured values) often outperform mean values in predictive power

## Figures

| Figure | Description |
|---|---|
| `figures/01_eda.png` | Class distribution & correlation heatmap |
| `figures/02_violin_plots.png` | Feature distributions by diagnosis |
| `figures/03_roc_clinical_metrics.png` | ROC curves & clinical metrics comparison |
| `figures/04_feature_importance.png` | RF importances & LR coefficients |
| `figures/05_confusion_matrices.png` | Confusion matrices with sensitivity/specificity |

## Setup & Usage

```bash
# Clone the repo
git clone https://github.com/Lroberts12/breast-cancer-classification.git
cd breast-cancer-classification

# Install dependencies
pip install -r requirements.txt

# Run the analysis
python breast_cancer_classification.py
```

## Requirements

```
numpy
pandas
matplotlib
seaborn
scikit-learn
```

## Skills Demonstrated

- Clinical classification with imbalanced priorities (sensitivity vs. specificity trade-off)
- Multi-model comparison with standardized evaluation
- Feature importance analysis across model types
- ROC-AUC, precision/recall, sensitivity, and specificity reporting
- Reproducible ML pipeline with stratified train/test splits
- Visualization of clinical performance metrics

## Clinical Relevance

This type of model supports clinical decision-making in oncology. A high-sensitivity classifier can serve as a screening tool to flag suspicious cases for further diagnostic workup, reducing the chance that malignant tumors are missed.


---
*Part of a healthcare ML portfolio. Dataset from UCI Machine Learning Repository.*
