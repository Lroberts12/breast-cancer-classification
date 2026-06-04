"""
Breast Cancer Classification
=============================
Dataset: Breast Cancer Wisconsin (Diagnostic) — built into scikit-learn
Author: Lindsay Roberts

Classify tumors as Benign or Malignant from digitized FNA (fine needle aspirate)
image features. Emphasis on clinical metrics: sensitivity, specificity, ROC-AUC.

Models: Logistic Regression, Random Forest, Support Vector Machine (SVM)
"""

# ── Imports ──────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import (
    train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_curve, auc, roc_auc_score,
    accuracy_score, precision_score, recall_score, f1_score
)
from sklearn.inspection import permutation_importance

import os
os.makedirs("figures", exist_ok=True)

# ── 1. Load Data ──────────────────────────────────────────────────────────────
print("=" * 60)
print("BREAST CANCER CLASSIFICATION")
print("=" * 60)

print("\n[1] Loading Breast Cancer Wisconsin Dataset...")
bc = load_breast_cancer()
df = pd.DataFrame(bc.data, columns=bc.feature_names)
df['target'] = bc.target          # 0 = malignant, 1 = benign
df['diagnosis'] = df['target'].map({0: 'Malignant', 1: 'Benign'})

print(f"   Shape: {df.shape[0]} samples × {len(bc.feature_names)} features")
print(f"\n   Class distribution:")
print(f"   Malignant (0): {(df.target == 0).sum()}")
print(f"   Benign    (1): {(df.target == 1).sum()}")
print(f"\n   Feature groups:")
print(f"   Mean features:    {[str(f) for f in bc.feature_names if 'mean' in f][:3]} ...")
print(f"   SE features:      {[str(f) for f in bc.feature_names if 'error' in f][:3]} ...")
print(f"   Worst features:   {[str(f) for f in bc.feature_names if 'worst' in f][:3]} ...")

# ── 2. EDA ────────────────────────────────────────────────────────────────────
print("\n[2] Exploratory Data Analysis...")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Class distribution
counts = df['diagnosis'].value_counts()
colors_bar = ['tomato', 'steelblue']
bars = axes[0].bar(counts.index, counts.values, color=colors_bar, edgecolor='black', width=0.4)
for bar in bars:
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3,
                 str(int(bar.get_height())), ha='center', va='bottom', fontsize=12)
axes[0].set_title('Class Distribution', fontsize=13, fontweight='bold')
axes[0].set_ylabel('Count')
axes[0].grid(axis='y', alpha=0.3)

# Correlation heatmap (mean features only — top 10 features)
mean_features = [f for f in bc.feature_names if 'mean' in f]
corr = df[mean_features + ['target']].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.1f', cmap='coolwarm',
            ax=axes[1], linewidths=0.4, cbar_kws={'shrink': 0.8},
            annot_kws={'size': 7})
axes[1].set_title('Correlation Matrix (Mean Features)', fontsize=13, fontweight='bold')
axes[1].tick_params(axis='x', rotation=45, labelsize=8)
axes[1].tick_params(axis='y', labelsize=8)

plt.tight_layout()
plt.savefig('figures/01_eda.png', dpi=150, bbox_inches='tight')
plt.close()
print("   Saved: figures/01_eda.png")

# Violin plots for most discriminating features
top_features = ['mean radius', 'mean texture', 'mean perimeter',
                'mean area', 'mean concavity', 'mean concave points']

fig, axes = plt.subplots(2, 3, figsize=(14, 8))
axes = axes.flatten()
palette = {'Malignant': 'tomato', 'Benign': 'steelblue'}

for i, feat in enumerate(top_features):
    sns.violinplot(data=df, x='diagnosis', y=feat, ax=axes[i],
                   palette=palette, inner='box', cut=0)
    axes[i].set_title(feat.title(), fontsize=11)
    axes[i].set_xlabel('')

plt.suptitle('Feature Distributions: Malignant vs Benign', fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('figures/02_violin_plots.png', dpi=150, bbox_inches='tight')
plt.close()
print("   Saved: figures/02_violin_plots.png")

# ── 3. Prepare Features ───────────────────────────────────────────────────────
print("\n[3] Preparing Features...")

X = df[list(bc.feature_names)]
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

print(f"   Train: {X_train.shape[0]} samples | Test: {X_test.shape[0]} samples")
print(f"   All features scaled for LR and SVM")

# ── 4. Train Models ───────────────────────────────────────────────────────────
print("\n[4] Training Models...")

models_config = {
    'Logistic Regression': (
        LogisticRegression(C=1.0, max_iter=10000, random_state=42),
        X_train_sc, X_test_sc
    ),
    'Random Forest': (
        RandomForestClassifier(n_estimators=300, max_depth=None,
                               min_samples_split=2, random_state=42),
        X_train, X_test
    ),
    'SVM': (
        SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42),
        X_train_sc, X_test_sc
    ),
}

results = {}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, (model, Xtr, Xte) in models_config.items():
    cv_auc = cross_val_score(model, Xtr, y_train, cv=cv, scoring='roc_auc')
    model.fit(Xtr, y_train)
    y_pred  = model.predict(Xte)
    y_proba = model.predict_proba(Xte)[:, 1]

    # Clinical metrics (malignant = 0 is the "positive" class for clinical sensitivity)
    # Sensitivity = recall for malignant class (we don't want to miss cancer)
    sensitivity = recall_score(y_test, y_pred, pos_label=0)
    # Specificity = recall for benign class
    specificity = recall_score(y_test, y_pred, pos_label=1)

    results[name] = {
        'model':       model,
        'y_pred':      y_pred,
        'y_proba':     y_proba,
        'accuracy':    accuracy_score(y_test, y_pred),
        'auc':         roc_auc_score(y_test, y_proba),
        'cv_auc':      cv_auc.mean(),
        'cv_std':      cv_auc.std(),
        'sensitivity': sensitivity,
        'specificity': specificity,
        'precision':   precision_score(y_test, y_pred, pos_label=0),
        'f1':          f1_score(y_test, y_pred, pos_label=0),
        'X_test':      Xte,
    }
    print(f"   {name:22s} | AUC: {results[name]['auc']:.3f} | "
          f"Sensitivity: {sensitivity:.3f} | Specificity: {specificity:.3f}")

# ── 5. ROC Curves ─────────────────────────────────────────────────────────────
print("\n[5] Generating ROC Curves & Clinical Metrics...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
colors = ['steelblue', 'darkorange', 'seagreen']

# ROC curves — plotted with malignant (0) as positive class
for (name, res), color in zip(results.items(), colors):
    # Invert probabilities since malignant=0 is the positive class here
    fpr, tpr, _ = roc_curve(y_test, 1 - res['y_proba'], pos_label=0)
    roc_auc = auc(fpr, tpr)
    axes[0].plot(fpr, tpr, color=color, lw=2,
                 label=f"{name} (AUC = {roc_auc:.3f})")

axes[0].plot([0, 1], [0, 1], 'k--', lw=1.5, label='Random Classifier')
axes[0].set_xlabel('False Positive Rate (1 - Specificity)', fontsize=11)
axes[0].set_ylabel('True Positive Rate (Sensitivity)', fontsize=11)
axes[0].set_title('ROC Curves – Malignant Detection', fontsize=13, fontweight='bold')
axes[0].legend(fontsize=10)
axes[0].grid(alpha=0.3)

# Clinical metrics grouped bar chart
metrics = ['accuracy', 'auc', 'sensitivity', 'specificity', 'precision']
metric_labels = ['Accuracy', 'ROC-AUC', 'Sensitivity\n(Recall Malignant)',
                 'Specificity\n(Recall Benign)', 'Precision\n(Malignant)']

x = np.arange(len(metrics))
width = 0.25
metric_colors = ['steelblue', 'darkorange', 'seagreen']

for i, (name, res) in enumerate(results.items()):
    vals = [res[m] for m in metrics]
    axes[1].bar(x + i * width, vals, width, label=name,
                color=metric_colors[i], edgecolor='black', alpha=0.85)

axes[1].set_xticks(x + width)
axes[1].set_xticklabels(metric_labels, fontsize=9)
axes[1].set_ylim(0, 1.1)
axes[1].set_title('Clinical Performance Metrics', fontsize=13, fontweight='bold')
axes[1].legend(fontsize=9)
axes[1].grid(axis='y', alpha=0.3)
axes[1].axhline(y=0.9, color='red', linestyle='--', linewidth=1, alpha=0.5, label='0.9 threshold')

plt.tight_layout()
plt.savefig('figures/03_roc_clinical_metrics.png', dpi=150, bbox_inches='tight')
plt.close()
print("   Saved: figures/03_roc_clinical_metrics.png")

# ── 6. Feature Importance ─────────────────────────────────────────────────────
print("\n[6] Feature Importance...")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Random Forest importances
rf = results['Random Forest']['model']
rf_imp = pd.Series(rf.feature_importances_, index=bc.feature_names)
rf_imp_top = rf_imp.nlargest(15).sort_values(ascending=True)
rf_imp_top.plot(kind='barh', ax=axes[0], color='darkorange', edgecolor='black')
axes[0].set_title('Random Forest — Top 15 Feature Importances', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Importance (Mean Decrease in Impurity)')
axes[0].grid(axis='x', alpha=0.3)

# Logistic Regression coefficients
lr = results['Logistic Regression']['model']
lr_coef = pd.Series(lr.coef_[0], index=bc.feature_names)
lr_coef_top = lr_coef.abs().nlargest(15)
lr_coef_sorted = lr_coef[lr_coef_top.index].sort_values()
bar_colors = ['tomato' if v < 0 else 'steelblue' for v in lr_coef_sorted]
axes[1].barh(lr_coef_sorted.index, lr_coef_sorted.values, color=bar_colors, edgecolor='black')
axes[1].axvline(0, color='black', linewidth=0.8)
axes[1].set_title('Logistic Regression — Top 15 Coefficients\n(blue = benign, red = malignant)', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Coefficient Value')
axes[1].grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('figures/04_feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()
print("   Saved: figures/04_feature_importance.png")

# ── 7. Confusion Matrices ─────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
labels = ['Malignant', 'Benign']

for ax, (name, res) in zip(axes, results.items()):
    cm = confusion_matrix(y_test, res['y_pred'])
    # Annotate with counts and rates
    cm_norm = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis]
    annot = np.array([[f"{cm[i,j]}\n({cm_norm[i,j]:.0%})"
                       for j in range(2)] for i in range(2)])
    sns.heatmap(cm, annot=annot, fmt='', cmap='Blues', ax=ax,
                xticklabels=labels, yticklabels=labels,
                linewidths=0.5, linecolor='gray')
    ax.set_title(f'{name}\nSens: {res["sensitivity"]:.3f} | Spec: {res["specificity"]:.3f}',
                 fontsize=11, fontweight='bold')
    ax.set_ylabel('Actual')
    ax.set_xlabel('Predicted')

plt.suptitle('Confusion Matrices — count (row %)', fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('figures/05_confusion_matrices.png', dpi=150, bbox_inches='tight')
plt.close()
print("   Saved: figures/05_confusion_matrices.png")

# ── 8. Final Summary ──────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("FINAL RESULTS SUMMARY")
print("=" * 60)
print(f"\n{'Model':<25} {'AUC':>6} {'Sens':>6} {'Spec':>6} {'Acc':>6} {'F1':>6}")
print("-" * 55)
for name, res in results.items():
    print(f"{name:<25} {res['auc']:>6.3f} {res['sensitivity']:>6.3f} "
          f"{res['specificity']:>6.3f} {res['accuracy']:>6.3f} {res['f1']:>6.3f}")

best = max(results, key=lambda k: results[k]['auc'])
print(f"\n✓ Best model by ROC-AUC: {best} ({results[best]['auc']:.4f})")

print("\nDetailed Classification Reports:")
for name, res in results.items():
    print(f"\n{name}")
    print("-" * 40)
    print(classification_report(y_test, res['y_pred'],
                                 target_names=['Malignant', 'Benign']))

print("\nNote on clinical priority:")
print("  Sensitivity (recall for malignant) is critical in cancer screening.")
print("  A missed malignant diagnosis is more harmful than a false positive.")
print("\nAll figures saved to /figures/")
print("=" * 60)
