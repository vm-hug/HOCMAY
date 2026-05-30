# ==========================================================
# LAB 4D - Model Evaluation (Metrics & Imbalanced Data)
# ==========================================================
# Nội dung:
# 1. Imbalanced Dataset
# 2. Logistic Regression
# 3. Accuracy Problem
# 4. Confusion Matrix
# 5. Precision, Recall, F1
# 6. ROC Curve & AUC
# 7. Threshold Analysis
# ==========================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    roc_auc_score
)

# ==========================================================
# STEP 1: CREATE IMBALANCED DATASET
# ==========================================================

print("=" * 60)
print("STEP 1 - CREATE IMBALANCED DATASET")
print("=" * 60)

X, y = make_classification(
    n_samples=5000,
    n_features=10,
    n_informative=6,
    n_redundant=2,
    n_classes=2,
    weights=[0.90, 0.10],  # 90% Negative, 10% Positive
    random_state=42
)

# Kiểm tra tỷ lệ class

unique, counts = np.unique(y, return_counts=True)

print("\nClass Distribution:")

for cls, count in zip(unique, counts):
    print(f"Class {cls}: {count}")

# Visualize class distribution

plt.figure(figsize=(6,4))

plt.bar(
    ["Negative (0)", "Positive (1)"],
    counts
)

plt.title("Class Imbalance Distribution")

plt.show()

# ==========================================================
# STEP 2: TRAIN BASELINE MODEL
# ==========================================================

print("\n")
print("=" * 60)
print("STEP 2 - TRAIN LOGISTIC REGRESSION")
print("=" * 60)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

model = LogisticRegression(
    max_iter=1000
)

model.fit(X_train, y_train)

# Prediction

y_pred = model.predict(X_test)

# Probability

y_prob = model.predict_proba(X_test)[:, 1]

# ==========================================================
# STEP 3: ACCURACY
# ==========================================================

print("\n")
print("=" * 60)
print("STEP 3 - ACCURACY")
print("=" * 60)

accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy = {accuracy:.4f}")

print(
    "\nLưu ý: Accuracy có thể cao "
    "dù mô hình bỏ sót nhiều Fraud Cases."
)

# ==========================================================
# STEP 4: CONFUSION MATRIX
# ==========================================================

print("\n")
print("=" * 60)
print("STEP 4 - CONFUSION MATRIX")
print("=" * 60)

cm = confusion_matrix(
    y_test,
    y_pred
)

print(cm)

plt.figure(figsize=(6,5))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues'
)

plt.title("Confusion Matrix")

plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.show()

# ==========================================================
# STEP 5: PRECISION RECALL F1
# ==========================================================

print("\n")
print("=" * 60)
print("STEP 5 - PRECISION / RECALL / F1")
print("=" * 60)

precision = precision_score(
    y_test,
    y_pred
)

recall = recall_score(
    y_test,
    y_pred
)

f1 = f1_score(
    y_test,
    y_pred
)

print(f"Precision = {precision:.4f}")
print(f"Recall    = {recall:.4f}")
print(f"F1 Score  = {f1:.4f}")

# ==========================================================
# STEP 6: ROC CURVE + AUC
# ==========================================================

print("\n")
print("=" * 60)
print("STEP 6 - ROC CURVE & AUC")
print("=" * 60)

fpr, tpr, thresholds = roc_curve(
    y_test,
    y_prob
)

auc_score = roc_auc_score(
    y_test,
    y_prob
)

print(f"AUC Score = {auc_score:.4f}")

plt.figure(figsize=(7,6))

plt.plot(
    fpr,
    tpr,
    label=f"AUC = {auc_score:.4f}"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle='--'
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title("ROC Curve")

plt.legend()

plt.show()

# ==========================================================
# STEP 7: THRESHOLD ANALYSIS
# ==========================================================

print("\n")
print("=" * 60)
print("STEP 7 - THRESHOLD ANALYSIS")
print("=" * 60)

thresholds_to_test = [
    0.3,
    0.5,
    0.7
]

results = []

for threshold in thresholds_to_test:

    y_pred_threshold = (
        y_prob >= threshold
    ).astype(int)

    acc = accuracy_score(
        y_test,
        y_pred_threshold
    )

    prec = precision_score(
        y_test,
        y_pred_threshold
    )

    rec = recall_score(
        y_test,
        y_pred_threshold
    )

    f1_val = f1_score(
        y_test,
        y_pred_threshold
    )

    results.append([
        threshold,
        acc,
        prec,
        rec,
        f1_val
    ])

results_df = pd.DataFrame(
    results,
    columns=[
        "Threshold",
        "Accuracy",
        "Precision",
        "Recall",
        "F1"
    ]
)

print("\nThreshold Comparison:\n")
print(results_df)

# ==========================================================
# SUMMARY TABLE
# ==========================================================

print("\n")
print("=" * 60)
print("FINAL METRICS")
print("=" * 60)

summary = pd.DataFrame({
    "Metric": [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
        "AUC"
    ],
    "Value": [
        accuracy,
        precision,
        recall,
        f1,
        auc_score
    ]
})

print(summary)

# Visualize Metrics

plt.figure(figsize=(7,4))

plt.bar(
    summary["Metric"],
    summary["Value"]
)

plt.ylim(0, 1)

plt.title("Evaluation Metrics")

plt.show()

print("\nLAB 4D COMPLETED SUCCESSFULLY!")