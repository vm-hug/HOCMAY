# ==========================================================
# LAB 5 - Ensemble Learning
# ==========================================================
# 1. Base Models
# 2. Voting Classifier (Hard / Soft)
# 3. Bagging Classifier + OOB Score
# 4. Random Forest + Feature Importance
# 5. XGBoost
# 6. So sánh Accuracy
# ==========================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import (
    VotingClassifier,
    BaggingClassifier,
    RandomForestClassifier,
    GradientBoostingClassifier
)

from sklearn.metrics import accuracy_score

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except:
    XGBOOST_AVAILABLE = False

# ==========================================================
# STEP 1: LOAD DATASET
# ==========================================================

print("=" * 60)
print("STEP 1 - LOAD DATASET")
print("=" * 60)

data = load_breast_cancer()

X = data.data
y = data.target

feature_names = data.feature_names

print("Dataset Shape:", X.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

results = []

# ==========================================================
# STEP 2: BASE MODELS
# ==========================================================

print("\nSTEP 2 - BASE MODELS")

models = {

    "Logistic Regression":
        LogisticRegression(max_iter=10000),

    "SVM":
        SVC(probability=True),

    "Decision Tree":
        DecisionTreeClassifier(random_state=42)
}

for name, model in models.items():

    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    acc = accuracy_score(y_test, pred)

    results.append([name, acc])

    print(f"{name}: {acc:.4f}")

# ==========================================================
# STEP 3: VOTING CLASSIFIER
# ==========================================================

print("\nSTEP 3 - VOTING CLASSIFIER")

hard_voting = VotingClassifier(

    estimators=[
        ('lr', LogisticRegression(max_iter=10000)),
        ('svm', SVC(probability=True)),
        ('tree', DecisionTreeClassifier())
    ],

    voting='hard'
)

hard_voting.fit(X_train, y_train)

pred = hard_voting.predict(X_test)

acc = accuracy_score(y_test, pred)

results.append(["Hard Voting", acc])

print("Hard Voting:", round(acc, 4))

# ----------------------------------------------------------

soft_voting = VotingClassifier(

    estimators=[
        ('lr', LogisticRegression(max_iter=10000)),
        ('svm', SVC(probability=True)),
        ('tree', DecisionTreeClassifier())
    ],

    voting='soft'
)

soft_voting.fit(X_train, y_train)

pred = soft_voting.predict(X_test)

acc = accuracy_score(y_test, pred)

results.append(["Soft Voting", acc])

print("Soft Voting:", round(acc, 4))

# ==========================================================
# STEP 4: BAGGING
# ==========================================================

print("\nSTEP 4 - BAGGING")

bagging = BaggingClassifier(

    estimator=DecisionTreeClassifier(),

    n_estimators=200,

    oob_score=True,

    random_state=42
)

bagging.fit(X_train, y_train)

pred = bagging.predict(X_test)

acc = accuracy_score(y_test, pred)

results.append(["Bagging", acc])

print("Bagging Accuracy:", round(acc, 4))
print("OOB Score:", round(bagging.oob_score_, 4))

# ==========================================================
# STEP 5: RANDOM FOREST
# ==========================================================

print("\nSTEP 5 - RANDOM FOREST")

rf = RandomForestClassifier(

    n_estimators=200,

    random_state=42
)

rf.fit(X_train, y_train)

pred = rf.predict(X_test)

acc = accuracy_score(y_test, pred)

results.append(["Random Forest", acc])

print("Random Forest:", round(acc, 4))

# ==========================================================
# FEATURE IMPORTANCE
# ==========================================================

importance = rf.feature_importances_

importance_df = pd.DataFrame({

    "Feature": feature_names,
    "Importance": importance

})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

print("\nTop 10 Important Features:\n")
print(importance_df.head(10))

# ----------------------------------------------------------
# FEATURE IMPORTANCE CHART
# ----------------------------------------------------------

top10 = importance_df.head(10)

plt.figure(figsize=(10, 6))

plt.barh(
    top10["Feature"],
    top10["Importance"]
)

plt.title("Top 10 Feature Importance (Random Forest)")

plt.gca().invert_yaxis()

plt.show()

# ==========================================================
# STEP 6: BOOSTING
# ==========================================================

print("\nSTEP 6 - BOOSTING")

gb = GradientBoostingClassifier()

gb.fit(X_train, y_train)

pred = gb.predict(X_test)

acc = accuracy_score(y_test, pred)

results.append(["Gradient Boosting", acc])

print("Gradient Boosting:", round(acc, 4))

# ==========================================================
# XGBOOST
# ==========================================================

if XGBOOST_AVAILABLE:

    print("\nSTEP 6B - XGBOOST")

    xgb = XGBClassifier(

        n_estimators=200,

        max_depth=4,

        learning_rate=0.1,

        eval_metric='logloss',

        random_state=42
    )

    xgb.fit(X_train, y_train)

    pred = xgb.predict(X_test)

    acc = accuracy_score(y_test, pred)

    results.append(["XGBoost", acc])

    print("XGBoost:", round(acc, 4))

# ==========================================================
# FINAL COMPARISON
# ==========================================================

print("\n")
print("=" * 60)
print("FINAL COMPARISON")
print("=" * 60)

result_df = pd.DataFrame(

    results,

    columns=[
        "Model",
        "Accuracy"
    ]
)

result_df = result_df.sort_values(
    by="Accuracy",
    ascending=False
)

print(result_df)

# ==========================================================
# ACCURACY CHART
# ==========================================================

plt.figure(figsize=(10, 6))

plt.bar(
    result_df["Model"],
    result_df["Accuracy"]
)

plt.xticks(rotation=30)

plt.ylabel("Accuracy")

plt.title(
    "Base Models vs Voting vs Bagging vs RF vs Boosting"
)

plt.show()

print("\nLAB 5 COMPLETED SUCCESSFULLY!")