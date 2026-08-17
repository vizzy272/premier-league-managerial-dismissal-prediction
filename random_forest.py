#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 05:05:32 2026

@author: vizzy
"""

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    RocCurveDisplay
)

final_dataset = pd.read_csv(
    "/Users/vizzy/Downloads/Premier league/Data/cleaned/Final_Dataset.csv"
)

print(final_dataset.head())

X = final_dataset[
    [
        "Wins",
        "Draws",
        "Goals_Against"
    ]
]

y = final_dataset["Dismissed"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training observations:", len(X_train))
print("Testing observations:", len(X_test))


#  Train the Tuned Random Forest

rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=4,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42
)

rf_model.fit(X_train, y_train)

predictions = rf_model.predict(X_test)

probabilities = rf_model.predict_proba(X_test)[:,1]


accuracy = accuracy_score(y_test, predictions)

print("\n==============================")
print("MODEL ACCURACY")
print("==============================")
print(round(accuracy,2))

cm = confusion_matrix(y_test, predictions)

print("\n==============================")
print("CONFUSION MATRIX")
print("==============================")
print(cm)

print("\n==============================")
print("CLASSIFICATION REPORT")
print("==============================")
print(classification_report(y_test, predictions))

auc = roc_auc_score(y_test, probabilities)

print("\n==============================")
print("ROC AUC")
print("==============================")
print(round(auc,2))


cm_df = pd.DataFrame(
    cm,
    index=["Actual 0","Actual 1"],
    columns=["Predicted 0","Predicted 1"]
)

cm_df.to_csv(
    "/Users/vizzy/Downloads/Premier league/Data/cleaned/Table_5_7_RandomForest_ConfusionMatrix.csv"
)

print("Confusion Matrix saved.")


RocCurveDisplay.from_predictions(
    y_test,
    probabilities
)

plt.title("Random Forest ROC Curve")

plt.tight_layout()

plt.savefig(
    "/Users/vizzy/Downloads/Premier league/Data/cleaned/Figure_5_6_RandomForest_ROC.png",
    dpi=300
)

plt.show()


importance = pd.DataFrame({
    "Variable": X.columns,
    "Importance": rf_model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print("\n==============================")
print("FEATURE IMPORTANCE")
print("==============================")
print(importance)

importance.to_csv(
    "/Users/vizzy/Downloads/Premier league/Data/cleaned/Table_5_8_RandomForest_FeatureImportance.csv",
    index=False
)

plt.figure(figsize=(7,4))

plt.bar(
    importance["Variable"],
    importance["Importance"]
)

plt.title("Random Forest Feature Importance")
plt.ylabel("Importance")

plt.tight_layout()

plt.savefig(
    "/Users/vizzy/Downloads/Premier league/Data/cleaned/Figure_5_7_RandomForest_FeatureImportance.png",
    dpi=300
)

plt.show()

print("Feature importance saved.")





































