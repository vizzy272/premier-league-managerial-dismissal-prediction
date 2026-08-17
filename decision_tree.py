#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 04:58:11 2026

@author: vizzy
"""

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree

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

tree_model = DecisionTreeClassifier(
    random_state=42,
    max_depth=3
)

tree_model.fit(X_train, y_train)

predictions = tree_model.predict(X_test)

probabilities = tree_model.predict_proba(X_test)[:,1]

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
    "/Users/vizzy/Downloads/Premier league/Data/cleaned/Table_5_5_DecisionTree_ConfusionMatrix.csv"
)

print("Confusion Matrix saved.")


RocCurveDisplay.from_predictions(
    y_test,
    probabilities
)

plt.title("Decision Tree ROC Curve")

plt.tight_layout()

plt.savefig(
    "/Users/vizzy/Downloads/Premier league/Data/cleaned/Figure_5_3_DecisionTree_ROC.png",
    dpi=300
)

plt.show()



plt.figure(figsize=(12,8))

plot_tree(
    tree_model,
    feature_names=X.columns,
    class_names=["Not Dismissed","Dismissed"],
    filled=True,
    rounded=True,
    fontsize=10
)

plt.tight_layout()

plt.savefig(
    "/Users/vizzy/Downloads/Premier league/Data/cleaned/Figure_5_4_DecisionTree.png",
    dpi=300
)

plt.show()


# ==========================================
# Feature Importance
# ==========================================

importance = pd.DataFrame({
    "Variable": X.columns,
    "Importance": tree_model.feature_importances_
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
    "/Users/vizzy/Downloads/Premier league/Data/cleaned/Table_5_6_Feature_Importance.csv",
    index=False
)

plt.figure(figsize=(7,4))

plt.bar(
    importance["Variable"],
    importance["Importance"]
)

plt.title("Decision Tree Feature Importance")
plt.ylabel("Importance")

plt.tight_layout()

plt.savefig(
    "/Users/vizzy/Downloads/Premier league/Data/cleaned/Figure_5_5_Feature_Importance.png",
    dpi=300
)

plt.show()

print("Feature importance saved.")

































