#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 05:10:07 2026

@author: vizzy
"""

# ==========================================
# Logistic Regression
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    roc_curve,
    roc_auc_score
)

import statsmodels.api as sm

# Load dataset
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


log_model = LogisticRegression(
    random_state=42
)

log_model.fit(
    X_train,
    y_train
)


y_pred = log_model.predict(X_test)

y_prob = log_model.predict_proba(X_test)[:,1]

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\n==============================")
print("MODEL ACCURACY")
print("==============================")
print(round(accuracy,3))


print("\n==============================")
print("CONFUSION MATRIX")
print("==============================")

cm = confusion_matrix(y_test, y_pred)

print(cm)

print("\n==============================")
print("CLASSIFICATION REPORT")
print("==============================")

print(classification_report(y_test, y_pred))


fpr, tpr, thresholds = roc_curve(y_test, y_prob)

auc = roc_auc_score(y_test, y_prob)

print("\n==============================")
print("ROC AUC")
print("==============================")
print(round(auc,3))

plt.figure(figsize=(6,6))

plt.plot(
    fpr,
    tpr,
    label=f"AUC = {auc:.3f}"
)

plt.plot(
    [0,1],
    [0,1],
    linestyle="--"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")

plt.legend()

plt.tight_layout()

plt.savefig(
"/Users/vizzy/Downloads/Premier league/Data/cleaned/Figure_5_1_ROC_Curve.png",
dpi=300
)

plt.show()

cm_df = pd.DataFrame(
    cm,
    index=["Actual No","Actual Yes"],
    columns=["Predicted No","Predicted Yes"]
)

cm_df.to_csv(
    "/Users/vizzy/Downloads/Premier league/Data/cleaned/Table_5_1_Confusion_Matrix.csv"
)

print("\nConfusion Matrix saved.")


# ==========================================
# Logistic Regression (Statistical Model)
# ==========================================

X_stats = final_dataset[
    [
        "Wins",
        "Draws",
        "Goals_Against"
    ]
]

X_stats = sm.add_constant(X_stats)

y_stats = final_dataset["Dismissed"]

logit_model = sm.Logit(
    y_stats,
    X_stats
)

result = logit_model.fit()

print(result.summary())


# Odds Ratios

odds_ratios = pd.DataFrame({
    "Variable": result.params.index,
    "Coefficient": result.params.values,
    "Odds Ratio": np.exp(result.params.values),
    "P-value": result.pvalues.values
})

print("\n==============================")
print("ODDS RATIOS")
print("==============================")

print(odds_ratios.round(4))

odds_ratios.round(4).to_csv(
    "/Users/vizzy/Downloads/Premier league/Data/cleaned/Table_5_2_Odds_Ratios.csv",
    index=False
)



# ==========================================
# Individual Logistic Regression Models
# ==========================================

variables = [
    "Wins",
    "Draws",
    "Losses",
    "Goals_For",
    "Goals_Against",
    "Goal_Difference",
    "Points"
]

results = []

for variable in variables:

    X_single = sm.add_constant(final_dataset[[variable]])

    model = sm.Logit(
        final_dataset["Dismissed"],
        X_single
    )

    result = model.fit(disp=False)

    results.append({
        "Variable": variable,
        "Coefficient": result.params[variable],
        "Odds Ratio": np.exp(result.params[variable]),
        "P-value": result.pvalues[variable],
        "Pseudo R²": result.prsquared,
        "AIC": result.aic
    })

individual_results = pd.DataFrame(results)

print("\n==============================")
print("INDIVIDUAL LOGISTIC REGRESSION")
print("==============================")

print(individual_results.round(4))

individual_results.round(4).to_csv(
    "/Users/vizzy/Downloads/Premier league/Data/cleaned/Table_5_3_Individual_Logistic_Models.csv",
    index=False
)

print("\nTable 5.3 saved successfully.")




# ==========================================
# Logistic Regression Model Comparison
# ==========================================

comparison = individual_results.copy()

comparison["Significant (p<0.05)"] = comparison["P-value"].apply(
    lambda x: "Yes" if x < 0.05 else "No"
)

comparison = comparison[
    [
        "Variable",
        "Coefficient",
        "Odds Ratio",
        "P-value",
        "Pseudo R²",
        "AIC",
        "Significant (p<0.05)"
    ]
]

print("\n==============================")
print("MODEL COMPARISON")
print("==============================")

print(comparison.round(4))

comparison.round(4).to_csv(
    "/Users/vizzy/Downloads/Premier league/Data/cleaned/Table_5_4_Logistic_Model_Comparison.csv",
    index=False
)

print("\nTable 5.4 saved successfully.")

# ==========================================
# Odds Ratio Plot
# ==========================================

plot_data = individual_results.sort_values("Odds Ratio")

plt.figure(figsize=(8,5))

plt.barh(
    plot_data["Variable"],
    plot_data["Odds Ratio"]
)

plt.axvline(
    x=1,
    linestyle="--"
)

plt.xlabel("Odds Ratio")
plt.title("Odds Ratios for Individual Logistic Regression Models")

plt.tight_layout()

plt.savefig(
    "/Users/vizzy/Downloads/Premier league/Data/cleaned/Figure_5_2_Odds_Ratios.png",
    dpi=300
)

plt.show()







































