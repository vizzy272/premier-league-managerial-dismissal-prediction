#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 26 20:54:58 2026

@author: vizzy
"""

import pandas as pd

# Load the final dataset
final_dataset = pd.read_csv(
    "/Users/vizzy/Downloads/Premier league/Data/cleaned/Final_Dataset.csv"
)

print(final_dataset.head())

print("\nDataset Information:")
print(final_dataset.info())

print("\nMissing Values:")
print(final_dataset.isnull().sum())

print("\nSummary Statistics:")
print(final_dataset.describe())


# ==========================================
# Summary Statistics by Dismissal Status
# ==========================================

summary = final_dataset.groupby("Dismissed").agg({
    "Wins": ["mean", "std"],
    "Draws": ["mean", "std"],
    "Losses": ["mean", "std"],
    "Goals_For": ["mean", "std"],
    "Goals_Against": ["mean", "std"],
    "Goal_Difference": ["mean", "std"],
    "Points": ["mean", "std"],
    "Points_Per_Match": ["mean", "std"]
})

print("\n==============================")
print("SUMMARY STATISTICS")
print("==============================")
print(summary.round(2))

summary.round(2).to_csv(
    "/Users/vizzy/Downloads/Premier league/Data/cleaned/Table_4_1_Summary_Statistics.csv"
)

print("\nTable 4.1 saved successfully.")


import matplotlib.pyplot as plt

plt.figure(figsize=(6,4))

final_dataset["Dismissed"].value_counts().sort_index().plot(
    kind="bar"
)

plt.title("Distribution of Managerial Dismissals")
plt.xlabel("Manager Dismissed")
plt.ylabel("Number of Team-Seasons")

plt.xticks([0,1],["No","Yes"])

plt.tight_layout()

plt.savefig(
"/Users/vizzy/Downloads/Premier league/Data/cleaned/Figure_4_1_Dismissals.png",
dpi=300
)

plt.show()


variables = [
    "Wins",
    "Draws",
    "Losses",
    "Goals_For",
    "Goals_Against",
    "Goal_Difference",
    "Points",
    "Points_Per_Match"
]

for var in variables:

    plt.figure(figsize=(6,4))

    final_dataset.boxplot(
        column=var,
        by="Dismissed"
    )

    plt.title(f"{var} by Managerial Dismissal")
    plt.suptitle("")
    plt.xlabel("Dismissed")
    plt.ylabel(var)

    plt.tight_layout()

    plt.savefig(
        f"/Users/vizzy/Downloads/Premier league/Data/cleaned/{var}_Boxplot.png",
        dpi=300
    )

    plt.show()
    
    
    
    import numpy as np

numeric = final_dataset.drop(
    columns=["Rk", "Matches_Played"]
).select_dtypes(include=np.number)

corr = numeric.corr()

plt.figure(figsize=(10,8))

plt.imshow(corr)

plt.xticks(
    range(len(corr.columns)),
    corr.columns,
    rotation=90
)

plt.yticks(
    range(len(corr.columns)),
    corr.columns
)

plt.colorbar()

plt.title("Correlation Matrix")

plt.tight_layout()

plt.savefig(
"/Users/vizzy/Downloads/Premier league/Data/cleaned/Figure_4_2_Correlation_Matrix.png",
dpi=300
)

plt.show()

print(corr.round(2))




# ==========================================
# Variance Inflation Factor (VIF)
# ==========================================

from statsmodels.stats.outliers_influence import variance_inflation_factor

# Select predictor variables
X = final_dataset[
    [
        "Wins",
        "Draws",
        "Losses",
        "Goals_For",
        "Goals_Against",
        "Goal_Difference",
        "Points",
        "Points_Per_Match"
    ]
]

# Calculate VIF
vif = pd.DataFrame()
vif["Variable"] = X.columns
vif["VIF"] = [
    variance_inflation_factor(X.values, i)
    for i in range(X.shape[1])
]

print("\n==============================")
print("VARIANCE INFLATION FACTOR")
print("==============================")
print(vif.round(2))



vif.round(2).to_csv(
    "/Users/vizzy/Downloads/Premier league/Data/cleaned/Table_4_2_VIF.csv",
    index=False
)

print("\nTable 4.2 saved successfully.")


# ==========================================
# Revised VIF Analysis
# ==========================================

from statsmodels.stats.outliers_influence import variance_inflation_factor

# Remove derived variables
X_revised = final_dataset[
    [
        "Wins",
        "Draws",
        "Losses",
        "Goals_For",
        "Goals_Against"
    ]
]

vif_revised = pd.DataFrame()

vif_revised["Variable"] = X_revised.columns

vif_revised["VIF"] = [
    variance_inflation_factor(X_revised.values, i)
    for i in range(X_revised.shape[1])
]

print("\n==============================")
print("REVISED VIF")
print("==============================")
print(vif_revised.round(2))


vif_revised.round(2).to_csv(
    "/Users/vizzy/Downloads/Premier league/Data/cleaned/Table_4_3_Revised_VIF.csv",
    index=False
)

print("\nTable 4.3 saved successfully.")


# ==========================================
# Final VIF Analysis
# ==========================================

X_final = final_dataset[
    [
        "Wins",
        "Draws",
        "Goals_For",
        "Goals_Against"
    ]
]

vif_final = pd.DataFrame()

vif_final["Variable"] = X_final.columns

vif_final["VIF"] = [
    variance_inflation_factor(X_final.values, i)
    for i in range(X_final.shape[1])
]

print("\n==============================")
print("FINAL VIF")
print("==============================")
print(vif_final.round(2))

vif_final.round(2).to_csv(
    "/Users/vizzy/Downloads/Premier league/Data/cleaned/Table_4_4_Final_VIF.csv",
    index=False
)

print("\nTable 4.4 saved successfully.")
