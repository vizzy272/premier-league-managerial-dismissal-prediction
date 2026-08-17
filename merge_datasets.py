#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 26 20:42:45 2026

@author: vizzy
"""

import pandas as pd

# Load datasets
performance = pd.read_csv(
    "/Users/vizzy/Downloads/Premier league/Data/cleaned/PremierLeague_Cleaned.csv"
)

dismissals = pd.read_csv(
    "/Users/vizzy/Downloads/Premier league/Data/cleaned/Manager_Dismissal_Template_Reviewed.csv"
)

# Merge
merged = performance.merge(
    dismissals[["Season", "Team", "Dismissed"]],
    on=["Season", "Team"],
    how="left"
)

# Fill any missing values
merged["Dismissed"] = merged["Dismissed"].fillna(0).astype(int)

# Save
merged.to_csv(
    "/Users/vizzy/Downloads/Premier league/Data/cleaned/Final_Dataset.csv",
    index=False
)

print("Merge complete!")
print(merged.head())
print()
print(merged.shape)
print()
print(merged["Dismissed"].value_counts())