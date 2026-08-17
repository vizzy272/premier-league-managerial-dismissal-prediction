#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 14:50:09 2026

@author: vizzy
"""

import pandas as pd
import os

# ==========================================
# Load merged dataset
# ==========================================

file_path = "/Users/vizzy/Downloads/Premier league/Data/cleaned/PremierLeague_2010_2025.csv"

df = pd.read_csv(file_path)

print("=" * 50)
print("DATASET INFORMATION")
print("=" * 50)

print("\nShape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nData Types:")
print(df.dtypes)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:")
print(df.duplicated().sum())

# ==========================================
# Remove columns I don't need
# ==========================================

columns_to_drop = [
    "Attendance",
    "Top Team Scorer",
    "Goalkeeper",
    "Notes"
]

df = df.drop(columns=columns_to_drop)

# ==========================================
# Rename columns
# ==========================================

df = df.rename(columns={
    "Squad": "Team",
    "MP": "Matches_Played",
    "W": "Wins",
    "D": "Draws",
    "L": "Losses",
    "GF": "Goals_For",
    "GA": "Goals_Against",
    "GD": "Goal_Difference",
    "Pts": "Points",
    "Pts/MP": "Points_Per_Match"
})

print("\n")
print("=" * 50)
print("CLEANED DATASET")
print("=" * 50)

print(df.head())

print("\nFinal Shape:")
print(df.shape)

# ==========================================
# Save cleaned dataset
# ==========================================

output_file = "/Users/vizzy/Downloads/Premier league/Data/cleaned/PremierLeague_Cleaned.csv"

df.to_csv(output_file, index=False)

print("\nCleaned dataset saved successfully!")
print(output_file)