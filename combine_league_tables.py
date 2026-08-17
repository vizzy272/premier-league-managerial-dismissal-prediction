#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 14:28:01 2026

@author: vizzy
"""

import pandas as pd
import glob
import os



folder_path = "/Users/vizzy/Downloads/Premier league/Data/Raw"


csv_files = glob.glob(os.path.join(folder_path, "*.csv"))

print("=" * 50)
print("CSV files found:", len(csv_files))
print("=" * 50)


for file in csv_files:
    print(os.path.basename(file))



# ==========================================
# Combine all seasons
# ==========================================

all_data = []

for file in sorted(csv_files):

    # Read CSV
    df = pd.read_csv(file)

    # Get season from filename
    season = os.path.basename(file).replace("_League_Table.csv", "")

    # Add season column
    df["Season"] = season

    # Store dataframe
    all_data.append(df)

# Merge all seasons
combined = pd.concat(all_data, ignore_index=True)

print("\nDataset merged successfully!")
print("Rows, Columns:", combined.shape)

print("\nFirst five rows:")
print(combined.head())

# ==========================================
# Save merged dataset
# ==========================================

output_folder = "/Users/vizzy/Downloads/Premier league/Data/cleaned"


os.makedirs(output_folder, exist_ok=True)

output_file = os.path.join(
    output_folder,
    "PremierLeague_2010_2025.csv"
)

combined.to_csv(output_file, index=False)

print("\nMerged dataset saved successfully!")
print(output_file)