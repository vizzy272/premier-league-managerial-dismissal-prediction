#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 15:02:57 2026

@author: vizzy
"""

import pandas as pd

# ==========================================
# Load cleaned league table
# ==========================================

file_path = "/Users/vizzy/Downloads/Premier league/Data/cleaned/PremierLeague_Cleaned.csv"

df = pd.read_csv(file_path)

# ==========================================
# Keep only the columns I need
# ==========================================

dismissal_df = df[["Season", "Team"]].copy()

# Add new columns
dismissal_df["Manager"] = ""
dismissal_df["Dismissed"] = 0

# Sort nicely
dismissal_df = dismissal_df.sort_values(
    by=["Season", "Team"]
).reset_index(drop=True)

# ==========================================
# Save template
# ==========================================

output_file = "/Users/vizzy/Downloads/Premier league/Data/cleaned/Manager_Dismissal_Template.csv"

dismissal_df.to_csv(output_file, index=False)

print("=" * 50)
print("Template created successfully!")
print("=" * 50)

print(dismissal_df.head(20))

print("\nShape:")
print(dismissal_df.shape)

print("\nSaved to:")
print(output_file)