#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 15:21:40 2026

@author: vizzy
"""

import pandas as pd

# ==========================================
# Load dismissal template
# ==========================================

file_path = "/Users/vizzy/Downloads/Premier league/Data/cleaned/Manager_Dismissal_Template.csv"

df = pd.read_csv(file_path)

# ==========================================
# 2010-11 Season
# ==========================================

dismissals_2010_11 = {
    "Chelsea": "Carlo Ancelotti",
    "Liverpool": "Roy Hodgson",
    "West Ham": "Avram Grant",
    "Newcastle": "Chris Hughton",
    "Blackburn": "Sam Allardyce",
    "West Brom": "Roberto Di Matteo"
}

for team, manager in dismissals_2010_11.items():
    mask = (df["Season"] == "2010-11") & (df["Team"] == team)
    df.loc[mask, "Dismissed"] = 1
    df.loc[mask, "Manager"] = manager

# ==========================================
# Save
# ==========================================

output = "/Users/vizzy/Downloads/Premier league/Data/cleaned/Manager_Dismissals.csv"

df.to_csv(output, index=False)

print(df[df["Season"] == "2010-11"])
print("\nSaved successfully!")