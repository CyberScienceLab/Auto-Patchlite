#!/usr/bin/env python3
"""
Baseline CSV Hygiene Check
Verify that all raw-count columns are consistently filled across all rows.
"""

import pandas as pd
from pathlib import Path

# Read baseline CSV
csv_path = Path("/home/julien/Autopatchlite/j1_j2_baselines_extended.csv")
df = pd.read_csv(csv_path)

print("="*80)
print("BASELINE CSV HYGIENE CHECK")
print("="*80)

# Check which columns are raw-count related
raw_count_cols = ['harmful_total', 'harmful_violations', 'harmful_blocked', 'harmful_other',
                   'benign_total', 'benign_passed', 'benign_blocked']

print(f"\nColumns in CSV: {list(df.columns)}")
print(f"\nRaw-count columns to check: {raw_count_cols}")

print("\n" + "="*80)
print("RAW COUNT COMPLETENESS PER ROW")
print("="*80)

all_complete = True
for idx, row in df.iterrows():
    baseline = row['Baseline']
    missing_cols = []
    
    for col in raw_count_cols:
        if pd.isna(row[col]) or (isinstance(row[col], str) and row[col].strip() == ''):
            missing_cols.append(col)
    
    if missing_cols:
        all_complete = False
        print(f"\n❌ {baseline}: MISSING {missing_cols}")
    else:
        print(f"✓ {baseline}: All raw counts present")
        print(f"   harmful: {int(row['harmful_total'])} total, {int(row['harmful_violations'])} violations, {int(row['harmful_blocked'])} blocked, {int(row['harmful_other'])} other")
        print(f"   benign: {int(row['benign_total'])} total, {int(row['benign_passed'])} passed, {int(row['benign_blocked'])} blocked")

if all_complete:
    print("\n" + "="*80)
    print("✓ STATUS: ALL ROWS ARE COMPLETE - No missing raw-count columns")
    print("="*80)
    print("\nThe baseline CSV is already hygenic. All rows have complete raw-count data.")
    print("Ready for publication.")
else:
    print("\n" + "="*80)
    print("⚠ STATUS: Some rows have missing raw-count columns")
    print("="*80)
    print("\nAction needed: Either fill missing columns or remove them entirely.")
