#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
convert_and_run_prefilter.py
-------------------------------------------------
Reads all .xlsx files from `testdata/`, converts them to a single combined CSV,
then runs the improved CSE-aware prefilter (light classifier).

Output:
    - combined_testdata.csv  (merged CSV from all XLSX files)
    - prefilter_scored.csv   (CSE-aware scores and risk buckets)
"""

import os
import pandas as pd
import subprocess

# ---------- CONFIG ----------
TEST_FOLDER = "testdata"
COMBINED_CSV = "combined_testdata.csv"
PREFILTER_SCRIPT = "lightClassifier.py"   # Ensure this file exists
PREFILTER_OUTPUT = "prefilter_scored.csv"

# ---------- STEP 1: Read all .xlsx files ----------
all_files = [f for f in os.listdir(TEST_FOLDER) if f.lower().endswith(".xlsx")]
if not all_files:
    raise FileNotFoundError(f"No .xlsx files found in '{TEST_FOLDER}'")

print(f"📂 Found {len(all_files)} Excel files in {TEST_FOLDER}/")

dfs = []
for fname in all_files:
    path = os.path.join(TEST_FOLDER, fname)
    try:
        df = pd.read_excel(path)
        df["source_file"] = fname  # keep track of origin
        dfs.append(df)
        print(f"✅ Loaded {fname} ({len(df)} rows)")
    except Exception as e:
        print(f"⚠️ Failed to read {fname}: {e}")

if not dfs:
    raise ValueError("No readable Excel files found!")

combined_df = pd.concat(dfs, ignore_index=True)
combined_df.to_csv(COMBINED_CSV, index=False)
print(f"\n💾 Combined CSV saved as {COMBINED_CSV} ({len(combined_df)} rows)")

# ---------- STEP 2: Run the light classifier ----------
print("\n🚀 Running CSE-aware prefilter (light classifier)...")
cmd = ["python3", PREFILTER_SCRIPT, "--input", COMBINED_CSV, "--out", PREFILTER_OUTPUT]
subprocess.run(cmd, check=True)

print(f"\n✅ Prefilter completed successfully. Results saved to {PREFILTER_OUTPUT}")
