"""
clean_data.py
Step 2: Clean the raw student data.

Reads students_raw.csv and writes students_clean.csv.
Steps: inspect -> remove duplicates -> impute missing values -> validate ranges
       -> standardize text -> add derived columns -> save.
"""

import pandas as pd

MARK_COLS = ["maths", "science", "english", "social_science"]
INTEREST_COLS = ["interest_science", "interest_commerce", "interest_humanities"]

# ---------- 1. Load and inspect ----------
df = pd.read_csv("students_raw.csv")

print("=== BEFORE CLEANING ===")
print("Shape (rows, columns):", df.shape)
print("Duplicate rows:", df.duplicated().sum())
print("Missing values:")
print(df.isna().sum()[df.isna().sum() > 0])

rows_before = len(df)

# ---------- 2. Remove duplicates ----------
df = df.drop_duplicates().reset_index(drop=True)

# ---------- 3. Impute missing marks ----------
# Stream-wise median: robust to outliers and more accurate than a global value.
for col in MARK_COLS:
    stream_median = df.groupby("stream")[col].transform("median")
    df[col] = df[col].fillna(stream_median)

# ---------- 4. Validate ranges ----------
bad_marks = ((df[MARK_COLS] < 0) | (df[MARK_COLS] > 100)).sum().sum()
bad_interest = ((df[INTEREST_COLS] < 1) | (df[INTEREST_COLS] > 10)).sum().sum()
print("\nInvalid marks found:", bad_marks)
print("Invalid interest scores found:", bad_interest)

# ---------- 5. Standardize text ----------
for col in ["stream", "takes_maths", "elective"]:
    df[col] = df[col].str.strip().str.upper()

# ---------- 6. Derived columns ----------
df["avg_marks"] = df[MARK_COLS].mean(axis=1).round(1)

# ---------- 7. Save ----------
df.to_csv("students_clean.csv", index=False)

# ---------- Summary ----------
print("\n=== AFTER CLEANING ===")
print("Rows:", rows_before, "->", len(df), f"({rows_before - len(df)} duplicates removed)")
print("Missing values left:", df.isna().sum().sum())

print("\nAverage marks by stream:")
print(df.groupby("stream")[MARK_COLS + ["avg_marks"]].mean().round(1))

print("\nAverage interest by stream:")
print(df.groupby("stream")[INTEREST_COLS].mean().round(1))

print("\nSaved: students_clean.csv")
