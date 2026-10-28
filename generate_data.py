"""
generate_data.py
Step 1: Generate a synthetic student dataset for the Subject Recommendation & Analytics project.

The data is synthetic (not real student records). Output: students_raw.csv.
It intentionally contains missing values and duplicate rows so the cleaning step has real work to do.
"""

import numpy as np
import pandas as pd

rng = np.random.default_rng(42)  # fixed seed -> reproducible dataset

N_STUDENTS = 800


def clip(values, low, high):
    """Keep values inside a valid range (marks 0-100, interest 1-10)."""
    return np.clip(values, low, high)


# ---------- 1. Class 10 marks ----------
# A shared "ability" factor makes marks moderately correlated, as in real data.
ability = rng.normal(0, 1, N_STUDENTS)

maths = clip(65 + 12 * ability + rng.normal(0, 12, N_STUDENTS), 15, 100)
science = clip(65 + 11 * ability + rng.normal(0, 12, N_STUDENTS), 15, 100)
english = clip(68 + 8 * ability + rng.normal(0, 11, N_STUDENTS), 20, 100)
social_science = clip(68 + 9 * ability + rng.normal(0, 12, N_STUDENTS), 20, 100)

# ---------- 2. Interest scores (1-10) ----------
interest_science = clip(np.round(5 + (maths + science - 130) / 20 + rng.normal(0, 2, N_STUDENTS)), 1, 10)
interest_commerce = clip(np.round(5 + (maths - 65) / 30 + rng.normal(0, 2.2, N_STUDENTS)), 1, 10)
interest_humanities = clip(np.round(5 + (social_science + english - 135) / 25 + rng.normal(0, 2, N_STUDENTS)), 1, 10)

# ---------- 3. Stream selection ----------
# Each stream gets a weighted score from marks and interest; the highest score wins.
# Random noise reflects that real decisions are not perfectly predictable.
sci_score = 0.40 * maths + 0.40 * science + 4 * interest_science
com_score = 0.45 * maths + 0.20 * english + 4 * interest_commerce + 8
hum_score = 0.35 * social_science + 0.35 * english + 4 * interest_humanities + 2

scores = np.vstack([sci_score, com_score, hum_score]).T
scores = scores + rng.normal(0, 6, scores.shape)
stream = np.array(["SCIENCE", "COMMERCE", "HUMANITIES"])[scores.argmax(axis=1)]

# ---------- 4. Maths choice and elective subject ----------
electives = ["PSYCHOLOGY", "ART", "MUSIC", "DANCE", "PHYSICAL EDUCATION"]

takes_maths = np.where(
    stream == "SCIENCE",
    np.where(maths > 55, "YES", "NO"),
    np.where(stream == "COMMERCE", np.where(maths > 65, "YES", "NO"), "NO"),
)
elective = rng.choice(electives, N_STUDENTS)

# ---------- 5. Build the DataFrame ----------
df = pd.DataFrame(
    {
        "student_id": [f"STU{i:04d}" for i in range(1, N_STUDENTS + 1)],
        "maths": np.round(maths, 1),
        "science": np.round(science, 1),
        "english": np.round(english, 1),
        "social_science": np.round(social_science, 1),
        "interest_science": interest_science.astype(int),
        "interest_commerce": interest_commerce.astype(int),
        "interest_humanities": interest_humanities.astype(int),
        "stream": stream,
        "takes_maths": takes_maths,
        "elective": elective,
    }
)

# ---------- 6. Inject realistic data-quality issues ----------
# (a) ~3% missing values in each marks column
for col in ["maths", "science", "english", "social_science"]:
    missing_idx = rng.choice(df.index, size=int(0.03 * N_STUDENTS), replace=False)
    df.loc[missing_idx, col] = np.nan

# (b) 15 duplicate rows
duplicates = df.sample(15, random_state=1)
df = pd.concat([df, duplicates], ignore_index=True)

# (c) shuffle row order
df = df.sample(frac=1, random_state=7).reset_index(drop=True)

# ---------- 7. Save ----------
df.to_csv("students_raw.csv", index=False)

print(f"Done. {len(df)} rows saved to students_raw.csv")
print("\nStream distribution (after removing duplicates):")
print(df.drop_duplicates()["stream"].value_counts())
print("\nMissing values per column:")
print(df.isna().sum())
