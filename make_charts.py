"""
make_charts.py
Step 4: Generate analytical charts from students.db.

Saves 5 PNG charts to the charts/ folder (used later in the Streamlit dashboard).
"""

import os
import sqlite3

import matplotlib
matplotlib.use("Agg")  # render to file without opening a window
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

os.makedirs("charts", exist_ok=True)
sns.set_theme(style="whitegrid")

# ---------- Load data from SQL ----------
conn = sqlite3.connect("students.db")
df = pd.read_sql_query("SELECT * FROM students", conn)
conn.close()

STREAM_ORDER = ["SCIENCE", "COMMERCE", "HUMANITIES"]
COLORS = {"SCIENCE": "#2a7de1", "COMMERCE": "#e8963a", "HUMANITIES": "#3aa66b"}

# ---------- Chart 1: Students per stream ----------
counts = df["stream"].value_counts().reindex(STREAM_ORDER)
plt.figure(figsize=(6, 4))
ax = sns.barplot(x=counts.index, y=counts.values, hue=counts.index, palette=COLORS, legend=False)
for i, v in enumerate(counts.values):
    ax.text(i, v + 3, str(v), ha="center")
plt.title("Students per Stream")
plt.xlabel("")
plt.ylabel("Number of students")
plt.tight_layout()
plt.savefig("charts/1_students_per_stream.png", dpi=150)
plt.close()

# ---------- Chart 2: Average marks by stream (grouped bar) ----------
marks = (
    df.groupby("stream")[["maths", "science", "english", "social_science"]]
    .mean()
    .reindex(STREAM_ORDER)
    .reset_index()
    .melt(id_vars="stream", var_name="subject", value_name="avg_marks")
)
plt.figure(figsize=(8, 4.5))
sns.barplot(data=marks, x="subject", y="avg_marks", hue="stream", hue_order=STREAM_ORDER, palette=COLORS)
plt.title("Average Class 10 Marks by Chosen Stream")
plt.xlabel("")
plt.ylabel("Average marks")
plt.ylim(0, 100)
plt.legend(title="Stream")
plt.tight_layout()
plt.savefig("charts/2_avg_marks_by_stream.png", dpi=150)
plt.close()

# ---------- Chart 3: Correlation heatmap (marks and interests) ----------
num_cols = [
    "maths", "science", "english", "social_science",
    "interest_science", "interest_commerce", "interest_humanities",
]
plt.figure(figsize=(8, 6))
sns.heatmap(df[num_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0, square=True)
plt.title("Correlation Heatmap (marks & interests)")
plt.tight_layout()
plt.savefig("charts/3_correlation_heatmap.png", dpi=150)
plt.close()

# ---------- Chart 4: Interest heatmap (stream vs interest area) ----------
interest = (
    df.groupby("stream")[["interest_science", "interest_commerce", "interest_humanities"]]
    .mean()
    .reindex(STREAM_ORDER)
)
interest.columns = ["Science", "Commerce", "Humanities"]
plt.figure(figsize=(6, 4))
sns.heatmap(interest, annot=True, fmt=".1f", cmap="YlGnBu")
plt.title("Average Interest Score\n(rows: chosen stream, columns: interest area)")
plt.ylabel("")
plt.tight_layout()
plt.savefig("charts/4_interest_heatmap.png", dpi=150)
plt.close()

# ---------- Chart 5: Overall marks distribution (boxplot) ----------
plt.figure(figsize=(6, 4.5))
sns.boxplot(data=df, x="stream", y="avg_marks", order=STREAM_ORDER, hue="stream", palette=COLORS, legend=False)
plt.title("Overall Average Marks Distribution by Stream")
plt.xlabel("")
plt.ylabel("Average marks")
plt.tight_layout()
plt.savefig("charts/5_marks_boxplot.png", dpi=150)
plt.close()

print("Done. 5 charts saved in the 'charts' folder:")
for f in sorted(os.listdir("charts")):
    print(" -", f)
