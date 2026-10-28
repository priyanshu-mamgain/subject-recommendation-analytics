"""
sql_analysis.py
Step 3: Load the cleaned data into SQLite and run analytical SQL queries.

Creates students.db with two tables:
  - students: cleaned student records
  - stream_subjects: compulsory subjects for each stream
Then runs 10 queries covering GROUP BY, HAVING, CASE, subqueries, window functions and JOINs.
"""

import sqlite3
import pandas as pd

DB_FILE = "students.db"

# ---------- 1. Create the database and load data ----------
conn = sqlite3.connect(DB_FILE)

df = pd.read_csv("students_clean.csv")
df.to_sql("students", conn, if_exists="replace", index=False)

# Compulsory subjects per stream (from the original school project)
stream_subjects = pd.DataFrame(
    [
        ("SCIENCE", "PHYSICS"),
        ("SCIENCE", "CHEMISTRY"),
        ("COMMERCE", "BST"),
        ("COMMERCE", "ACCOUNTS"),
        ("HUMANITIES", "HISTORY"),
        ("HUMANITIES", "GEOGRAPHY"),
    ],
    columns=["stream", "compulsory_subject"],
)
stream_subjects.to_sql("stream_subjects", conn, if_exists="replace", index=False)

print(f"Database ready: {DB_FILE}  ({len(df)} students)\n")

# ---------- 2. Queries ----------
# Each entry: (title, SQL)
queries = [
    (
        "1. Number of students per stream",
        """
        SELECT stream, COUNT(*) AS total_students
        FROM students
        GROUP BY stream
        ORDER BY total_students DESC;
        """,
    ),
    (
        "2. Average marks per subject by stream",
        """
        SELECT stream,
               ROUND(AVG(maths), 1)          AS avg_maths,
               ROUND(AVG(science), 1)        AS avg_science,
               ROUND(AVG(english), 1)        AS avg_english,
               ROUND(AVG(social_science), 1) AS avg_sst
        FROM students
        GROUP BY stream;
        """,
    ),
    (
        "3. Top 5 students by average marks",
        """
        SELECT student_id, stream, avg_marks
        FROM students
        ORDER BY avg_marks DESC
        LIMIT 5;
        """,
    ),
    (
        "4. Percentage of students who took Maths, by stream",
        """
        SELECT stream,
               ROUND(100.0 * SUM(CASE WHEN takes_maths = 'YES' THEN 1 ELSE 0 END) / COUNT(*), 1) AS pct_took_maths
        FROM students
        GROUP BY stream;
        """,
    ),
    (
        "5. Elective subject popularity by stream",
        """
        SELECT stream, elective, COUNT(*) AS students
        FROM students
        GROUP BY stream, elective
        ORDER BY stream, students DESC;
        """,
    ),
    (
        "6. Streams with an average interest score above 6",
        """
        SELECT stream,
               ROUND(AVG(interest_science), 1)    AS avg_int_science,
               ROUND(AVG(interest_commerce), 1)   AS avg_int_commerce,
               ROUND(AVG(interest_humanities), 1) AS avg_int_humanities
        FROM students
        GROUP BY stream
        HAVING AVG(interest_science) > 6
            OR AVG(interest_commerce) > 6
            OR AVG(interest_humanities) > 6;
        """,
    ),
    (
        "7. Students per marks band (A / B / C) by stream",
        """
        SELECT stream,
               CASE
                   WHEN avg_marks >= 75 THEN 'A (75+)'
                   WHEN avg_marks >= 60 THEN 'B (60-74)'
                   ELSE 'C (below 60)'
               END AS band,
               COUNT(*) AS students
        FROM students
        GROUP BY stream, band
        ORDER BY stream, band;
        """,
    ),
    (
        "8. Mismatch: chosen stream differs from the highest-interest stream",
        """
        SELECT stream AS chosen_stream,
               COUNT(*) AS mismatched_students
        FROM (
            SELECT stream,
                   CASE
                       WHEN interest_science >= interest_commerce AND interest_science >= interest_humanities THEN 'SCIENCE'
                       WHEN interest_commerce >= interest_humanities THEN 'COMMERCE'
                       ELSE 'HUMANITIES'
                   END AS top_interest
            FROM students
        )
        WHERE stream <> top_interest
        GROUP BY stream;
        """,
    ),
    (
        "9. Top 3 students per stream (window function: RANK)",
        """
        SELECT stream, student_id, avg_marks, rank_in_stream
        FROM (
            SELECT stream, student_id, avg_marks,
                   RANK() OVER (PARTITION BY stream ORDER BY avg_marks DESC) AS rank_in_stream
            FROM students
        )
        WHERE rank_in_stream <= 3
        ORDER BY stream, rank_in_stream;
        """,
    ),
    (
        "10. JOIN: students required to study each compulsory subject",
        """
        SELECT ss.stream, ss.compulsory_subject, COUNT(s.student_id) AS students
        FROM stream_subjects ss
        JOIN students s ON s.stream = ss.stream
        GROUP BY ss.stream, ss.compulsory_subject
        ORDER BY ss.stream;
        """,
    ),
]

for title, sql in queries:
    print("=" * 70)
    print(title)
    print("=" * 70)
    result = pd.read_sql_query(sql, conn)
    print(result.to_string(index=False))
    print()

conn.close()
print("Done. Database file:", DB_FILE)
