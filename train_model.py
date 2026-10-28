"""
train_model.py
Step 5: Train and evaluate a stream recommendation model.

  1. Load data from students.db
  2. Compare Logistic Regression and Random Forest on a held-out test set
  3. Save the better model (stream_model.joblib) for the dashboard
  4. Explain predictions with permutation feature importance
  5. Demonstrate a recommendation with confidence scores
"""

import sqlite3

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# ---------- 1. Load data ----------
conn = sqlite3.connect("students.db")
df = pd.read_sql_query("SELECT * FROM students", conn)
conn.close()

# X = input features, y = target (stream)
FEATURES = [
    "maths", "science", "english", "social_science",
    "interest_science", "interest_commerce", "interest_humanities",
]
X = df[FEATURES]
y = df["stream"]

# ---------- 2. Train / Test split ----------
# 80% of the data trains the model; 20% is held out to evaluate it on unseen students
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train rows: {len(X_train)}, Test rows: {len(X_test)}\n")

# ---------- 3. Compare two models ----------
models = {
    "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    results[name] = acc
    print(f"{name}: test accuracy = {acc:.1%}")

best_name = max(results, key=results.get)
best_model = models[best_name]
print(f"\nBest model: {best_name}")

# ---------- 4. Detailed report ----------
y_pred = best_model.predict(X_test)
print("\nClassification report:")
print(classification_report(y_test, y_pred))

labels = ["SCIENCE", "COMMERCE", "HUMANITIES"]
cm = pd.DataFrame(
    confusion_matrix(y_test, y_pred, labels=labels),
    index=[f"actual_{l}" for l in labels],
    columns=[f"pred_{l}" for l in labels],
)
print("Confusion matrix (rows = actual stream, columns = predicted stream):")
print(cm)

# ---------- 5. Feature importance (explainability) ----------
# Permutation importance: shuffle one feature and measure the drop in accuracy.
# A larger drop means the feature matters more to the model.
perm = permutation_importance(best_model, X_test, y_test, n_repeats=20, random_state=42)
importance = (
    pd.DataFrame({"feature": FEATURES, "importance": perm.importances_mean})
    .sort_values("importance", ascending=False)
    .reset_index(drop=True)
)
importance["importance"] = importance["importance"].round(3)
print("\nFeature importance (accuracy drop when feature is shuffled):")
print(importance.to_string(index=False))

importance.to_csv("feature_importance.csv", index=False)

# ---------- 6. Save the model ----------
joblib.dump({"model": best_model, "features": FEATURES}, "stream_model.joblib")
print("\nSaved: stream_model.joblib, feature_importance.csv")


# ---------- 7. Recommendation function ----------
def recommend(maths, science, english, social_science,
              interest_science, interest_commerce, interest_humanities):
    """Return the model's confidence for each stream, given one student's marks and interests."""
    saved = joblib.load("stream_model.joblib")
    row = pd.DataFrame([[maths, science, english, social_science,
                         interest_science, interest_commerce, interest_humanities]],
                       columns=saved["features"])
    probs = saved["model"].predict_proba(row)[0]
    classes = saved["model"].classes_
    return sorted(zip(classes, probs), key=lambda x: x[1], reverse=True)


# ---------- 8. Demo ----------
print("\n" + "=" * 60)
print("DEMO: sample student")
print("=" * 60)
sample = dict(maths=72, science=64, english=70, social_science=66,
              interest_science=6, interest_commerce=7, interest_humanities=5)
print("Input:", sample)
for stream, p in recommend(**sample):
    print(f"  {stream:<11} {p:.0%}")
