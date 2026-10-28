"""
app.py
Step 6: Streamlit dashboard for stream recommendation and analytics.

Run with: streamlit run app.py
Required files in the same folder: students.db, stream_model.joblib, feature_importance.csv, charts/
"""

import os
import sqlite3

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Subject Recommendation & Analytics", page_icon="🎓", layout="wide")


@st.cache_data
def load_students():
    conn = sqlite3.connect("students.db")
    df = pd.read_sql_query("SELECT * FROM students", conn)
    conn.close()
    return df


@st.cache_resource
def load_model():
    return joblib.load("stream_model.joblib")


df = load_students()
saved = load_model()

st.title("🎓 Student Subject Recommendation & Analytics")
st.caption("Note: this dashboard is built on synthetic student data.")

tab1, tab2, tab3 = st.tabs(["Recommendation", "Analytics", "About"])

# ---------- Tab 1: Recommendation ----------
with tab1:
    st.subheader("Enter marks and interests")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Class 10 marks**")
        maths = st.slider("Maths", 0, 100, 70)
        science = st.slider("Science", 0, 100, 65)
        english = st.slider("English", 0, 100, 70)
        sst = st.slider("Social Science", 0, 100, 65)
    with col2:
        st.markdown("**Interest level (1 = low, 10 = high)**")
        i_sci = st.slider("Science", 1, 10, 6)
        i_com = st.slider("Commerce", 1, 10, 6)
        i_hum = st.slider("Humanities", 1, 10, 5)

    if st.button("Recommend stream", type="primary"):
        row = pd.DataFrame(
            [[maths, science, english, sst, i_sci, i_com, i_hum]],
            columns=saved["features"],
        )
        probs = saved["model"].predict_proba(row)[0]
        result = pd.Series(probs, index=saved["model"].classes_).sort_values(ascending=False)

        st.success(f"Recommended stream: **{result.index[0]}** ({result.iloc[0]:.0%} confidence)")
        st.bar_chart((result * 100).round(1), y_label="Confidence %")

        if result.iloc[0] < 0.6:
            st.info("Confidence is low: the top streams are close. Consider exploring both before deciding.")

    st.divider()
    st.markdown("**Which factors influence the recommendation most?**")
    if os.path.exists("feature_importance.csv"):
        imp = pd.read_csv("feature_importance.csv").set_index("feature")
        st.bar_chart(imp["importance"], y_label="Importance")

# ---------- Tab 2: Analytics ----------
with tab2:
    st.subheader("Dataset overview")
    m1, m2, m3 = st.columns(3)
    m1.metric("Total students", len(df))
    m2.metric("Average marks", f"{df['avg_marks'].mean():.1f}")
    m3.metric("Took Maths", f"{(df['takes_maths'] == 'YES').mean():.0%}")

    selected = st.multiselect(
        "Filter by stream", ["SCIENCE", "COMMERCE", "HUMANITIES"],
        default=["SCIENCE", "COMMERCE", "HUMANITIES"],
    )
    view = df[df["stream"].isin(selected)]

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Students per stream**")
        st.bar_chart(view["stream"].value_counts())
    with c2:
        st.markdown("**Average marks by stream**")
        st.bar_chart(view.groupby("stream")[["maths", "science", "english", "social_science"]].mean())

    st.markdown("**Heatmaps**")
    h1, h2 = st.columns(2)
    if os.path.exists("charts/3_correlation_heatmap.png"):
        h1.image("charts/3_correlation_heatmap.png")
    if os.path.exists("charts/4_interest_heatmap.png"):
        h2.image("charts/4_interest_heatmap.png")

    with st.expander("View raw data"):
        st.dataframe(view, width="stretch")

# ---------- Tab 3: About ----------
with tab3:
    st.markdown(
        """
**Pipeline:** synthetic data generation (Python) → cleaning (pandas) → SQLite and SQL analysis →
visualisation (matplotlib, seaborn) → modelling (scikit-learn, Logistic Regression) → Streamlit dashboard.

**Limitation:** the dataset is synthetic, so model accuracy does not reflect real-world performance.
The project demonstrates an end-to-end data workflow.
"""
    )
