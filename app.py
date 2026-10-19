"""
Student Performance Analytics Dashboard
Run with:  streamlit run app.py
"""
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import analytics as A

st.set_page_config(page_title="Student Performance Analytics", page_icon="🎓", layout="wide")
sns.set_theme(style="whitegrid")

FILES = {"Mathematics": "data/student-mat.csv", "Portuguese": "data/student-por.csv"}


@st.cache_data
def get_data(subject, uploaded=None):
    if uploaded is not None:
        return A.load_data(uploaded, "Uploaded")
    return A.load_data(FILES[subject], subject)


def train_for(df, use_prior):
    reg_tbl, reg_fit, _, Xr, _ = A.train_regressors(df, use_prior)
    clf_tbl, clf_fit, cms, Xc, _ = A.train_classifiers(df, use_prior)
    return reg_tbl, reg_fit, clf_tbl, clf_fit, cms, Xc


# ---------------------------------------------------------------- sidebar
st.sidebar.title("🎓 Controls")
subject = st.sidebar.radio("Course dataset", list(FILES.keys()))
up = st.sidebar.file_uploader("...or upload a csv in the same format (';' separated)", type="csv")
df_all = get_data(subject, up)

st.sidebar.markdown("---")
st.sidebar.subheader("Filters")
schools = st.sidebar.multiselect("School", sorted(df_all.school.unique()), default=sorted(df_all.school.unique()))
sexes = st.sidebar.multiselect("Sex", sorted(df_all.sex.unique()), default=sorted(df_all.sex.unique()))
age_lo, age_hi = st.sidebar.slider("Age range", int(df_all.age.min()), int(df_all.age.max()),
                                   (int(df_all.age.min()), int(df_all.age.max())))
addr = st.sidebar.multiselect("Address", sorted(df_all.address.unique()), default=sorted(df_all.address.unique()))

df = df_all[df_all.school.isin(schools) & df_all.sex.isin(sexes) &
            df_all.age.between(age_lo, age_hi) & df_all.address.isin(addr)]

st.title("Student Performance Analytics Dashboard")
st.caption("B.Tech CSE NTCC project - data source: UCI Student Performance dataset (Cortez & Silva, 2008)")

if df.empty:
    st.warning("No students match the current filters.")
    st.stop()

tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Overview", "🔎 Explore factors", "🤖 Predict grade",
                                          "⚠️ At-risk students", "ℹ️ About"])

# ---------------------------------------------------------------- overview
with tab1:
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Students", len(df))
    c2.metric("Average final grade", f"{df.G3.mean():.2f} / 20")
    c3.metric("Pass rate (G3 >= 10)", f"{(df.G3 >= A.PASS_MARK).mean()*100:.1f}%")
    c4.metric("At-risk students", int(df.at_risk.sum()))
    c5.metric("Avg. absences", f"{df.absences.mean():.1f}")

    a, b = st.columns(2)
    with a:
        fig, ax = plt.subplots(figsize=(5.5, 3.6))
        sns.histplot(df.G3, bins=21, ax=ax, color="#2E5C8A")
        ax.axvline(A.PASS_MARK, color="red", ls="--"); ax.set_title("Distribution of final grade (G3)")
        st.pyplot(fig)
    with b:
        fig, ax = plt.subplots(figsize=(5.5, 3.6))
        df[["G1", "G2", "G3"]].mean().plot(kind="bar", ax=ax, color="#C9732B")
        ax.set_ylabel("Mean grade"); ax.set_title("Average grade by period"); plt.xticks(rotation=0)
        st.pyplot(fig)

    st.subheader("Correlation of each numeric factor with the final grade")
    num = ["age", "Medu", "Fedu", "traveltime", "studytime", "failures", "famrel", "freetime",
           "goout", "Dalc", "Walc", "health", "absences", "G1", "G2"]
    cor = df[num + ["G3"]].corr()["G3"].drop("G3").sort_values()
    fig, ax = plt.subplots(figsize=(8, 4))
    cor.plot(kind="barh", ax=ax, color=["#C0392B" if v < 0 else "#2E5C8A" for v in cor])
    ax.set_xlabel("Pearson correlation with G3")
    st.pyplot(fig)

# ---------------------------------------------------------------- explore
with tab2:
    cat_options = {"Weekly study time": "studytime", "Past failures": "failures",
                   "Wants higher education": "higher", "Internet at home": "internet",
                   "Mother's education": "Medu", "Father's education": "Fedu",
                   "Sex": "sex", "Address (urban/rural)": "address", "Extra school support": "schoolsup",
                   "Romantic relationship": "romantic", "Going out with friends": "goout",
                   "Weekend alcohol use": "Walc", "Paid extra classes": "paid"}
    pick = st.selectbox("Compare final grade by:", list(cat_options.keys()))
    col = cat_options[pick]
    a, b = st.columns(2)
    with a:
        fig, ax = plt.subplots(figsize=(5.5, 3.8))
        sns.boxplot(data=df, x=col, y="G3", ax=ax, palette="Blues"); ax.set_title(f"G3 by {pick.lower()}")
        st.pyplot(fig)
    with b:
        summary = df.groupby(col).agg(students=("G3", "size"), mean_G3=("G3", "mean"),
                                      at_risk_pct=("at_risk", lambda s: s.mean() * 100)).round(2)
        st.dataframe(summary)
        st.caption("Small groups (few students) can give unreliable averages - check the 'students' column.")

    st.subheader("Absences vs final grade")
    ex_zero = st.checkbox("Hide students with G3 = 0 (they often have 0 absences - see About tab)")
    d2 = df[df.G3 > 0] if ex_zero else df
    fig, ax = plt.subplots(figsize=(7, 3.8))
    sns.regplot(data=d2, x="absences", y="G3", scatter_kws={"alpha": .5}, line_kws={"color": "red"}, ax=ax)
    st.pyplot(fig)
    st.write(f"Correlation: **{d2.absences.corr(d2.G3):.3f}**")

# ---------------------------------------------------------------- predict
with tab3:
    st.subheader("Predict a student's final grade")
    mode = st.radio("Which information is available?",
                    ["Early warning (before any exam - no G1/G2)", "Mid-term (G1 and G2 known)"], horizontal=True)
    use_prior = mode.startswith("Mid")
    reg_tbl, reg_fit, clf_tbl, clf_fit, cms, Xc = train_for(df_all, use_prior)
    st.write("Model quality on the 20% test set:")
    st.dataframe(reg_tbl)
    if not use_prior:
        st.info("Without earlier grades the model explains only a small part of the variation. "
                "Treat predictions as a rough warning, not a verdict.")

    base = df_all.copy()
    with st.form("what_if"):
        c = st.columns(4)
        vals = {}
        vals["studytime"] = c[0].selectbox("Study time (1-4)", [1, 2, 3, 4], index=1)
        vals["failures"] = c[1].selectbox("Past failures", [0, 1, 2, 3], index=0)
        vals["absences"] = c[2].number_input("Absences", 0, 100, 4)
        vals["higher"] = c[3].selectbox("Wants higher education", ["yes", "no"])
        c = st.columns(4)
        vals["goout"] = c[0].slider("Going out (1-5)", 1, 5, 3)
        vals["health"] = c[1].slider("Health (1-5)", 1, 5, 3)
        vals["Medu"] = c[2].slider("Mother's education (0-4)", 0, 4, 2)
        vals["Fedu"] = c[3].slider("Father's education (0-4)", 0, 4, 2)
        if use_prior:
            c = st.columns(2)
            vals["G1"] = c[0].slider("G1", 0, 20, 10)
            vals["G2"] = c[1].slider("G2", 0, 20, 10)
        go = st.form_submit_button("Predict")
    if go:
        # start from the "typical student" (median / most common value) and override the inputs
        row = base.drop(columns=["subject", "at_risk", "G3"]).iloc[[0]].copy()
        for c_ in row.columns:
            row[c_] = base[c_].median() if pd.api.types.is_numeric_dtype(base[c_]) else base[c_].mode().iloc[0]
        for k, v in vals.items():
            row[k] = v
        row["avg_alcohol"] = (row["Dalc"] * 5 + row["Walc"] * 2) / 7
        row["parent_edu"] = (row["Medu"] + row["Fedu"]) / 2
        row["grade_trend"] = row["G2"] - row["G1"]
        tmp = pd.concat([base.drop(columns=["subject", "at_risk", "G3"]), row], ignore_index=True)
        tmp["subject"] = "x"; tmp["at_risk"] = 0; tmp["G3"] = 0
        Xall = A.encode(tmp, use_prior)
        xr = Xall.iloc[[-1]][reg_fit["Random Forest"].feature_names_in_]
        pred = float(np.clip(reg_fit["Random Forest"].predict(xr)[0], 0, 20))
        prob = float(clf_fit["Random Forest"].predict_proba(xr[clf_fit["Random Forest"].feature_names_in_])[0, 1])
        m1, m2 = st.columns(2)
        m1.metric("Predicted final grade", f"{pred:.1f} / 20")
        m2.metric("Probability of being at risk", f"{prob*100:.0f}%")
        st.caption("Other, unlisted features are set to the typical (median / most common) value of the dataset.")

    st.subheader("What drives the model?")
    imp = A.feature_importance(reg_fit["Random Forest"], Xc.columns, 12)
    fig, ax = plt.subplots(figsize=(7, 4)); imp[::-1].plot(kind="barh", ax=ax, color="#2E5C8A")
    ax.set_title("Top 12 features (Random Forest)"); st.pyplot(fig)

# ---------------------------------------------------------------- at risk
with tab4:
    st.subheader("Students flagged by the early-warning model")
    st.write("The model below uses only background and behaviour data (no exam grades), "
             "so the list can be produced before the first test.")
    _, _, clf_tbl2, clf_fit2, cms2, Xc2 = train_for(df_all, False)
    rt = A.risk_table(df_all, clf_fit2["Random Forest"], Xc2)
    rt = rt.loc[df.index.intersection(rt.index)] if not df.empty else rt
    lvl = st.multiselect("Risk level", ["High", "Medium", "Low"], default=["High"])
    view = rt[rt.risk_level.isin(lvl)].sort_values("risk_probability", ascending=False)
    cols = ["school", "sex", "age", "studytime", "failures", "absences", "higher", "risk_probability", "risk_level", "G3"]
    st.dataframe(view[cols].round(2))
    st.download_button("Download this list (csv)", view[cols].to_csv(index=False).encode(), "at_risk_students.csv")
    st.caption("G3 is shown only so that you can see how the warning compared with the real outcome. "
               "Note: the model was trained on 80% of these students, so the list looks a little better than it would for brand-new students.")
    st.write("Model performance for 'at risk' detection (test set):")
    st.dataframe(clf_tbl2)

# ---------------------------------------------------------------- about
with tab5:
    st.markdown("""
**About this project**  
Student Performance Analytics Dashboard - B.Tech CSE NTCC project, Amity University Jharkhand, Ranchi.  
**Student:** Palak Singh (A35705225122)  **Guide:** Dr. Vandana Singh  

**Data:** UCI *Student Performance* dataset - 395 Maths and 649 Portuguese-course students from two Portuguese secondary schools; 30 background attributes plus three grades (G1, G2, G3, each 0-20).

**Definitions used**  
- *At risk* = final grade G3 below 10 out of 20.  
- *Early-warning model* = uses no G1/G2 grades.  
- *Mid-term model* = also uses G1 and G2.

**Things to keep in mind**  
- Correlation is not causation.  
- In the Maths data all 38 students with G3 = 0 have 0 absences. Most likely they left or skipped the exam, so absences looks "important" to the model for a misleading reason.  
- Data is from 2005-06 Portugal, so results may not carry over directly to other settings.  
""")
