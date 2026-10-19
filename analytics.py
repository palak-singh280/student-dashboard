"""
analytics.py - data loading, cleaning, feature building and model training
for the Student Performance Analytics Dashboard.
Used by both app.py (Streamlit) and run_analysis.py (report figures).
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
                             accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix)

PASS_MARK = 10          # a final grade below 10 (out of 20) is treated as "at risk"
SEED = 42

BINARY_COLS = ["school", "sex", "address", "famsize", "Pstatus", "schoolsup",
               "famsup", "paid", "activities", "nursery", "higher",
               "internet", "romantic"]
NOMINAL_COLS = ["Mjob", "Fjob", "reason", "guardian"]


def load_data(path_or_buffer, subject="Subject"):
    """Read one of the UCI csv files (semicolon separated)."""
    df = pd.read_csv(path_or_buffer, sep=";")
    df.columns = [c.strip() for c in df.columns]
    df = df.drop_duplicates().reset_index(drop=True)
    for c in ["G1", "G2", "G3", "absences"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna().reset_index(drop=True)
    df["subject"] = subject
    df["at_risk"] = (df["G3"] < PASS_MARK).astype(int)
    df["avg_alcohol"] = (df["Dalc"] * 5 + df["Walc"] * 2) / 7
    df["parent_edu"] = (df["Medu"] + df["Fedu"]) / 2
    df["grade_trend"] = df["G2"] - df["G1"]
    return df


def encode(df, use_prior_grades=True):
    """Turn the table into a numeric matrix X and keep the targets separate."""
    work = df.drop(columns=["subject", "at_risk", "G3"]).copy()
    if not use_prior_grades:
        work = work.drop(columns=["G1", "G2", "grade_trend"])
    for c in BINARY_COLS:
        work[c] = pd.factorize(work[c], sort=True)[0]
    work = pd.get_dummies(work, columns=NOMINAL_COLS, drop_first=True)
    return work.astype(float)


def train_regressors(df, use_prior_grades=True):
    X = encode(df, use_prior_grades)
    y = df["G3"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=SEED)
    models = {
        "Linear Regression": make_pipeline(StandardScaler(), LinearRegression()),
        "Random Forest": RandomForestRegressor(n_estimators=300, random_state=SEED,
                                               min_samples_leaf=2, n_jobs=-1),
    }
    rows, fitted, preds = [], {}, {}
    kf = KFold(5, shuffle=True, random_state=SEED)
    for name, mdl in models.items():
        mdl.fit(Xtr, ytr)
        p = mdl.predict(Xte)
        p = np.clip(p, 0, 20)
        cv = cross_val_score(mdl, X, y, cv=kf, scoring="r2").mean()
        rows.append({"Model": name,
                     "MAE": mean_absolute_error(yte, p),
                     "RMSE": mean_squared_error(yte, p) ** 0.5,
                     "R2": r2_score(yte, p),
                     "CV R2 (5-fold)": cv})
        fitted[name] = mdl
        preds[name] = (yte.values, p)
    return pd.DataFrame(rows).round(3), fitted, preds, X, (Xtr, Xte, ytr, yte)


def train_classifiers(df, use_prior_grades=True):
    X = encode(df, use_prior_grades)
    y = df["at_risk"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=SEED,
                                          stratify=y)
    models = {
        "Logistic Regression": make_pipeline(StandardScaler(),
                                             LogisticRegression(max_iter=2000)),
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=SEED,
                                                min_samples_leaf=2,
                                                class_weight="balanced", n_jobs=-1),
    }
    rows, fitted, cms = [], {}, {}
    for name, mdl in models.items():
        mdl.fit(Xtr, ytr)
        p = mdl.predict(Xte)
        rows.append({"Model": name,
                     "Accuracy": accuracy_score(yte, p),
                     "Precision": precision_score(yte, p, zero_division=0),
                     "Recall": recall_score(yte, p, zero_division=0),
                     "F1": f1_score(yte, p, zero_division=0)})
        fitted[name] = mdl
        cms[name] = confusion_matrix(yte, p)
    return pd.DataFrame(rows).round(3), fitted, cms, X, (Xtr, Xte, ytr, yte)


def feature_importance(model, columns, top=12):
    imp = pd.Series(model.feature_importances_, index=columns)
    return imp.sort_values(ascending=False).head(top)


def risk_table(df, clf, X):
    """Probability of being at risk for every student."""
    out = df.copy()
    out["risk_probability"] = clf.predict_proba(X)[:, 1]
    out["risk_level"] = pd.cut(out["risk_probability"], [-0.01, 0.33, 0.66, 1.0],
                               labels=["Low", "Medium", "High"])
    return out
