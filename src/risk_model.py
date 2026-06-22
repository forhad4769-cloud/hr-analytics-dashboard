"""
risk_model.py

A logistic regression "flight risk" model: given an employee's profile,
predict the probability they end up leaving. Trained on the same synthetic
dataset used elsewhere in this project.

Important framing (see notebook section 8 for the full discussion):
this is a *static, point-in-time* classifier - it treats "did this person
ever leave" as a fixed label, using each employee's tenure-to-date as a
feature. Active employees are technically right-censored (we don't yet
know if/when they'll leave) - a logistic classifier ignores that and just
treats them as "hasn't left (yet)". That's a standard, widely-used
simplification for a first attrition model, but the statistically correct
way to handle the censoring is survival analysis (e.g. Cox proportional
hazards). Noted here, and in the notebook, on purpose rather than glossed
over.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC_FEATURES = [
    "age_at_hire",
    "monthly_salary_bdt",
    "performance_rating",
    "attendance_rate_pct",
    "leave_balance_days",
    "tenure_months",
]
CATEGORICAL_FEATURES = ["branch", "department", "job_level", "gender", "hire_source"]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "left"


def get_feature_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    X = df[ALL_FEATURES].copy()
    y = (~df["is_active"]).astype(int).rename(TARGET)
    return X, y


def build_pipeline() -> Pipeline:
    preprocess = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )
    model = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)
    return Pipeline([("preprocess", preprocess), ("model", model)])


def train_and_evaluate(df: pd.DataFrame, test_size: float = 0.25, random_state: int = 42) -> dict:
    """
    Trains on a stratified train/test split and returns the fitted pipeline
    plus evaluation artifacts. Intended for the notebook walkthrough.
    """
    X, y = get_feature_target(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    pipe = build_pipeline()
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]

    fpr, tpr, _ = roc_curve(y_test, y_proba)

    return {
        "pipeline": pipe,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "y_pred": y_pred,
        "y_proba": y_proba,
        "roc_auc": roc_auc_score(y_test, y_proba),
        "roc_curve": (fpr, tpr),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "classification_report": classification_report(y_test, y_pred, target_names=["Stayed", "Left"]),
    }


def fit_full_pipeline(df: pd.DataFrame) -> Pipeline:
    """Fits on the *entire* dataset - used to score currently-active employees
    for the dashboard, where the goal is the best possible risk estimate
    rather than held-out evaluation (that evaluation happens separately,
    in train_and_evaluate, on a proper test split)."""
    X, y = get_feature_target(df)
    pipe = build_pipeline()
    pipe.fit(X, y)
    return pipe


def get_feature_importance(pipe: Pipeline) -> pd.DataFrame:
    """Logistic regression coefficients, mapped back to human-readable
    feature names, sorted by magnitude. Positive coefficient = associated
    with higher attrition risk; negative = associated with lower risk."""
    feature_names = pipe.named_steps["preprocess"].get_feature_names_out()
    feature_names = [f.replace("num__", "").replace("cat__", "") for f in feature_names]
    coefs = pipe.named_steps["model"].coef_[0]
    out = pd.DataFrame({"feature": feature_names, "coefficient": coefs})
    out["abs_coefficient"] = out["coefficient"].abs()
    return out.sort_values("abs_coefficient", ascending=False).drop(columns="abs_coefficient")


def score_active_employees(df: pd.DataFrame, pipe: Pipeline) -> pd.DataFrame:
    """Scores every currently-active employee with a predicted attrition
    probability, using a pipeline already fit on the full dataset."""
    active = df[df["is_active"]].copy()
    X_active, _ = get_feature_target(active)
    active["predicted_risk"] = pipe.predict_proba(X_active)[:, 1]
    cols = [
        "employee_id", "branch", "department", "job_level", "manager_id",
        "performance_rating", "attendance_rate_pct", "tenure_months", "predicted_risk",
    ]
    return active[cols].sort_values("predicted_risk", ascending=False).reset_index(drop=True)
