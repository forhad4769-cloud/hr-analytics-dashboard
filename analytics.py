"""
analytics.py

Reusable HR analytics functions shared by the exploratory notebook
and the Streamlit dashboard (app.py). Keeping the logic here means
both surfaces stay in sync and the calculations are unit-testable.
"""

from datetime import date
import os

import pandas as pd

# Anchor to this file's location (src/), then up one level to the project
# root - works regardless of what the current working directory happens to
# be when the app is launched (notably, on Streamlit Community Cloud).
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(os.path.dirname(_THIS_DIR), "data", "employees.csv")


def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    """Load the employee dataset and parse date columns."""
    df = pd.read_csv(path)
    df["hire_date"] = pd.to_datetime(df["hire_date"])
    df["termination_date"] = pd.to_datetime(df["termination_date"])
    df["is_active"] = df["is_active"].astype(bool)
    return df


def headcount_by_month(df: pd.DataFrame) -> pd.Series:
    """Active headcount at the end of each month across the dataset's date range."""
    start = df["hire_date"].min().to_period("M")
    end = pd.Timestamp(date(2026, 6, 21)).to_period("M")
    months = pd.period_range(start, end, freq="M")

    counts = []
    for m in months:
        month_end = m.to_timestamp(how="end")
        active_then = (df["hire_date"] <= month_end) & (
            df["termination_date"].isna() | (df["termination_date"] > month_end)
        )
        counts.append(active_then.sum())

    return pd.Series(counts, index=months.to_timestamp(), name="headcount")


def attrition_rate(df: pd.DataFrame, group_by: str | None = None) -> pd.Series | float:
    """
    Overall attrition rate, or attrition rate broken out by a grouping
    column (e.g. 'department', 'branch', 'job_level').
    """
    if group_by is None:
        return float((~df["is_active"]).mean())
    return df.groupby(group_by)["is_active"].apply(lambda s: (~s).mean())


def attrition_rate_by(df: pd.DataFrame, group_by: str) -> pd.DataFrame:
    """Cleaner groupby version: headcount, leavers, and attrition rate per group."""
    g = df.groupby(group_by)
    out = pd.DataFrame(
        {
            "headcount": g.size(),
            "leavers": g.apply(lambda x: (~x["is_active"]).sum()),
        }
    )
    out["attrition_rate"] = (out["leavers"] / out["headcount"]).round(3)
    return out.sort_values("attrition_rate", ascending=False)


def tenure_summary(df: pd.DataFrame, group_by: str | None = None) -> pd.DataFrame:
    """Average tenure in months, optionally split by a grouping column."""
    if group_by is None:
        return df["tenure_months"].describe().to_frame("tenure_months")
    return df.groupby(group_by)["tenure_months"].agg(["mean", "median", "count"]).round(1)


def attendance_vs_attrition(df: pd.DataFrame) -> pd.DataFrame:
    """Average attendance rate for employees who stayed vs. who left."""
    return (
        df.groupby("is_active")["attendance_rate_pct"]
        .mean()
        .rename(index={True: "Active", False: "Left"})
        .round(1)
        .to_frame("avg_attendance_rate_pct")
    )


def exit_reason_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    """Count and share of each exit reason among employees who left."""
    left = df[~df["is_active"]]
    counts = left["exit_reason"].value_counts()
    share = (counts / counts.sum()).round(3)
    return pd.DataFrame({"count": counts, "share": share})


def early_tenure_flight_risk(df: pd.DataFrame, months_threshold: int = 6) -> float:
    """Share of leavers who exited within `months_threshold` months of joining."""
    left = df[~df["is_active"]]
    if len(left) == 0:
        return 0.0
    return float((left["tenure_months"] <= months_threshold).mean())
