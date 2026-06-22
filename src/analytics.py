"""
analytics.py

Reusable HR analytics functions shared by the exploratory notebook
and the Streamlit dashboard (app.py). The employee dataset is generated
fully in memory - no CSV file or folder structure required.
"""

import random
from datetime import date, timedelta

import numpy as np
import pandas as pd

# ------------------------------------------------------------------ #
#  Synthetic data generation (inline - no external file dependency)   #
# ------------------------------------------------------------------ #

SEED = 42
TODAY = date(2026, 6, 21)
COMPANY_START = date(2022, 1, 1)
N_EMPLOYEES = 900

BRANCHES = {"Dhaka - Mirpur": 0.22, "Dhaka - Gulshan": 0.16, "Chattogram": 0.18,
            "Sylhet": 0.12, "Khulna": 0.14, "Rajshahi": 0.10, "Head Office": 0.08}
DEPARTMENTS = {"Sales & Retail Ops": 0.32, "Warehouse & Logistics": 0.18,
               "Merchandising": 0.12, "HR & Admin": 0.08, "Finance & Accounts": 0.08,
               "IT & Systems": 0.06, "Quality Control": 0.08, "Marketing": 0.08}
JOB_LEVELS = {"Entry": (0.45, (16000, 26000)), "Mid": (0.32, (26000, 42000)),
              "Senior": (0.16, (42000, 68000)), "Manager": (0.07, (68000, 130000))}
JOB_LEVEL_WEIGHTS = {k: v[0] for k, v in JOB_LEVELS.items()}
HIRE_SOURCES = {"Walk-in": 0.30, "Online application": 0.28, "Employee referral": 0.24,
                "Agency": 0.12, "Campus hiring": 0.06}
EXIT_REASONS = {"Resignation - better opportunity": 0.42, "Resignation - personal reasons": 0.20,
                "Performance-related": 0.14, "Contract end": 0.12, "Relocation": 0.08, "Other": 0.04}

def _wc(d):
    return random.choices(list(d.keys()), weights=list(d.values()), k=1)[0]

def _make_employee(emp_id, rng):
    branch = _wc(BRANCHES)
    department = _wc(DEPARTMENTS)
    job_level = _wc(JOB_LEVEL_WEIGHTS)
    sal_low, sal_high = JOB_LEVELS[job_level][1]
    monthly_salary = int(rng.integers(sal_low, sal_high))
    gender = random.choices(["Male", "Female"], weights=[0.62, 0.38], k=1)[0]
    age_at_hire = int(np.clip(rng.normal(27, 5.5), 19, 55))
    days_span = (TODAY - COMPANY_START).days
    offset = int((random.random() ** 0.7) * days_span)
    hire_date = COMPANY_START + timedelta(days=offset)
    hire_source = _wc(HIRE_SOURCES)
    performance_rating = int(np.clip(rng.normal(3.2, 0.8), 1, 5))
    attendance_rate = float(np.clip(rng.normal(93, 6), 55, 100))

    risk = 0.10
    if department in ("Sales & Retail Ops", "Warehouse & Logistics"): risk += 0.10
    if job_level == "Entry": risk += 0.08
    if performance_rating <= 2: risk += 0.18
    if attendance_rate < 85: risk += 0.15
    if (TODAY - hire_date).days < 180: risk += 0.10
    risk = min(risk, 0.85)

    termination_date = pd.NaT
    exit_reason = None
    if random.random() < risk:
        max_tenure = (TODAY - hire_date).days
        if max_tenure > 30:
            mean_tenure = 260 if performance_rating <= 2 else 420
            tenure = max(int(rng.exponential(mean_tenure)), 14)
            if tenure <= max_tenure:
                termination_date = hire_date + timedelta(days=tenure)
                exit_reason = _wc(EXIT_REASONS)

    is_active = pd.isna(termination_date)
    leave_balance = float(np.clip(rng.normal(10 if is_active else 4, 4), 0, 24))
    manager_id = f"EMP{random.randint(1, max(1, emp_id - 1)):04d}" if emp_id > 20 else "EMP0001"

    return {
        "employee_id": f"EMP{emp_id:04d}", "gender": gender, "age_at_hire": age_at_hire,
        "branch": branch, "department": department, "job_level": job_level,
        "monthly_salary_bdt": monthly_salary, "hire_source": hire_source,
        "hire_date": hire_date, "termination_date": termination_date,
        "exit_reason": exit_reason, "is_active": is_active,
        "performance_rating": performance_rating,
        "attendance_rate_pct": round(attendance_rate, 1),
        "leave_balance_days": round(leave_balance, 1), "manager_id": manager_id,
    }

def _generate_df() -> pd.DataFrame:
    random.seed(SEED)
    rng = np.random.default_rng(SEED)
    rows = [_make_employee(i, rng) for i in range(1, N_EMPLOYEES + 1)]
    df = pd.DataFrame(rows)
    df["hire_date"] = pd.to_datetime(df["hire_date"])
    df["termination_date"] = pd.to_datetime(df["termination_date"])
    df["tenure_days"] = (
        df["termination_date"].fillna(pd.Timestamp(TODAY)) - df["hire_date"]
    ).dt.days
    df["tenure_months"] = (df["tenure_days"] / 30.44).round(1)
    return df

# ------------------------------------------------------------------ #
#  Public API                                                          #
# ------------------------------------------------------------------ #

def load_data() -> pd.DataFrame:
    """Generate the synthetic employee dataset in memory and return it."""
    return _generate_df()


def headcount_by_month(df: pd.DataFrame) -> pd.Series:
    start = df["hire_date"].min().to_period("M")
    end = pd.Timestamp(TODAY).to_period("M")
    months = pd.period_range(start, end, freq="M")
    counts = []
    for m in months:
        month_end = m.to_timestamp(how="end")
        active_then = (df["hire_date"] <= month_end) & (
            df["termination_date"].isna() | (df["termination_date"] > month_end)
        )
        counts.append(active_then.sum())
    return pd.Series(counts, index=months.to_timestamp(), name="headcount")


def attrition_rate(df: pd.DataFrame, group_by=None):
    if group_by is None:
        return float((~df["is_active"]).mean())
    return df.groupby(group_by)["is_active"].apply(lambda s: (~s).mean())


def attrition_rate_by(df: pd.DataFrame, group_by: str) -> pd.DataFrame:
    g = df.groupby(group_by)
    out = pd.DataFrame({
        "headcount": g.size(),
        "leavers": g.apply(lambda x: (~x["is_active"]).sum()),
    })
    out["attrition_rate"] = (out["leavers"] / out["headcount"]).round(3)
    return out.sort_values("attrition_rate", ascending=False)


def tenure_summary(df: pd.DataFrame, group_by=None) -> pd.DataFrame:
    if group_by is None:
        return df["tenure_months"].describe().to_frame("tenure_months")
    return df.groupby(group_by)["tenure_months"].agg(["mean", "median", "count"]).round(1)


def attendance_vs_attrition(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("is_active")["attendance_rate_pct"]
        .mean()
        .rename(index={True: "Active", False: "Left"})
        .round(1)
        .to_frame("avg_attendance_rate_pct")
    )


def exit_reason_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    left = df[~df["is_active"]]
    counts = left["exit_reason"].value_counts()
    share = (counts / counts.sum()).round(3)
    return pd.DataFrame({"count": counts, "share": share})


def early_tenure_flight_risk(df: pd.DataFrame, months_threshold: int = 6) -> float:
    left = df[~df["is_active"]]
    if len(left) == 0:
        return 0.0
    return float((left["tenure_months"] <= months_threshold).mean())
