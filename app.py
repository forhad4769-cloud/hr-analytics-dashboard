"""
app.py

Interactive HR analytics dashboard for Brightline Retail Group (synthetic
data - see data/generate_data.py). Run with:

    streamlit run app.py
"""

import os
import sys

import pandas as pd
import plotly.express as px
import streamlit as st

# Anchor to this file's own location, not the current working directory -
# the working directory isn't guaranteed to be the project root when this
# runs on Streamlit Community Cloud, even though it always is when you run
# `streamlit run app.py` from inside the project folder locally.
APP_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(APP_DIR, "src"))
import analytics as a  # noqa: E402
import risk_model as rm  # noqa: E402

st.set_page_config(page_title="Brightline HR Analytics", layout="wide", page_icon="📊")

PRIMARY = "#1D9E75"
WARN = "#D85A30"


@st.cache_data
def get_data():
    return a.load_data()


@st.cache_resource
def get_risk_pipeline(_df):
    """Fits the logistic regression once per session (the underscore tells
    Streamlit not to try hashing the dataframe as a cache key)."""
    return rm.fit_full_pipeline(_df)


df = get_data()

# ---------------------------------------------------------------- Sidebar
st.sidebar.header("Filters")

branches = st.sidebar.multiselect(
    "Branch", sorted(df["branch"].unique()), default=sorted(df["branch"].unique())
)
departments = st.sidebar.multiselect(
    "Department", sorted(df["department"].unique()), default=sorted(df["department"].unique())
)
job_levels = st.sidebar.multiselect(
    "Job level",
    ["Entry", "Mid", "Senior", "Manager"],
    default=["Entry", "Mid", "Senior", "Manager"],
)
status = st.sidebar.radio("Employee status", ["All", "Active only", "Left only"], index=0)

st.sidebar.markdown("---")
st.sidebar.caption(
    "Data is 100% synthetic, generated with `data/generate_data.py` for "
    "portfolio purposes. It does not represent any real company or employee."
)

filtered = df[
    df["branch"].isin(branches)
    & df["department"].isin(departments)
    & df["job_level"].isin(job_levels)
]
if status == "Active only":
    filtered = filtered[filtered["is_active"]]
elif status == "Left only":
    filtered = filtered[~filtered["is_active"]]

# ---------------------------------------------------------------- Header
st.title("Brightline Retail Group — HR analytics")
st.caption(
    "A people-analytics dashboard built on a synthetic dataset modelled after a "
    "multi-branch Bangladeshi retail chain. Use the filters on the left to drill in."
)

if filtered.empty:
    st.warning("No employees match the current filters.")
    st.stop()

# ---------------------------------------------------------------- KPI row
col1, col2, col3, col4 = st.columns(4)
col1.metric("Headcount (filtered)", f"{len(filtered):,}")
col2.metric("Attrition rate", f"{a.attrition_rate(filtered):.1%}")
col3.metric("Avg. tenure", f"{filtered['tenure_months'].mean():.1f} mo")
col4.metric("Avg. attendance", f"{filtered['attendance_rate_pct'].mean():.1f}%")

st.markdown("---")

# ---------------------------------------------------------------- Headcount trend
st.subheader("Headcount over time")
hc = a.headcount_by_month(filtered)
fig_hc = px.area(x=hc.index, y=hc.values, labels={"x": "", "y": "Active headcount"})
fig_hc.update_traces(line_color=PRIMARY, fillcolor="rgba(29,158,117,0.12)")
fig_hc.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=320)
st.plotly_chart(fig_hc, width='stretch')

left_col, right_col = st.columns(2)

# ---------------------------------------------------------------- Attrition by department
with left_col:
    st.subheader("Attrition by department")
    dept = a.attrition_rate_by(filtered, "department").reset_index()
    overall = a.attrition_rate(filtered)
    dept["above_avg"] = dept["attrition_rate"] > overall
    fig_dept = px.bar(
        dept.sort_values("attrition_rate"),
        x="attrition_rate",
        y="department",
        orientation="h",
        color="above_avg",
        color_discrete_map={True: WARN, False: PRIMARY},
        labels={"attrition_rate": "Attrition rate", "department": ""},
    )
    fig_dept.add_vline(x=overall, line_dash="dash", line_color="#444441")
    fig_dept.update_layout(showlegend=False, margin=dict(l=10, r=10, t=10, b=10), height=380)
    fig_dept.update_xaxes(tickformat=".0%")
    st.plotly_chart(fig_dept, width='stretch')

# ---------------------------------------------------------------- Attrition by job level
with right_col:
    st.subheader("Attrition by job level")
    level_order = ["Entry", "Mid", "Senior", "Manager"]
    level = a.attrition_rate_by(filtered, "job_level").reindex(
        [lv for lv in level_order if lv in filtered["job_level"].unique()]
    ).reset_index()
    fig_level = px.bar(
        level,
        x="job_level",
        y="attrition_rate",
        labels={"attrition_rate": "Attrition rate", "job_level": ""},
        color_discrete_sequence=["#7F77DD"],
    )
    fig_level.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=380)
    fig_level.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig_level, width='stretch')

left_col2, right_col2 = st.columns(2)

# ---------------------------------------------------------------- Tenure at exit
with left_col2:
    st.subheader("Tenure at exit")
    leavers = filtered[~filtered["is_active"]]
    if len(leavers) > 0:
        fig_tenure = px.histogram(
            leavers, x="tenure_months", nbins=20, labels={"tenure_months": "Months at company"}
        )
        fig_tenure.add_vline(x=6, line_dash="dash", line_color="#444441", annotation_text="6 mo")
        fig_tenure.update_traces(marker_color="#D4537E")
        fig_tenure.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=340)
        st.plotly_chart(fig_tenure, width='stretch')
        flight_risk = a.early_tenure_flight_risk(filtered, months_threshold=6)
        st.caption(f"{flight_risk:.0%} of leavers in this view exited within 6 months of joining.")
    else:
        st.info("No exits in the current filter selection.")

# ---------------------------------------------------------------- Exit reasons
with right_col2:
    st.subheader("Exit reasons")
    leavers = filtered[~filtered["is_active"]]
    if len(leavers) > 0:
        reasons = leavers["exit_reason"].value_counts().reset_index()
        reasons.columns = ["exit_reason", "count"]
        fig_reasons = px.bar(
            reasons.sort_values("count"),
            x="count",
            y="exit_reason",
            orientation="h",
            labels={"count": "Employees", "exit_reason": ""},
            color_discrete_sequence=["#378ADD"],
        )
        fig_reasons.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=340)
        st.plotly_chart(fig_reasons, width='stretch')
    else:
        st.info("No exits in the current filter selection.")

st.markdown("---")

# ---------------------------------------------------------------- Predicted flight risk
st.subheader("🎯 Predicted flight risk (active employees)")
st.caption(
    "A logistic regression trained on every employee's outcome (left vs. stayed), "
    "applied here to score *currently active* employees. It's a point-in-time "
    "classifier, not a survival model - see the notebook for the full methodology "
    "and its limitations."
)

if status == "Left only":
    st.info("Switch the **Employee status** filter to 'All' or 'Active only' to see flight-risk scores - they only apply to people still with the company.")
else:
    pipe = get_risk_pipeline(df)
    at_risk = rm.score_active_employees(df, pipe)
    at_risk = at_risk[
        at_risk["branch"].isin(branches)
        & at_risk["department"].isin(departments)
        & at_risk["job_level"].isin(job_levels)
    ]

    threshold = st.slider("Flag employees above this predicted risk", 0.0, 1.0, 0.5, 0.05)
    flagged = at_risk[at_risk["predicted_risk"] >= threshold]

    risk_col1, risk_col2 = st.columns([1, 3])
    risk_col1.metric("Flagged as high-risk", f"{len(flagged):,} of {len(at_risk):,}")

    st.dataframe(
        at_risk.head(25),
        width="stretch",
        column_config={
            "predicted_risk": st.column_config.ProgressColumn(
                "Predicted risk", min_value=0.0, max_value=1.0, format="%.0f%%"
            ),
            "tenure_months": st.column_config.NumberColumn("Tenure (mo)", format="%.1f"),
            "attendance_rate_pct": st.column_config.NumberColumn("Attendance %", format="%.1f"),
        },
        hide_index=True,
    )
    st.download_button(
        "Download full at-risk list as CSV",
        at_risk.to_csv(index=False).encode("utf-8"),
        file_name="brightline_flight_risk.csv",
        mime="text/csv",
    )

st.markdown("---")

# ---------------------------------------------------------------- Raw data + download
with st.expander("View filtered employee records"):
    st.dataframe(
        filtered[
            [
                "employee_id", "branch", "department", "job_level", "hire_date",
                "termination_date", "is_active", "performance_rating",
                "attendance_rate_pct", "tenure_months",
            ]
        ],
        width='stretch',
    )
    st.download_button(
        "Download filtered data as CSV",
        filtered.to_csv(index=False).encode("utf-8"),
        file_name="brightline_hr_filtered.csv",
        mime="text/csv",
    )

st.caption(
    "Built with pandas, Plotly and Streamlit. See notebooks/exploratory_analysis.ipynb "
    "for the underlying analysis walkthrough."
)
