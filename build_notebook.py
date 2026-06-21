"""
build_notebook.py

Generates notebooks/exploratory_analysis.ipynb using nbformat.
Run once from the project root:

    python build_notebook.py
    jupyter nbconvert --to notebook --execute --inplace notebooks/exploratory_analysis.ipynb

This script is not part of the portfolio deliverable itself - it's the
tool used to produce the notebook. It's kept in the repo for
transparency/reproducibility.
"""

import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

md = lambda src: cells.append(nbf.v4.new_markdown_cell(src))
code = lambda src: cells.append(nbf.v4.new_code_cell(src))

md(
"""# HR people-analytics exploration — Brightline Retail Group (synthetic)

**Business question:** Brightline Retail Group, a fictional Bangladeshi retail chain, wants to
understand where it is losing employees and why, so HR can focus retention effort
where it actually moves the needle instead of applying blanket policies.

**About the data:** every record in this notebook is synthetically generated
(`data/generate_data.py`, seeded for reproducibility) and does **not** represent
any real company or employee. It was built to mirror realistic HR dynamics —
department-level turnover differences, early-tenure flight risk, performance and
attendance signals — so the analysis techniques below transfer directly to a real
HRIS export.

900 employee records, January 2022 – June 2026, across 7 branches and 8 departments."""
)

code(
"""import os, sys

# Make relative paths work whether this notebook is opened from notebooks/
# in Jupyter, or executed from the project root via nbconvert.
if os.path.basename(os.getcwd()) == "notebooks":
    os.chdir("..")
sys.path.insert(0, "src")

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import analytics as a

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 110

df = a.load_data()
df.shape"""
)

md("## 1. Headcount growth\n\nActive headcount at the end of each month since the company's data starts.")

code(
"""hc = a.headcount_by_month(df)

fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(hc.index, hc.values, color="#1D9E75", linewidth=2)
ax.fill_between(hc.index, hc.values, color="#1D9E75", alpha=0.08)
ax.set_title("Active headcount over time")
ax.set_ylabel("Employees")
ax.set_xlabel("")
plt.tight_layout()
plt.show()"""
)

md(
"""## 2. Overall attrition

Attrition rate here is *lifetime* (share of all 900 employees ever hired who have since left),
not an annualised rate - useful for comparing groups, less useful on its own."""
)

code(
"""overall = a.attrition_rate(df)
print(f"Overall attrition rate: {overall:.1%}")
print(f"Employees who left: {(~df['is_active']).sum()} of {len(df)}")"""
)

md("## 3. Where is attrition concentrated?\n\nBroken out by department and by job level.")

code(
"""dept = a.attrition_rate_by(df, "department")

fig, ax = plt.subplots(figsize=(9, 4.5))
colors = ["#D85A30" if v > overall else "#5DCAA5" for v in dept["attrition_rate"]]
ax.barh(dept.index, dept["attrition_rate"], color=colors)
ax.axvline(overall, color="#444441", linestyle="--", linewidth=1, label=f"Company avg ({overall:.0%})")
ax.set_xlabel("Attrition rate")
ax.set_title("Attrition rate by department")
ax.invert_yaxis()
ax.legend()
plt.tight_layout()
plt.show()
dept"""
)

code(
"""level = a.attrition_rate_by(df, "job_level").reindex(["Entry", "Mid", "Senior", "Manager"])

fig, ax = plt.subplots(figsize=(7, 4))
ax.bar(level.index, level["attrition_rate"], color="#7F77DD")
ax.axhline(overall, color="#444441", linestyle="--", linewidth=1, label=f"Company avg ({overall:.0%})")
ax.set_ylabel("Attrition rate")
ax.set_title("Attrition rate by job level")
ax.legend()
plt.tight_layout()
plt.show()
level"""
)

md(
"""## 4. Early-tenure flight risk

How much of total attrition happens in the first 6 months on the job? If it's high,
that points HR toward fixing onboarding rather than long-term retention programmes."""
)

code(
"""flight_risk = a.early_tenure_flight_risk(df, months_threshold=6)
print(f"Share of leavers who exited within 6 months of joining: {flight_risk:.1%}")

leavers = df[~df["is_active"]]

fig, ax = plt.subplots(figsize=(9, 4))
ax.hist(leavers["tenure_months"], bins=24, color="#D4537E", edgecolor="white")
ax.axvline(6, color="#444441", linestyle="--", linewidth=1, label="6-month mark")
ax.set_xlabel("Tenure at exit (months)")
ax.set_ylabel("Employees who left")
ax.set_title("How long do leavers stay before exiting?")
ax.legend()
plt.tight_layout()
plt.show()"""
)

md("## 5. Performance rating vs. attrition")

code(
"""perf = a.attrition_rate_by(df, "performance_rating").sort_index()

fig, ax = plt.subplots(figsize=(7, 4))
ax.bar(perf.index.astype(str), perf["attrition_rate"], color="#EF9F27")
ax.set_xlabel("Performance rating (1=lowest, 5=highest)")
ax.set_ylabel("Attrition rate")
ax.set_title("Attrition rate by performance rating")
plt.tight_layout()
plt.show()
perf"""
)

md("## 6. Why do people say they're leaving?")

code(
"""reasons = a.exit_reason_breakdown(df)

fig, ax = plt.subplots(figsize=(7, 5))
ax.barh(reasons.index, reasons["share"], color="#378ADD")
ax.set_xlabel("Share of all exits")
ax.set_title("Exit reasons")
ax.invert_yaxis()
plt.tight_layout()
plt.show()
reasons"""
)

md(
"""## 7. Predicting flight risk: logistic regression

Everything so far has been descriptive - it tells us *where* attrition is
concentrated, one variable at a time. A logistic regression lets us ask a
sharper question: **holding everything else constant, which factors actually
move the odds of leaving, and can we score currently-active employees by risk
today?**

**A methodological note, stated up front rather than glossed over:** this is a
*static, point-in-time* classifier. It uses each employee's tenure-to-date as
a feature and treats "did this person ever leave" as a fixed label. Active
employees are technically *right-censored* - we don't yet know if or when
they'll leave, the model just treats them as "hasn't left (yet)". That's a
standard, widely-used first approach to attrition modelling, but the
statistically rigorous way to handle censoring is **survival analysis** (e.g.
Cox proportional hazards) - flagged here as a deliberate scope decision, and
listed again under next steps."""
)

code(
"""import sys
sys.path.insert(0, "src")
import risk_model as rm

result = rm.train_and_evaluate(df)
print(f"ROC-AUC (held-out test set): {result['roc_auc']:.3f}")
print()
print(result["classification_report"])"""
)

md(
"""`class_weight="balanced"` is doing real work here: only ~19% of employees in
the data have left, so an unweighted model could hit 80%+ accuracy by
predicting "stays" for everyone and never flag a single at-risk employee. The
classification report above optimises for catching leavers (recall on the
"Left" class) over raw accuracy - the right trade-off for a retention tool,
where missing a flight risk is more costly than one extra unnecessary
check-in."""
)

code(
"""fpr, tpr = result["roc_curve"]
fig, ax = plt.subplots(figsize=(6, 6))
ax.plot(fpr, tpr, color="#1D9E75", linewidth=2, label=f"AUC = {result['roc_auc']:.3f}")
ax.plot([0, 1], [0, 1], color="#999999", linestyle="--", linewidth=1, label="Random guess")
ax.set_xlabel("False positive rate")
ax.set_ylabel("True positive rate")
ax.set_title("ROC curve - flight risk model")
ax.legend()
plt.tight_layout()
plt.show()"""
)

md(
"""### What's actually driving the predictions?

Coefficients below are on standardised/encoded features, so they're directly
comparable to each other. **Positive = associated with higher attrition risk,
negative = associated with lower risk**, holding the other features fixed."""
)

code(
"""importance = rm.get_feature_importance(result["pipeline"]).head(12)

fig, ax = plt.subplots(figsize=(8, 5.5))
colors = ["#D85A30" if c > 0 else "#1D9E75" for c in importance["coefficient"]]
ax.barh(importance["feature"], importance["coefficient"], color=colors)
ax.axvline(0, color="#444441", linewidth=0.8)
ax.set_xlabel("Coefficient (effect on attrition log-odds)")
ax.set_title("What moves the odds of leaving, holding other factors fixed")
ax.invert_yaxis()
plt.tight_layout()
plt.show()
importance"""
)

md(
"""This is a useful sanity check on the earlier charts, not just a repeat of
them: tenure and leave-balance dominate (consistent with the early-tenure
finding), and department/job-level effects survive even after controlling for
performance and attendance - meaning Sales & Retail Ops genuinely carries
extra structural risk, it isn't just an artefact of who happens to work there.

### Turning this into a retention worklist

The real payoff: fit the same model on the full dataset and score every
*currently active* employee, producing a ranked list HR can actually act on."""
)

code(
"""full_pipe = rm.fit_full_pipeline(df)
at_risk = rm.score_active_employees(df, full_pipe)

print(f"{len(at_risk)} active employees scored.")
print(f"{(at_risk['predicted_risk'] > 0.5).sum()} flagged above a 50% risk threshold.")
at_risk.head(15)"""
)

md(
"""## 8. Key findings & what HR could do next

These read directly off the analysis above - the kind of summary that would go
at the top of a stakeholder-facing report:

1. **Sales & Retail Ops and Warehouse & Logistics carry the highest attrition**,
   both above the company average - and the regression confirms this isn't
   just a side-effect of who staffs those departments; the department effect
   holds after controlling for performance, attendance, and tenure.
2. **Entry-level roles leave more than twice as often as Senior roles.** Attrition
   drops steadily as job level rises, which is a fairly classic pattern but worth
   quantifying rather than assuming.
3. **Most attrition happens early.** A majority of leavers exit within 6 months of
   joining, and tenure-to-date is the single strongest predictor in the model -
   this points toward an onboarding/early-engagement problem more than a
   long-tenure retention one.
4. **Lower performance ratings and lower leave balances both correlate with
   higher attrition risk**, consistent with disengagement showing up in the
   data before someone formally resigns.
5. **"Better opportunity" is the single largest stated exit reason**, which
   usually signals a compensation or career-growth-path gap relative to the
   market rather than a workplace-culture problem specifically.

**Suggested next steps for HR:** prioritise outreach using the at-risk list
above - the highest-leverage move is tightening onboarding and 90-day
check-ins for Sales/Warehouse entry-level hires specifically, since that's
where both the descriptive data and the model agree risk concentrates.

**Suggested next steps for the analysis itself:** replace this point-in-time
classifier with a proper survival model (Cox proportional hazards) to handle
censoring correctly and estimate *when*, not just *whether*, someone is likely
to leave.

---
*This notebook is the analysis layer behind the interactive dashboard in
`app.py`, which surfaces the same at-risk list with live filters - run
`streamlit run app.py` to explore it.*"""
)

nb["cells"] = cells
nbf.write(nb, "notebooks/exploratory_analysis.ipynb")
print("Notebook written.")
