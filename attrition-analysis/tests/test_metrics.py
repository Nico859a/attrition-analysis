import pandas as pd
import pytest
from src.metrics import (
    attrition_rate,
    attrition_by_department,
    attrition_by_overtime,
    average_income_by_attrition,
    satisfaction_summary,
)


def make_df():
    """Six employees across three departments with one leaver."""
    return pd.DataFrame({
        "employee_id": [1, 2, 3, 4, 5, 6],
        "department":  ["Sales", "Sales", "HR", "HR", "IT", "IT"],
        "overtime":    ["Yes", "No", "Yes", "No", "No", "No"],
        "monthly_income": [4000, 6000, 3000, 7000, 5000, 8000],
        "job_satisfaction": [2, 4, 2, 3, 4, 3],
        "attrition":   ["Yes", "No", "No", "No", "No", "No"],
    })


# ---------------------------------------------------------------------------
# attrition_rate
# ---------------------------------------------------------------------------

def test_attrition_rate_returns_expected_percent():
    df = pd.DataFrame({
        "employee_id": [1, 2, 3, 4],
        "department":  ["Sales", "Sales", "HR", "HR"],
        "attrition":   ["Yes", "No", "No", "Yes"],
    })
    assert attrition_rate(df) == 50.0


def test_attrition_rate_all_leave():
    df = pd.DataFrame({"employee_id": [1, 2], "attrition": ["Yes", "Yes"]})
    assert attrition_rate(df) == 100.0


def test_attrition_rate_none_leave():
    df = pd.DataFrame({"employee_id": [1, 2], "attrition": ["No", "No"]})
    assert attrition_rate(df) == 0.0


# ---------------------------------------------------------------------------
# attrition_by_department
# ---------------------------------------------------------------------------

def test_attrition_by_department_returns_expected_columns():
    df = pd.DataFrame({
        "employee_id": [1, 2, 3, 4],
        "department":  ["Sales", "Sales", "HR", "HR"],
        "attrition":   ["Yes", "No", "No", "Yes"],
    })
    result = attrition_by_department(df)
    assert list(result.columns) == ["department", "employees", "leavers", "attrition_rate"]


def test_attrition_by_department_calculates_correct_rates():
    df = make_df()
    result = attrition_by_department(df)

    sales = result[result["department"] == "Sales"].iloc[0]
    assert sales["employees"] == 2
    assert sales["leavers"] == 1
    assert sales["attrition_rate"] == 50.0

    hr = result[result["department"] == "HR"].iloc[0]
    assert hr["employees"] == 2
    assert hr["leavers"] == 0
    assert hr["attrition_rate"] == 0.0


def test_attrition_by_department_sorted_descending():
    df = make_df()
    result = attrition_by_department(df)
    rates = list(result["attrition_rate"])
    assert rates == sorted(rates, reverse=True)


# ---------------------------------------------------------------------------
# attrition_by_overtime
# ---------------------------------------------------------------------------

def test_attrition_by_overtime_returns_expected_columns():
    df = make_df()
    result = attrition_by_overtime(df)
    assert list(result.columns) == ["overtime", "employees", "leavers", "attrition_rate"]


def test_attrition_by_overtime_calculates_correct_rates():
    df = make_df()
    result = attrition_by_overtime(df)

    # overtime=Yes: employees 1 (leaver) and 3 (stayer) → 50%
    yes_row = result[result["overtime"] == "Yes"].iloc[0]
    assert yes_row["employees"] == 2
    assert yes_row["leavers"] == 1
    assert yes_row["attrition_rate"] == 50.0

    # overtime=No: employees 2, 4, 5, 6 — none left → 0%
    no_row = result[result["overtime"] == "No"].iloc[0]
    assert no_row["employees"] == 4
    assert no_row["leavers"] == 0
    assert no_row["attrition_rate"] == 0.0


# ---------------------------------------------------------------------------
# average_income_by_attrition
# ---------------------------------------------------------------------------

def test_average_income_by_attrition_returns_expected_columns():
    df = make_df()
    result = average_income_by_attrition(df)
    assert list(result.columns) == ["attrition", "avg_monthly_income"]


def test_average_income_by_attrition_calculates_correct_averages():
    df = make_df()
    result = average_income_by_attrition(df)

    # Only employee 1 left, income=4000
    yes_row = result[result["attrition"] == "Yes"].iloc[0]
    assert yes_row["avg_monthly_income"] == 4000.0

    # Stayers: 6000 + 3000 + 7000 + 5000 + 8000 = 29000 / 5 = 5800
    no_row = result[result["attrition"] == "No"].iloc[0]
    assert no_row["avg_monthly_income"] == 5800.0


# ---------------------------------------------------------------------------
# satisfaction_summary
# ---------------------------------------------------------------------------

def test_satisfaction_summary_returns_expected_columns():
    df = make_df()
    result = satisfaction_summary(df)
    assert list(result.columns) == ["job_satisfaction", "total_employees", "leavers", "attrition_rate"]


def test_satisfaction_summary_calculates_rate_within_group():
    df = make_df()
    result = satisfaction_summary(df)

    # satisfaction=2: employees 1 (leaver) and 3 (stayer) → 1/2 = 50%
    group_2 = result[result["job_satisfaction"] == 2].iloc[0]
    assert group_2["total_employees"] == 2
    assert group_2["leavers"] == 1
    assert group_2["attrition_rate"] == 50.0

    # satisfaction=4: employees 2 and 5, neither left → 0/2 = 0%
    group_4 = result[result["job_satisfaction"] == 4].iloc[0]
    assert group_4["total_employees"] == 2
    assert group_4["leavers"] == 0
    assert group_4["attrition_rate"] == 0.0


def test_satisfaction_summary_sorted_by_satisfaction():
    df = make_df()
    result = satisfaction_summary(df)
    scores = list(result["job_satisfaction"])
    assert scores == sorted(scores)
