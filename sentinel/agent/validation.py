"""Numbers are checked in code, not by the LLM: ranges come from the lender's policy, and a bare number
("1", "12k", "8 lakhs") is assigned to the field that was just asked instead of leaving the model to guess.
"""
import re
from typing import Any

from ..tenants import Tenant
from .state import FIELD_LABELS

NUMERIC_FIELDS = ["person_age", "person_income", "person_emp_length", "loan_amnt", "loan_int_rate",
                  "cb_person_cred_hist_length"]

_MULTIPLIERS = {"k": 1e3, "thousand": 1e3, "m": 1e6, "mn": 1e6, "million": 1e6, "lakh": 1e5, "lakhs": 1e5,
                "lac": 1e5, "lacs": 1e5, "crore": 1e7, "crores": 1e7}
_NUMBER = re.compile(
    r"\s*(?:rs\.?|inr|usd|\$|₹)?\s*(-?[0-9][0-9,]*(?:\.[0-9]+)?)\s*"
    r"(k|thousand|mn|m|million|lakhs?|lacs?|crores?|%|percent|years?|yrs?)?\s*\.?\s*", re.I)


def parse_number(text: str) -> float | None:
    """The whole message is one number, optionally with a unit: "12k", "12,000", "11.5%", "8 lakhs"."""
    match = _NUMBER.fullmatch(text)
    if not match:
        return None
    value = float(match.group(1).replace(",", ""))
    return value * _MULTIPLIERS.get((match.group(2) or "").lower(), 1)


def bounds(field: str, tenant: Tenant) -> tuple[float, float]:
    return {
        "person_age": (18, 100),
        "person_income": (1, 100_000_000),
        "person_emp_length": (0, 60),
        "loan_amnt": (tenant.min_loan, tenant.max_loan),
        "loan_int_rate": (1, 40),
        "cb_person_cred_hist_length": (0, 80),
    }[field]


def out_of_range_message(field: str, tenant: Tenant, value: float) -> str:
    lo, hi = bounds(field, tenant)
    if field == "person_income" and value <= 0:
        # Not a typo but an eligibility question: say so kindly instead of quoting a range
        return (f"{tenant.name} needs applicants to have a regular income, such as a salary, self-employment, "
                "a pension or benefits. If you have any income, how much is it per year?")
    if field == "person_income":
        return "That income looks too high to be right. Could you check the amount per year?"
    if field == "person_age" and value < lo:
        return f"You need to be at least {lo:g} to apply. How old are you?"
    if field == "loan_amnt":
        return f"{tenant.name} lends between {lo:,.0f} and {hi:,.0f}. How much would you like to borrow?"
    if field == "loan_int_rate":
        return f"Interest rates are between {lo:g}% and {hi:g}%. What rate were you offered?"
    return f"That doesn't look right: your {FIELD_LABELS[field]} should be between {lo:,g} and {hi:,g}. " \
           "Could you check it?"


def validate(updates: dict[str, Any], tenant: Tenant) -> tuple[dict[str, Any], str | None]:
    """Drop out-of-range numbers; return the cleaned updates and a message about the first bad one."""
    clean, error = dict(updates), None
    for field in NUMERIC_FIELDS:
        if field in clean:
            lo, hi = bounds(field, tenant)
            value = float(clean[field])
            if not lo <= value <= hi:
                del clean[field]
                error = error or out_of_range_message(field, tenant, value)
    return clean, error
