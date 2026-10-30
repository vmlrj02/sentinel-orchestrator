"""Lenders using Sentinel. Each has its own policy documents and decision thresholds.

Credit policy is code, not prompt: the model estimates risk, these rules turn it into a decision.
Numbers here must match the lender's policy documents in sentinel/knowledge/<tenant_id>/.
"""
from dataclasses import dataclass
from typing import Any, Literal

Decision = Literal["approved", "declined", "referred"]


@dataclass(frozen=True)
class Tenant:
    id: str
    name: str
    approve_below: float        # default probability under this: auto-approve
    decline_above: float        # over this: auto-decline; in between: human review
    auto_limit: float           # loans above this always go to a human
    max_loan_to_income: float   # hard policy limit, regardless of model score
    review_sla: str
    min_loan: float = 1_000
    max_loan: float = 40_000

    def decide(self, probability: float, application: dict[str, Any]) -> tuple[Decision, str]:
        """Policy decision from the model's default probability. Returns (decision, reason)."""
        loan_to_income = application["loan_amnt"] / application["person_income"]
        if loan_to_income > self.max_loan_to_income:
            return "declined", (f"loan is {loan_to_income:.0%} of annual income, above the "
                                f"{self.max_loan_to_income:.0%} policy limit")
        if probability > self.decline_above:
            return "declined", f"estimated default risk {probability:.0%} is above {self.decline_above:.0%}"
        if application["loan_amnt"] > self.auto_limit:
            return "referred", f"loans above {self.auto_limit:,.0f} are always reviewed by an underwriter"
        if probability >= self.approve_below:
            return "referred", (f"estimated default risk {probability:.0%} is in the review band "
                                f"({self.approve_below:.0%}-{self.decline_above:.0%})")
        return "approved", f"estimated default risk {probability:.0%} is below {self.approve_below:.0%}"


def max_approvable_loan(tenant: Tenant, application: dict[str, Any],
                        probability_of: Any, step: float = 500) -> float | None:
    """Largest loan (in `step` increments, up to the requested amount) this lender would auto-approve,
    holding everything else fixed. Lets explanations say exactly what would change the outcome."""
    best = None
    low, high = 1, int(application["loan_amnt"] // step)
    while low <= high:  # risk rises with loan size, so binary search the step count
        mid = (low + high) // 2
        scenario = {**application, "loan_amnt": mid * step,
                    "loan_percent_income": round(mid * step / application["person_income"], 2)}
        if tenant.decide(probability_of(scenario), scenario)[0] == "approved":
            best, low = mid * step, mid + 1
        else:
            high = mid - 1
    return best


TENANTS: dict[str, Tenant] = {
    "aster": Tenant("aster", "Aster Bank", approve_below=0.30, decline_above=0.60, auto_limit=20_000,
                    max_loan_to_income=0.40, review_sla="2 business days", min_loan=1_000, max_loan=40_000),
    "kestrel": Tenant("kestrel", "Kestrel Credit Union", approve_below=0.40, decline_above=0.70, auto_limit=15_000,
                      max_loan_to_income=0.50, review_sla="1 business day", min_loan=500, max_loan=30_000),
}
DEFAULT_TENANT = "aster"


def get_tenant(tenant_id: str | None) -> Tenant:
    return TENANTS.get(tenant_id or DEFAULT_TENANT, TENANTS[DEFAULT_TENANT])
