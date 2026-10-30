"""Credit-risk model: logistic regression with per-feature contributions as explanations.

From github.com/vmlrj02/creditRiskModel-logisticRegression, with one fix: the single-row
one-hot encoding no longer drops every categorical column.
"""
import asyncio
import json
from functools import cache
from pathlib import Path
from typing import Any, Protocol

import httpx
import numpy as np

MODEL_DIR = Path(__file__).resolve().parent / "model"

CATEGORICAL = ["person_home_ownership", "loan_intent", "loan_grade", "cb_person_default_on_file"]

MODEL_FEATURES = [
    "person_age", "person_income", "person_emp_length", "person_home_ownership",
    "loan_amnt", "loan_int_rate", "loan_intent", "loan_grade", "loan_percent_income",
    "cb_person_cred_hist_length", "cb_person_default_on_file",
]


class CreditModel:
    """Standardise, then logistic regression, in numpy. Weights come from the trained scikit-learn
    model via scripts/export_model.py, so the runtime needs neither pandas nor scikit-learn."""

    def __init__(self, model_dir: Path = MODEL_DIR) -> None:
        weights = json.loads((model_dir / "logreg.json").read_text())
        self.version: str = weights["version"]  # recorded on every decision for audit
        self.train_columns: list[str] = weights["columns"]
        self.mean = np.array(weights["mean"])
        self.scale = np.array(weights["scale"])
        self.coef = np.array(weights["coef"])
        self.intercept = float(weights["intercept"])

    def encode(self, data: dict[str, Any]) -> np.ndarray:
        """One-hot encode like the training data: a categorical column such as loan_grade_B is 1 only
        for that value; the baseline level (e.g. grade A) has no column and encodes as all zeros."""
        row = []
        for column in self.train_columns:
            feature = next((c for c in CATEGORICAL if column.startswith(f"{c}_")), None)
            if feature is None:
                row.append(float(data.get(column) or 0))
            else:
                row.append(1.0 if str(data.get(feature)) == column[len(feature) + 1:] else 0.0)
        return np.array(row)

    def predict(self, data: dict[str, Any]) -> dict[str, Any]:
        x_scaled = (self.encode(data) - self.mean) / self.scale
        proba = float(1 / (1 + np.exp(-(x_scaled @ self.coef + self.intercept))))
        prediction = int(proba >= 0.5)
        contributions = self.contributions(data, x_scaled)
        return {
            "prediction": prediction,
            "decision": "declined" if prediction else "approved",
            "probability": round(proba, 2),
            "explanations": {
                "top_risk_factors": {k: v for k, v in list(contributions.items())[::-1][:3] if v > 0},
                "top_strength_factors": {k: v for k, v in list(contributions.items())[:3] if v < 0},
            },
        }

    def contributions(self, data: dict[str, Any], x_scaled: np.ndarray) -> dict[str, float]:
        """Per-feature contribution to the log-odds of default (scaled value x coefficient), sorted.

        One-hot columns are summed back into their original feature and labelled with the applicant's
        own value. Per-column values would credit a grade-B applicant with "loan_grade_D" (meaning
        "not being grade D lowered risk"), which reads as nonsense to people and LLMs alike.
        """
        grouped: dict[str, float] = {}
        for column, value in zip(self.train_columns, x_scaled * self.coef, strict=True):
            feature = next((c for c in CATEGORICAL if column.startswith(f"{c}_")), column)
            grouped[feature] = grouped.get(feature, 0.0) + float(value)
        labelled = {f"{feature}={data.get(feature)}": round(v, 3) for feature, v in grouped.items()}
        return dict(sorted(labelled.items(), key=lambda kv: kv[1]))


@cache
def get_credit_model() -> CreditModel:
    return CreditModel()


class Scorer(Protocol):
    version: str

    async def score(self, application: dict[str, Any]) -> dict[str, Any]: ...


class LocalScorer:
    """Runs the model in-process. CPU-bound, so it runs in a thread to keep the event loop free."""

    def __init__(self, model: CreditModel | None = None) -> None:
        self.model = model or get_credit_model()
        self.version = self.model.version

    async def score(self, application: dict[str, Any]) -> dict[str, Any]:
        features = {k: application.get(k) for k in MODEL_FEATURES}
        return await asyncio.to_thread(self.model.predict, features)


class HttpScorer:
    """Calls the model deployed as its own service (the original creditRiskModel API)."""

    def __init__(self, url: str, timeout_s: float = 10) -> None:
        self.url = url
        self.version = f"remote:{url}"
        self.timeout_s = timeout_s

    async def score(self, application: dict[str, Any]) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=self.timeout_s) as client:
            response = await client.post(self.url, json={"data": {k: application.get(k) for k in MODEL_FEATURES}})
            response.raise_for_status()
            return response.json()
