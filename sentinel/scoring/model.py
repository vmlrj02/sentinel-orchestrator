"""Credit-risk model: logistic regression with per-feature contributions as explanations.

From github.com/vmlrj02/creditRiskModel-logisticRegression, with one fix: the single-row
one-hot encoding no longer drops every categorical column.
"""
import asyncio
import hashlib
import json
from functools import cache
from pathlib import Path
from typing import Any, Protocol

import httpx
import joblib
import pandas as pd

MODEL_DIR = Path(__file__).resolve().parent / "model"

CATEGORICAL = ["person_home_ownership", "loan_intent", "loan_grade", "cb_person_default_on_file"]

MODEL_FEATURES = [
    "person_age", "person_income", "person_emp_length", "person_home_ownership",
    "loan_amnt", "loan_int_rate", "loan_intent", "loan_grade", "loan_percent_income",
    "cb_person_cred_hist_length", "cb_person_default_on_file",
]


class CreditModel:
    def __init__(self, model_dir: Path = MODEL_DIR) -> None:
        self.model = joblib.load(model_dir / "logistic_credit_model.joblib")
        self.scaler = joblib.load(model_dir / "scaler.joblib")
        self.train_columns: list[str] = json.loads((model_dir / "train_columns.json").read_text())
        # Recorded on every decision, so an audit can tell which model produced it
        digest = hashlib.sha256()
        for name in ("logistic_credit_model.joblib", "scaler.joblib", "train_columns.json"):
            digest.update((model_dir / name).read_bytes())
        self.version = f"logreg-{digest.hexdigest()[:10]}"

    def encode(self, data: dict[str, Any]) -> pd.DataFrame:
        # No drop_first: on a single row it would drop every categorical column.
        # The reindex drops the baseline levels the model was trained without.
        encoded = pd.get_dummies(pd.DataFrame([data]))
        return encoded.reindex(columns=self.train_columns, fill_value=0).astype(float)

    def predict(self, data: dict[str, Any]) -> dict[str, Any]:
        x_scaled = self.scaler.transform(self.encode(data))
        prediction = int(self.model.predict(x_scaled)[0])
        proba = float(self.model.predict_proba(x_scaled)[0, 1])
        contributions = self.contributions(data, x_scaled[0])
        return {
            "prediction": prediction,
            "decision": "declined" if prediction else "approved",
            "probability": round(proba, 2),
            "explanations": {
                "top_risk_factors": {k: v for k, v in list(contributions.items())[::-1][:3] if v > 0},
                "top_strength_factors": {k: v for k, v in list(contributions.items())[:3] if v < 0},
            },
        }

    def contributions(self, data: dict[str, Any], x_scaled: Any) -> dict[str, float]:
        """Per-feature contribution to the log-odds of default (scaled value x coefficient), sorted.

        One-hot columns are summed back into their original feature and labelled with the applicant's
        own value. Per-column values would credit a grade-B applicant with "loan_grade_D" (meaning
        "not being grade D lowered risk"), which reads as nonsense to people and LLMs alike.
        """
        impacts = pd.Series(x_scaled * self.model.coef_[0], index=self.train_columns)
        grouped: dict[str, float] = {}
        for column, value in impacts.items():
            feature = next((c for c in CATEGORICAL if str(column).startswith(f"{c}_")), str(column))
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
