"""Credit-risk scoring API: logistic regression + per-feature contributions.

Vendored from github.com/vmlrj02/creditRiskModel-logisticRegression so Sentinel runs standalone.
"""
import json
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

MODEL_DIR = Path(__file__).resolve().parent / "model"

model = joblib.load(MODEL_DIR / "logistic_credit_model.joblib")
scaler = joblib.load(MODEL_DIR / "scaler.joblib")
train_columns = json.loads((MODEL_DIR / "train_columns.json").read_text())

app = FastAPI(title="Credit Risk Model API")


class CreditData(BaseModel):
    data: dict


def encode(data: dict) -> pd.DataFrame:
    # No drop_first here: on a single row it would drop every categorical column.
    # The reindex drops the baseline levels the model was trained without.
    df_encoded = pd.get_dummies(pd.DataFrame([data]))
    return df_encoded.reindex(columns=train_columns, fill_value=0).astype(float)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(input_data: CreditData):
    X_scaled = scaler.transform(encode(input_data.data))
    prediction = model.predict(X_scaled)[0]
    proba = model.predict_proba(X_scaled)[0, 1]

    # Feature contribution = scaled value * coefficient; positive raises default risk
    impacts = pd.Series(X_scaled[0] * model.coef_[0], index=train_columns).round(3)

    return {
        "prediction": int(prediction),
        "probability": round(float(proba), 2),
        "explanations": {
            "top_risk_factors": impacts.nlargest(2).to_dict(),
            "top_strength_factors": impacts.nsmallest(2).to_dict(),
        },
    }
