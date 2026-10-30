from sentinel.scoring.model import get_credit_model

APPLICANT = {
    "person_age": 30, "person_income": 60000, "person_emp_length": 5,
    "person_home_ownership": "RENT", "loan_amnt": 10000, "loan_int_rate": 11.5,
    "loan_intent": "EDUCATION", "loan_grade": "B", "loan_percent_income": 0.17,
    "cb_person_cred_hist_length": 6, "cb_person_default_on_file": "N",
}


def test_categoricals_are_encoded_for_a_single_row():
    model = get_credit_model()
    row = dict(zip(model.train_columns, model.encode(APPLICANT), strict=True))
    assert row["person_home_ownership_RENT"] == 1
    assert row["loan_intent_EDUCATION"] == 1
    assert row["loan_grade_B"] == 1
    assert row["cb_person_default_on_file_Y"] == 0


def test_predict_returns_decision_and_explanations():
    result = get_credit_model().predict(APPLICANT)
    assert result["decision"] == ("declined" if result["prediction"] else "approved")
    assert 0 <= result["probability"] <= 1
    assert 1 <= len(result["explanations"]["top_risk_factors"]) <= 3


def test_explanations_name_the_applicants_own_values():
    factors = get_credit_model().predict(APPLICANT)["explanations"]
    named = {**factors["top_risk_factors"], **factors["top_strength_factors"]}
    assert all("=" in name for name in named)
    assert not any("loan_grade_" in name or "_RENT" in name for name in named)  # no raw one-hot columns
    assert all(v > 0 for v in factors["top_risk_factors"].values())
    assert all(v < 0 for v in factors["top_strength_factors"].values())


def test_categoricals_change_the_score():
    risky = {**APPLICANT, "loan_grade": "G", "cb_person_default_on_file": "Y"}
    model = get_credit_model()
    assert model.predict(risky)["probability"] > model.predict(APPLICANT)["probability"]


def test_numpy_model_matches_the_trained_sklearn_model():
    """The runtime uses exported weights; prove it scores exactly like the original pipeline."""
    import json
    import random

    import joblib
    import pandas as pd

    from sentinel.scoring.model import MODEL_DIR

    sk_model = joblib.load(MODEL_DIR / "logistic_credit_model.joblib")
    scaler = joblib.load(MODEL_DIR / "scaler.joblib")
    columns = json.loads((MODEL_DIR / "train_columns.json").read_text())
    model, rng = get_credit_model(), random.Random(7)
    for _ in range(300):
        income, loan = rng.uniform(8_000, 250_000), rng.uniform(500, 35_000)
        applicant = {
            "person_age": rng.randint(18, 80), "person_income": income, "person_emp_length": rng.uniform(0, 30),
            "person_home_ownership": rng.choice(["RENT", "OWN", "MORTGAGE", "OTHER"]), "loan_amnt": loan,
            "loan_int_rate": rng.uniform(5, 25),
            "loan_intent": rng.choice(["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "HOMEIMPROVEMENT",
                                       "DEBTCONSOLIDATION"]),
            "loan_grade": rng.choice("ABCDEFG"), "loan_percent_income": round(loan / income, 2),
            "cb_person_cred_hist_length": rng.randint(0, 30), "cb_person_default_on_file": rng.choice("YN"),
        }
        x = pd.get_dummies(pd.DataFrame([applicant])).reindex(columns=columns, fill_value=0).astype(float)
        expected = sk_model.predict_proba(scaler.transform(x))[0, 1]
        ours = 1 / (1 + 2.718281828459045 ** -(((model.encode(applicant) - model.mean) / model.scale) @ model.coef
                                               + model.intercept))
        assert abs(ours - expected) < 1e-9
