from fastapi.testclient import TestClient

from scoring.app import app, encode

client = TestClient(app)

APPLICANT = {
    "person_age": 30, "person_income": 60000, "person_emp_length": 5,
    "person_home_ownership": "RENT", "loan_amnt": 10000, "loan_int_rate": 11.5,
    "loan_intent": "EDUCATION", "loan_grade": "B", "loan_percent_income": 0.17,
    "cb_person_cred_hist_length": 6, "cb_person_default_on_file": "N",
}


def test_categoricals_are_encoded_for_a_single_row():
    row = encode(APPLICANT).iloc[0]
    assert row["person_home_ownership_RENT"] == 1
    assert row["loan_intent_EDUCATION"] == 1
    assert row["loan_grade_B"] == 1
    assert row["cb_person_default_on_file_Y"] == 0


def test_predict_returns_decision_and_explanations():
    body = client.post("/predict", json={"data": APPLICANT}).json()
    assert body["prediction"] in (0, 1)
    assert 0 <= body["probability"] <= 1
    assert len(body["explanations"]["top_risk_factors"]) == 2


def test_categoricals_change_the_score():
    risky = {**APPLICANT, "loan_grade": "G", "cb_person_default_on_file": "Y"}
    p_safe = client.post("/predict", json={"data": APPLICANT}).json()["probability"]
    p_risky = client.post("/predict", json={"data": risky}).json()["probability"]
    assert p_risky > p_safe
