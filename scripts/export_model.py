"""Export the trained scikit-learn model to plain JSON weights for numpy inference.

The joblib files stay in the repo as the source of truth; the service loads logreg.json, which keeps
pandas, scipy and scikit-learn out of the runtime image (smaller image, faster Lambda cold starts).
tests/test_scoring.py checks the two give identical results.

    uv run python scripts/export_model.py
"""
import hashlib
import json
from pathlib import Path

import joblib

MODEL_DIR = Path(__file__).resolve().parents[1] / "sentinel" / "scoring" / "model"

model = joblib.load(MODEL_DIR / "logistic_credit_model.joblib")
scaler = joblib.load(MODEL_DIR / "scaler.joblib")
columns = json.loads((MODEL_DIR / "train_columns.json").read_text())
digest = hashlib.sha256()
for name in ("logistic_credit_model.joblib", "scaler.joblib", "train_columns.json"):
    digest.update((MODEL_DIR / name).read_bytes())

weights = {
    "version": f"logreg-{digest.hexdigest()[:10]}",
    "columns": columns,
    "mean": scaler.mean_.tolist(),
    "scale": scaler.scale_.tolist(),
    "coef": model.coef_[0].tolist(),
    "intercept": float(model.intercept_[0]),
}
(MODEL_DIR / "logreg.json").write_text(json.dumps(weights, indent=1))
print(f"Wrote logreg.json ({weights['version']}, {len(columns)} features)")
