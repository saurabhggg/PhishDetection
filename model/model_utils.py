import joblib
from pathlib import Path

MODEL_PATH = Path("model/phishing_model.pkl")

def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Model not trained yet.")
    return joblib.load(MODEL_PATH)
