import joblib

def save_model(model, scaler, encoders):
    joblib.dump(model, "models/model.pkl")
    joblib.dump(scaler, "models/scaler.pkl")
    joblib.dump(encoders, "models/encoders.pkl")


def load_model():
    model = joblib.load("models/model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    encoders = joblib.load("models/encoders.pkl")

    return model, scaler, encoders