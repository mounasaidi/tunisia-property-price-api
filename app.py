from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
import os

app = FastAPI(
    title="Tunisia Property Price API",
    description="Estimation du prix des propriétés en Tunisie",
    version="1.0.0"
)

# =========================
# LOAD MODELS
# =========================
BASE = os.path.dirname(__file__)

def load_artifacts(label):
    model   = joblib.load(os.path.join(BASE, f"models/model_{label}.pkl"))
    scaler  = joblib.load(os.path.join(BASE, f"models/scaler_{label}.pkl"))
    encoder = joblib.load(os.path.join(BASE, f"models/encoder_{label}.pkl"))
    return model, scaler, encoder

model_louer,  scaler_louer,  encoder_louer  = load_artifacts("À_Louer")
model_vendre, scaler_vendre, encoder_vendre = load_artifacts("À_Vendre")

# =========================
# REQUEST SCHEMA
# =========================
class PropertyInput(BaseModel):
    type: str          # "À Louer" or "À Vendre"
    category: str      # ex: "Appartement"
    city: str          # ex: "Tunis"
    region: str        # ex: "Tunis"
    rooms: float
    bathrooms: float
    size: float        # en m²

# =========================
# PREDICTION FUNCTION
# =========================
def predict_price(model, scaler, encoder, input_data: dict):
    categorical_cols = ['category', 'type', 'city', 'region']

    df = pd.DataFrame([input_data])

    encoded = encoder.transform(df[categorical_cols])
    feature_names = encoder.get_feature_names_out(categorical_cols)
    encoded_df = pd.DataFrame(encoded, columns=feature_names, index=df.index)

    df = df.drop(columns=categorical_cols)
    df = pd.concat([df, encoded_df], axis=1)
    df = df.reindex(columns=scaler.feature_names_in_, fill_value=0)

    X = scaler.transform(df)
    log_pred = model.predict(X)
    return float(np.exp(log_pred)[0])

# =========================
# ENDPOINTS
# =========================
@app.get("/")
def root():
    return {"status": "ok", "message": "Tunisia Property Price API is running 🚀"}

@app.post("/predict")
def predict(data: PropertyInput):
    input_dict = data.dict()
    prop_type = input_dict.get("type")

    try:
        if prop_type == "À Louer":
            price = predict_price(model_louer, scaler_louer, encoder_louer, input_dict)
        elif prop_type == "À Vendre":
            price = predict_price(model_vendre, scaler_vendre, encoder_vendre, input_dict)
        else:
            raise HTTPException(status_code=400, detail="Type must be 'À Louer' or 'À Vendre'")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {
        "type": prop_type,
        "estimated_price": round(price, 2),
        "currency": "TND"
    }