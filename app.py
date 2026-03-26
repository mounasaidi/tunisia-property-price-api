from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import numpy as np
import pandas as pd
import os

# =========================
# INIT API
# =========================
app = FastAPI(
    title="Tunisia Property Price API",
    description="Estimation du prix des propriétés en Tunisie",
    version="1.0.0"
)

# =========================
# LOAD MODELS
# =========================
BASE = os.path.dirname(__file__)

def load_artifacts(label: str):
    """
    Charge le modèle, scaler et encoder pour un type spécifique.
    """
    model   = joblib.load(os.path.join(BASE, f"models/model_{label}.pkl"))
    scaler  = joblib.load(os.path.join(BASE, f"models/scaler_{label}.pkl"))
    encoder = joblib.load(os.path.join(BASE, f"models/encoder_{label}.pkl"))
    return model, scaler, encoder

# ⚠️ Éviter les caractères spéciaux pour Linux / Render
model_louer,  scaler_louer,  encoder_louer  = load_artifacts("louer")
model_vendre, scaler_vendre, encoder_vendre = load_artifacts("vendre")

# =========================
# REQUEST SCHEMA
# =========================
class PropertyInput(BaseModel):
    type: str = Field(..., description="Type de transaction: 'À Louer' ou 'À Vendre'")
    category: str = Field(..., description="Catégorie: ex 'Appartement', 'Villa' ...")
    city: str = Field(..., description="Ville: ex 'Tunis'")
    region: str = Field(..., description="Région: ex 'Tunis'")
    rooms: float = Field(..., gt=0)
    bathrooms: float = Field(..., ge=0)
    size: float = Field(..., gt=0, description="Surface en m²")

# =========================
# PREDICTION FUNCTION
# =========================
def predict_price(model, scaler, encoder, input_data: dict):
    categorical_cols = ['category', 'type', 'city', 'region']

    df = pd.DataFrame([input_data])

    # ⚡ Encode categorical features, ignore unknowns
    encoded = encoder.transform(df[categorical_cols])
    feature_names = encoder.get_feature_names_out(categorical_cols)
    encoded_df = pd.DataFrame(encoded, columns=feature_names, index=df.index)

    # Remove categorical columns and concat encoded
    df = df.drop(columns=categorical_cols)
    df = pd.concat([df, encoded_df], axis=1)

    # Reindex pour correspondre au scaler
    df = df.reindex(columns=scaler.feature_names_in_, fill_value=0)

    # Scale features
    X = scaler.transform(df)

    # Prédiction log(price) → convertit en prix réel
    log_pred = model.predict(X)
    return float(np.exp(log_pred)[0])

# =========================
# ENDPOINTS
# =========================
@app.get("/")
def root():
    return {"status": "ok", "message": "Tunisia Property Price API is running 🚀"}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/predict")
def predict(data: PropertyInput):
    input_dict = data.model_dump()  # Pydantic v2 compatible
    prop_type = input_dict.get("type")

    try:
        if prop_type == "À Louer":
            price = predict_price(model_louer, scaler_louer, encoder_louer, input_dict)
        elif prop_type == "À Vendre":
            price = predict_price(model_vendre, scaler_vendre, encoder_vendre, input_dict)
        else:
            raise HTTPException(status_code=400, detail="Type must be 'À Louer' or 'À Vendre'")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

    return {
        "type": prop_type,
        "estimated_price": round(price, 2),
        "currency": "TND"
    }