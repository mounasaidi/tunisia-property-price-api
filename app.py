from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
import os

import sys
sys.path.append(os.path.dirname(__file__))
from src.preprocessing import compute_features

app = FastAPI(
    title="Tunisia Property Price API",
    description="Estimation du prix des propriétés en Tunisie",
    version="2.0.0"
)

# =========================
# CHARGEMENT DES MODELES
# =========================
BASE = os.path.dirname(__file__)

def load_artifacts(label: str):
    model   = joblib.load(os.path.join(BASE, f"models/model_{label}.pkl"))
    scaler  = joblib.load(os.path.join(BASE, f"models/scaler_{label}.pkl"))
    encoder = joblib.load(os.path.join(BASE, f"models/encoder_{label}.pkl"))
    return model, scaler, encoder

model_louer,  scaler_louer,  encoder_louer  = load_artifacts("louer")
model_vendre, scaler_vendre, encoder_vendre = load_artifacts("vendre")

# =========================
# SCHEMA DE REQUETE
# =========================
class PropertyInput(BaseModel):
    type      : str
    category  : str
    city      : str
    region    : str
    rooms     : float
    bathrooms : float
    size      : float

# =========================
# FORMATAGE DU PRIX
# =========================
def format_price(price: float) -> str:
    """Formate le prix avec espace comme séparateur de milliers. Ex: 950 000 DT"""
    return f"{int(round(price)):,}".replace(",", " ") + " DT"

# =========================
# PREDICTION
# =========================
def predict_price(model, scaler, encoder, input_data: dict) -> float:
    categorical_cols = ['category', 'type', 'city', 'region']

    features = compute_features(
        size      = input_data["size"],
        rooms     = input_data["rooms"],
        bathrooms = input_data["bathrooms"]
    )

    for col in categorical_cols:
        features[col] = input_data[col]

    df = pd.DataFrame([features])

    encoded = encoder.transform(df[categorical_cols])
    feature_names = encoder.get_feature_names_out(categorical_cols)
    encoded_df = pd.DataFrame(encoded, columns=feature_names, index=df.index)

    df = df.drop(columns=categorical_cols)
    df = pd.concat([df, encoded_df], axis=1)
    df = df.reindex(columns=scaler.feature_names_in_, fill_value=0)

    X = scaler.transform(df)
    log_pred = model.predict(X)[0]

    return float(np.exp(log_pred))

# =========================
# ENDPOINTS
# =========================
@app.get("/")
def root():
    return {
        "status" : "ok",
        "message": "Tunisia Property Price API is running 🚀",
        "version": "2.0.0"
    }

@app.post("/predict")
def predict(data: PropertyInput):
    input_dict = data.dict()
    prop_type  = input_dict["type"].strip().lower()
    input_dict["type"] = prop_type

    # Normalisation de la casse — "la marsa" = "La Marsa" = "LA MARSA"
    for col in ["category", "city", "region"]:
        input_dict[col] = input_dict[col].strip().title()

    if input_dict["size"] <= 0:
        raise HTTPException(status_code=400, detail="size doit être > 0")
    if input_dict["rooms"] <= 0:
        raise HTTPException(status_code=400, detail="rooms doit être > 0")
    if input_dict["bathrooms"] <= 0:
        raise HTTPException(status_code=400, detail="bathrooms doit être > 0")

    try:
        if prop_type == "louer":
            price = predict_price(model_louer, scaler_louer, encoder_louer, input_dict)
        elif prop_type == "vendre":
            price = predict_price(model_vendre, scaler_vendre, encoder_vendre, input_dict)
        else:
            raise HTTPException(status_code=400, detail="type doit être 'louer' ou 'vendre'")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur prédiction : {str(e)}")

    return {
        "type"                   : prop_type,
        "category"               : input_dict["category"],
        "city"                   : input_dict["city"],
        "estimated_price"        : round(price, 2),          # valeur numérique pour Salesforce
        "estimated_price_format" : format_price(price),      # valeur lisible "950 000 DT"
        "currency"               : "TND"
    }