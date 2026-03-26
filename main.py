import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

from src.data_loader import load_data
from src.preprocessing import feature_engineering, encode_features
from src.training import train_models
from src.evaluation import evaluate_model


# =========================
# 1. LOAD DATA
# =========================
df = load_data("data/Property Prices in Tunisia.csv")

# =========================
# 2. FEATURE ENGINEERING
# =========================
df = feature_engineering(df)

print(df['type'].value_counts())


# =========================
# 3. SPLIT DATA
# =========================
df_louer = df[df['type'] == 'À Louer']
df_vendre = df[df['type'] == 'À Vendre']


# =========================
# 4. PIPELINE FUNCTION
# =========================
def train_pipeline(data, label):
    print(f"\n🚀 Training model: {label}")

    # split
    train_df, test_df = train_test_split(data, test_size=0.2, random_state=42)

    # encode
    train_df, test_df, encoder = encode_features(train_df, test_df)

    # features & target
    X_train = train_df.drop(columns=['price', 'log_price'])
    y_train = train_df['log_price']

    X_test = test_df.drop(columns=['price', 'log_price'])
    y_test = test_df['log_price']

    # =========================
    # SCALING (IMPORTANT FIX)
    # =========================
    scaler = StandardScaler()

    # garder les noms de colonnes AVANT transformation
    feature_names = X_train.columns

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # sauvegarder les noms pour la prédiction
    scaler.feature_names_in_ = feature_names

    # =========================
    # TRAIN MODELS
    # =========================
    models = train_models(X_train_scaled, y_train)

    # =========================
    # EVALUATION
    # =========================
    best_model = None
    best_score = -np.inf

    for name, model in models.items():
        mae, r2, rmse = evaluate_model(model, X_test_scaled, y_test)
        print(f"{name} → MAE: {mae:.3f} | R2: {r2:.3f} | RMSE: {rmse:.3f}")
    
        # cross validation
        from sklearn.model_selection import cross_val_score
        scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
        print(f"CV R2 ({name}): {scores.mean():.4f}")

        if r2 > best_score:
            best_score = r2
            best_model = model
    
    print(f"\n🔥 CHECK OVERFITTING - {label}")

    train_pred = best_model.predict(X_train_scaled)
    test_pred = best_model.predict(X_test_scaled)

    print("Train R2:", r2_score(y_train, train_pred))
    print("Test R2:", r2_score(y_test, test_pred))

    import os
    import joblib

# créer dossier models si non existant
    os.makedirs("models", exist_ok=True)

# sauvegarde du meilleur modèle
    model_path = f"models/model_{label.replace(' ', '_')}.pkl"
    joblib.dump(best_model, model_path)

# sauvegarde du scaler
    scaler_path = f"models/scaler_{label.replace(' ', '_')}.pkl"
    joblib.dump(scaler, scaler_path)

# sauvegarde de l'encoder
    encoder_path = f"models/encoder_{label.replace(' ', '_')}.pkl"
    joblib.dump(encoder, encoder_path)

    print(f"✅ Modèle sauvegardé: {model_path}")

    return best_model, scaler, encoder



# =========================
# 5. TRAIN MODELS
# =========================
model_louer, scaler_louer, encoder_louer = train_pipeline(df_louer, "À Louer")
model_vendre, scaler_vendre, encoder_vendre = train_pipeline(df_vendre, "À Vendre")


# =========================
# 6. PREDICTION FUNCTION
# =========================
def predict_price(model, scaler, encoder, sample_df):
    sample_df = sample_df.copy()

    categorical_cols = ['category', 'type', 'city', 'region']

    # encode
    encoded = encoder.transform(sample_df[categorical_cols])
    feature_names = encoder.get_feature_names_out(categorical_cols)

    encoded_df = pd.DataFrame(encoded, columns=feature_names, index=sample_df.index)

    # drop categorical
    sample_df = sample_df.drop(columns=categorical_cols)

    # concat
    sample_df = pd.concat([sample_df, encoded_df], axis=1)

    # align columns (VERY IMPORTANT)
    sample_df = sample_df.reindex(columns=scaler.feature_names_in_, fill_value=0)

    # scale
    X = scaler.transform(sample_df)

    # predict
    log_pred = model.predict(X)

    # inverse log
    return np.exp(log_pred)


# =========================
# 7. TEST PREDICTION
# =========================
sample = df_louer.sample(5)

pred = predict_price(model_louer, scaler_louer, encoder_louer, sample)

print("\n🔥 Predictions (real price):")
print(pred)



