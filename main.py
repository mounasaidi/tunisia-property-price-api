import numpy as np
import pandas as pd
import os
import joblib

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

from src.data_loader import load_data
from src.preprocessing import feature_engineering, encode_features
from src.training import train_models
from src.evaluation import evaluate_model


# =========================
# 1. LOAD DATA
# =========================
df = load_data("data/mubawab_clean_pro1_80.csv")
print(f"✅ Dataset chargé : {df.shape}")
print(df['type'].value_counts())
print(df['category'].value_counts())

# =========================
# 2. FEATURE ENGINEERING
# =========================
df = feature_engineering(df)
print(f"\n✅ Après feature engineering : {df.shape}")

# =========================
# 3. SPLIT PAR TYPE
# =========================
df_louer  = df[df['type'] == 'À Louer']
df_vendre = df[df['type'] == 'À Vendre']

print(f"\n  À Louer  : {len(df_louer)} lignes")
print(f"  À Vendre : {len(df_vendre)} lignes")


# =========================
# 4. PIPELINE
# =========================
def train_pipeline(data, label):
    print(f"\n{'='*55}")
    print(f"🚀 Training pipeline : {label}")
    print(f"{'='*55}")

    # Split
    train_df, test_df = train_test_split(data, test_size=0.2, random_state=42)

    # Encode
    train_df, test_df, encoder = encode_features(train_df, test_df)

    # Features & target
    X_train = train_df.drop(columns=['price', 'log_price'])
    y_train = train_df['log_price']
    X_test  = test_df.drop(columns=['price', 'log_price'])
    y_test  = test_df['log_price']

    # Scaling
    scaler = StandardScaler()
    feature_names = X_train.columns

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)
    scaler.feature_names_in_ = feature_names

    # Train
    models = train_models(X_train_scaled, y_train)

    # Evaluate
    best_model = None
    best_score = -np.inf

    print(f"\n📊 Résultats :")
    print(f"{'Modèle':<20} {'MAE':>8} {'R2':>8} {'RMSE':>8} {'CV R2':>8}")
    print("-" * 56)

    for name, model in models.items():
        mae, r2, rmse = evaluate_model(model, X_test_scaled, y_test)
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
        cv_mean = cv_scores.mean()
        print(f"{name:<20} {mae:>8.3f} {r2:>8.3f} {rmse:>8.3f} {cv_mean:>8.4f}")

        if r2 > best_score:
            best_score = r2
            best_model = model
            best_name  = name

    print(f"\n🏆 Meilleur modèle : {best_name} (R2 = {best_score:.4f})")

    # Overfitting check
    print(f"\n🔥 Overfitting check :")
    print(f"  Train R2 : {r2_score(y_train, best_model.predict(X_train_scaled)):.4f}")
    print(f"  Test  R2 : {r2_score(y_test,  best_model.predict(X_test_scaled)):.4f}")

    # Save
    os.makedirs("models", exist_ok=True)
    tag = label.replace(' ', '_')

    joblib.dump(best_model, f"models/model_{tag}.pkl")
    joblib.dump(scaler,     f"models/scaler_{tag}.pkl")
    joblib.dump(encoder,    f"models/encoder_{tag}.pkl")

    print(f"\n✅ Modèles sauvegardés dans models/")
    return best_model, scaler, encoder


# =========================
# 5. TRAIN
# =========================
model_louer,  scaler_louer,  encoder_louer  = train_pipeline(df_louer,  "À_Louer")
model_vendre, scaler_vendre, encoder_vendre = train_pipeline(df_vendre, "À_Vendre")


# =========================
# 6. FONCTION DE PRÉDICTION
# =========================
def predict_price(model, scaler, encoder, sample_df):
    sample_df = sample_df.copy()
    categorical_cols = ['category', 'type', 'city', 'region']

    encoded = encoder.transform(sample_df[categorical_cols])
    feature_names = encoder.get_feature_names_out(categorical_cols)
    encoded_df = pd.DataFrame(encoded, columns=feature_names, index=sample_df.index)

    sample_df = sample_df.drop(columns=categorical_cols)
    sample_df = pd.concat([sample_df, encoded_df], axis=1)
    sample_df = sample_df.reindex(columns=scaler.feature_names_in_, fill_value=0)

    X = scaler.transform(sample_df)
    log_pred = model.predict(X)
    return np.exp(log_pred)


# =========================
# 7. TEST
# =========================
print(f"\n{'='*55}")
print("🔥 TEST DE PRÉDICTION")
print(f"{'='*55}")

sample_louer  = df_louer.sample(5, random_state=1)
sample_vendre = df_vendre.sample(5, random_state=1)

pred_louer  = predict_price(model_louer,  scaler_louer,  encoder_louer,  sample_louer)
pred_vendre = predict_price(model_vendre, scaler_vendre, encoder_vendre, sample_vendre)

print("\n📍 À Louer — Prix réels vs prédits (DT/mois) :")
for real, pred in zip(sample_louer['price'].values, pred_louer):
    print(f"  Réel : {real:>10,.0f} DT  |  Prédit : {pred:>10,.0f} DT")

print("\n📍 À Vendre — Prix réels vs prédits (DT) :")
for real, pred in zip(sample_vendre['price'].values, pred_vendre):
    print(f"  Réel : {real:>10,.0f} DT  |  Prédit : {pred:>10,.0f} DT")