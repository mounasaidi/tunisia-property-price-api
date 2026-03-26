import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

def feature_engineering(df):
    df = df.copy()

    # ========================
    # Filtres de sécurité
    # ========================
    df = df[df['size'] > 0]
    df = df[df['rooms'] > 0]
    df = df[df['bathrooms'] > 0]
    df = df[df['price'] > 1000]

    # ========================
    # Features dérivées
    # ========================
    df['price_per_m2']    = df['price'] / df['size']
    df['room_size_ratio'] = df['rooms'] / df['size']
    df['bathroom_ratio']  = df['bathrooms'] / df['rooms']
    df['size_per_room']   = df['size'] / df['rooms']
    df['bath_per_room']   = df['bathrooms'] / df['rooms']
    df['room_density']    = df['rooms'] / df['size']

    # ========================
    # Log target
    # ========================
    df['log_price'] = np.log(df['price'])

    # Nettoyer infinités / NaN
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()

    return df


def encode_features(train_df, test_df):
    categorical_cols = ['category', 'type', 'city', 'region']

    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    encoder.fit(train_df[categorical_cols])

    train_encoded = encoder.transform(train_df[categorical_cols])
    test_encoded  = encoder.transform(test_df[categorical_cols])

    feature_names = encoder.get_feature_names_out(categorical_cols)

    train_encoded_df = pd.DataFrame(train_encoded, columns=feature_names, index=train_df.index)
    test_encoded_df  = pd.DataFrame(test_encoded,  columns=feature_names, index=test_df.index)

    train_df = train_df.drop(columns=categorical_cols)
    test_df  = test_df.drop(columns=categorical_cols)

    train_df = pd.concat([train_df, train_encoded_df], axis=1)
    test_df  = pd.concat([test_df,  test_encoded_df],  axis=1)

    return train_df, test_df, encoder