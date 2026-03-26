import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

def feature_engineering(df):
    df = df.copy()

    # ========================
    # Clean data
    # ========================
    df = df[df['size'] > 0]
    df = df[df['room_count'] > 0]
    df = df[df['bathroom_count'] > 0]

    # remove outliers
    df = df[df['price'] > 50]
    df = df[df['price'] < 1_000_000]

    # ========================
    # Features
    # ========================
    df['price_per_m2'] = df['price'] / df['size']
    df['room_size_ratio'] = df['room_count'] / df['size']
    df['bathroom_ratio'] = df['bathroom_count'] / df['room_count']
    
    df['size_per_room'] = df['size'] / df['room_count']
    df['bath_per_room'] = df['bathroom_count'] / df['room_count']
    df['room_density'] = df['room_count'] / df['size']

    # log target
    df['log_price'] = np.log(df['price'])

    # clean infinities
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()

    return df


def encode_features(train_df, test_df):
    categorical_cols = ['category', 'type', 'city', 'region']

    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

    encoder.fit(train_df[categorical_cols])

    train_encoded = encoder.transform(train_df[categorical_cols])
    test_encoded = encoder.transform(test_df[categorical_cols])

    # 🔥 récupérer noms des colonnes
    feature_names = encoder.get_feature_names_out(categorical_cols)

    # 🔥 convertir en DataFrame avec noms corrects
    import pandas as pd

    train_encoded_df = pd.DataFrame(train_encoded, columns=feature_names, index=train_df.index)
    test_encoded_df = pd.DataFrame(test_encoded, columns=feature_names, index=test_df.index)

    # 🔥 supprimer colonnes originales
    train_df = train_df.drop(columns=categorical_cols)
    test_df = test_df.drop(columns=categorical_cols)

    # 🔥 concat propre
    train_df = pd.concat([train_df, train_encoded_df], axis=1)
    test_df = pd.concat([test_df, test_encoded_df], axis=1)

    return train_df, test_df, encoder