import pandas as pd

def load_data(path):
    # ✅ lire avec mauvais encodage volontaire
    df = pd.read_csv(path, encoding="latin1")

    # ✅ corriger proprement
    df['type'] = df['type'].str.encode('latin1').str.decode('utf-8', errors='ignore')

    return df