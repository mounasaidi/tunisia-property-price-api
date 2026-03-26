import pandas as pd

def load_data(path):
    df = pd.read_csv(path, encoding="utf-8-sig")
    return df