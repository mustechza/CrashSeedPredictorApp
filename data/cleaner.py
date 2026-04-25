import pandas as pd

def clean_data(df):
    df = df.copy()
    
    df = df.dropna()
    df["crash"] = df["crash"].astype(float)

    # Feature engineering
    df["rolling_mean"] = df["crash"].rolling(10).mean()
    df["rolling_std"] = df["crash"].rolling(10).std()

    df = df.dropna()
    return df
