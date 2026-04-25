import pandas as pd

def clean_data(df):
    df = df.copy()

    df = df.dropna()

    # Core feature
    df["crash"] = df["crash"].astype(float)

    # Time-based features
    df["round_duration"] = (df["endTime"] - df["beginTime"]).dt.total_seconds()
    df["prep_gap"] = (df["beginTime"] - df["prepareTime"]).dt.total_seconds()

    # Rolling stats
    df["rolling_mean"] = df["crash"].rolling(10).mean()
    df["rolling_std"] = df["crash"].rolling(10).std()

    # Volatility proxy
    df["delta"] = df["crash"].diff()

    df = df.dropna()

    return df
