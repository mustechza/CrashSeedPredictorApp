import pandas as pd

FEATURES = [
    "rolling_mean",
    "rolling_std",
    "round_duration",
    "prep_gap",
    "delta"
]

def clean_data(df):
    df = df.copy()

    df = df.dropna(subset=["crash"])

    # Time features
    df["round_duration"] = (df["endTime"] - df["beginTime"]).dt.total_seconds()
    df["prep_gap"] = (df["beginTime"] - df["prepareTime"]).dt.total_seconds()

    # Rolling features
    df["rolling_mean"] = df["crash"].rolling(10).mean()
    df["rolling_std"] = df["crash"].rolling(10).std()

    # Momentum
    df["delta"] = df["crash"].diff()

    # Drop NaNs ONLY after feature creation
    df = df.dropna(subset=FEATURES)

    return df
