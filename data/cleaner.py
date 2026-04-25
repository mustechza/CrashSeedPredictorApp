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

    # TARGET
    df["crash"] = df["rate"].astype(float)

    # TIME CONVERSION
    df["prepareTime"] = pd.to_datetime(df["prepareTime"], unit="ms", errors="coerce")
    df["beginTime"] = pd.to_datetime(df["beginTime"], unit="ms", errors="coerce")
    df["endTime"] = pd.to_datetime(df["endTime"], unit="ms", errors="coerce")
    df["fetchedAt"] = pd.to_datetime(df["fetchedAt"], errors="coerce")

    # FEATURES
    df["round_duration"] = (df["endTime"] - df["beginTime"]).dt.total_seconds()
    df["prep_gap"] = (df["beginTime"] - df["prepareTime"]).dt.total_seconds()

    df["rolling_mean"] = df["crash"].rolling(10).mean()
    df["rolling_std"] = df["crash"].rolling(10).std()
    df["delta"] = df["crash"].diff()

    # FINAL CLEAN
    df = df.dropna(subset=FEATURES + ["crash"])

    return df
