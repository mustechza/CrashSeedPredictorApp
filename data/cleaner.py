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

    # ----------------------------
    # TARGET
    # ----------------------------
    df["crash"] = pd.to_numeric(df["rate"], errors="coerce")

    # ----------------------------
    # TIME PARSING
    # ----------------------------
    df["prepareTime"] = pd.to_datetime(df["prepareTime"], unit="ms", errors="coerce")
    df["beginTime"] = pd.to_datetime(df["beginTime"], unit="ms", errors="coerce")
    df["endTime"] = pd.to_datetime(df["endTime"], unit="ms", errors="coerce")
    df["fetchedAt"] = pd.to_datetime(df["fetchedAt"], errors="coerce")

    # ----------------------------
    # IMPORTANT FIX:
    # Convert to ML order (OLD → NEW)
    # ----------------------------
    df = df.sort_values(by="fetchedAt", ascending=True).reset_index(drop=True)

    # ----------------------------
    # FEATURE ENGINEERING
    # ----------------------------
    df["round_duration"] = (df["endTime"] - df["beginTime"]).dt.total_seconds()
    df["prep_gap"] = (df["beginTime"] - df["prepareTime"]).dt.total_seconds()

    df["rolling_mean"] = df["crash"].rolling(10).mean()
    df["rolling_std"] = df["crash"].rolling(10).std()
    df["delta"] = df["crash"].diff()

    # ----------------------------
    # CLEAN FINAL DATASET
    # ----------------------------
    df = df.dropna(subset=FEATURES + ["crash"])

    return df
