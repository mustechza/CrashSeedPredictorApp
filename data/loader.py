import json
import pandas as pd

def load_data(file):
    data = json.load(file)
    df = pd.DataFrame(data)

    # Normalize
    df["crash"] = df["rate"].astype(float)

    # Convert timestamps
    df["prepareTime"] = pd.to_datetime(df["prepareTime"], unit="ms", errors="coerce")
    df["beginTime"] = pd.to_datetime(df["beginTime"], unit="ms", errors="coerce")
    df["endTime"] = pd.to_datetime(df["endTime"], unit="ms", errors="coerce")
    df["fetchedAt"] = pd.to_datetime(df["fetchedAt"], errors="coerce")

    return df
