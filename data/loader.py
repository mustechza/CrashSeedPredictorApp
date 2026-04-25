import json
import pandas as pd

def load_data(file):
    data = json.load(file)

    df = pd.DataFrame(data)

    # Normalize columns
    df["crash"] = df["rate"].astype(float)

    # Convert timestamps
    df["prepareTime"] = pd.to_datetime(df["prepareTime"], unit="ms")
    df["beginTime"] = pd.to_datetime(df["beginTime"], unit="ms")
    df["endTime"] = pd.to_datetime(df["endTime"], unit="ms")

    df["fetchedAt"] = pd.to_datetime(df["fetchedAt"])

    return df
