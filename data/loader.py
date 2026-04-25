import json
import pandas as pd

def load_data(file=None):
    if file is not None:
        data = json.load(file)
        return pd.DataFrame(data)
    else:
        return pd.DataFrame()  # empty fallback
