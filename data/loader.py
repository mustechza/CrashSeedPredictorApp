import json
import pandas as pd
from config import DATA_PATH

def load_data():
    with open(DATA_PATH, "r") as f:
        data = json.load(f)
    
    df = pd.DataFrame(data)
    return df
