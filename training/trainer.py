from sklearn.model_selection import train_test_split

FEATURES = [
    "rolling_mean",
    "rolling_std",
    "round_duration",
    "prep_gap",
    "delta"
]

def prepare_data(df):
    X = df[FEATURES]
    y = (df["crash"] >= 1.5).astype(int)

    return train_test_split(X, y, test_size=0.2, shuffle=False)
