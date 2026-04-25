def prepare_data(df):
    features = [
        "rolling_mean",
        "rolling_std",
        "round_duration",
        "prep_gap",
        "delta"
    ]

    X = df[features]

    # Target: hit 1.5x
    y = (df["crash"] >= 1.5).astype(int)

    from sklearn.model_selection import train_test_split
    return train_test_split(X, y, test_size=0.2, shuffle=False)
