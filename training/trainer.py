from sklearn.model_selection import train_test_split

def prepare_data(df):
    X = df[["rolling_mean", "rolling_std"]]
    y = (df["crash"] >= 1.5).astype(int)
    return train_test_split(X, y, test_size=0.2, shuffle=False)
