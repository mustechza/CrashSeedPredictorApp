from config import INITIAL_BANKROLL, BET_SIZE, TARGET_MULTIPLIER

def run_backtest(df, model):
    bankroll = INITIAL_BANKROLL
    history = []

    for i in range(len(df)-1):
        row = df.iloc[i]
        X = [[row["rolling_mean"], row["rolling_std"]]]
        
        pred = model.predict(X)[0]
        next_crash = df.iloc[i+1]["crash"]

        if pred == 1:
            if next_crash >= TARGET_MULTIPLIER:
                bankroll += BET_SIZE * (TARGET_MULTIPLIER - 1)
            else:
                bankroll -= BET_SIZE

        history.append(bankroll)

    return history
