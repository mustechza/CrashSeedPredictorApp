def calculate_metrics(history):
    returns = history[-1] - history[0]
    max_drawdown = min(history) - history[0]

    return {
        "profit": returns,
        "max_drawdown": max_drawdown
    }
