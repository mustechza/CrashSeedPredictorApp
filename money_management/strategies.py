def flat_bet(balance, base_bet):
    return base_bet

def martingale(balance, last_bet, loss):
    return last_bet * 2 if loss else last_bet
