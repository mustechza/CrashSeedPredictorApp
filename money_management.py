class Bankroll:
    def __init__(self, initial):
        self.balance = initial

    def win(self, amount):
        self.balance += amount

    def lose(self, amount):
        self.balance -= amount
