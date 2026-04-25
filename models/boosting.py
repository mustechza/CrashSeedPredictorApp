from sklearn.ensemble import GradientBoostingClassifier

class BoostingModel:
    def __init__(self):
        self.model = GradientBoostingClassifier()

    def train(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)
