import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

class LSTMModel:
    def __init__(self, input_shape):
        self.model = Sequential([
            LSTM(50, return_sequences=True, input_shape=input_shape),
            LSTM(50),
            Dense(1, activation="sigmoid")
        ])
        self.model.compile(optimizer="adam", loss="binary_crossentropy")

    def train(self, X, y, epochs=5):
        self.model.fit(X, y, epochs=epochs, verbose=0)

    def predict(self, X):
        return (self.model.predict(X) > 0.5).astype(int)
