import numpy as np
import random

class DQNAgent:
    def __init__(self):
        self.q_table = {}

    def get_state_key(self, state):
        return tuple(state)

    def act(self, state):
        key = self.get_state_key(state)
        if key not in self.q_table:
            self.q_table[key] = [0, 0]
        return np.argmax(self.q_table[key])

    def update(self, state, action, reward):
        key = self.get_state_key(state)
        self.q_table[key][action] += reward
