import os

DATA_PATH = "data/crash_data.json"

TRAIN_SPLIT = 0.8

INITIAL_BANKROLL = 1000
BET_SIZE = 10

TARGET_MULTIPLIER = 1.5

MODEL_PARAMS = {
    "lstm": {"epochs": 5, "batch_size": 32},
    "rf": {"n_estimators": 100},
}
