algorithms = [
    {"label": "LSTM", "value": "LSTM"},
    {"label": "RNN", "value": "RNN"},
    {"label": "XGBoost", "value": "XGB"},
]

coin_labels = [
    {"label": "BTC-USD", "value": "btcusd"},
    {"label": "ETH-USD", "value": "ethusd"},
    {"label": "ADA-USD", "value": "adausd"},
]

day_number = [10, 20, 30, 60, 120]

timeframes = {
    "day": {
        "label": "1 ngày",
        "value": 86400,
    },
    "hour": {
        "label": "1 giờ",
        "value": 3600,
    },
    "minute": {
        "label": "1 phút",
        "value": 60,
    },
}

features = ["close", "ROC"]

windowSize = 50

models = ["LSTM", "RNN", "XGB"]

coins = [
    "btcusd",
    "ethusd",
    "adausd",
]

candel_columns = ["open", "high", "low", "close"]

lstm_units = 50

simple_rnn_units = 50