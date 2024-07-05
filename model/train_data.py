from model.utils import CoinValidator, FeatureValidator
import pandas as pd
from constant import candel_columns
import numpy as np
from model.utils import DataScaler, ROCCalculator


class TrainDataProvider:
    def __init__(self, coin, features, windowSize):
        self.windowSize = windowSize
        # check if coin is valid
        if not CoinValidator().isValidCoin(coin):
            raise ValueError(f"Invalid coin: {coin}")
        self.coin = coin
        # check if features are valid
        if not FeatureValidator().areValidFeatures(features):
            raise ValueError(f"Invalid features: {features}")
        self.features = features
        self.features.sort()
        self.rocCalculator = ROCCalculator()
        self.scaler = DataScaler(data=self.getDataFromFile())

    def getRawDataFromFile(self):
        data = pd.read_csv(f"./data/{self.coin}.csv")
        data["timestamp"] = pd.to_datetime(data["timestamp"])
        data.index = data["timestamp"]
        data = data.drop(["timestamp"], axis=1)
        data.sort_index(ascending=True, axis=0, inplace=True)
        return data

    def extractData(self, data: pd.DataFrame):
        features = list(set([*self.features, *candel_columns]))
        features.sort()
        return data[features]

    def getDataFromFile(self) -> pd.DataFrame:
        data = self.getRawDataFromFile()
        data["ROC"] = self.rocCalculator.fromClose(data["close"])
        extracted_data = self.extractData(data)
        return extracted_data

    def getXYData(self, data: pd.DataFrame):
        x_data = data[self.features].values
        y_data = data[candel_columns].values
        return x_data, y_data

    def scaleData(self, data: pd.DataFrame):
        scaled_data = self.scaler.scale(data)
        return scaled_data

    def getWindowedTrainData(self, x_data, y_data, windowSize):
        assert x_data.ndim == 2
        assert isinstance(x_data, np.ndarray)
        assert len(x_data) == len(y_data)

        num_features = x_data.shape[1]
        X = np.lib.stride_tricks.sliding_window_view(
            x_data[:-1, :], window_shape=(windowSize, num_features), axis=(0, 1)
        )
        X = X.reshape(-1, windowSize, num_features)
        Y = y_data[windowSize:, :]

        return X, Y

    def getTrainData(self):
        data = self.getDataFromFile()
        scaled_data = self.scaleData(data)
        x_data, y_data = self.getXYData(scaled_data)
        x_windowed_train, y_windowed_train = self.getWindowedTrainData(
            x_data, y_data, self.windowSize
        )

        return x_windowed_train, y_windowed_train
