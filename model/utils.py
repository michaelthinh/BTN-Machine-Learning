from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from constant import coins
from constant import features


class DataScaler:
    def __init__(self, data: pd.DataFrame):
        scaler = MinMaxScaler(feature_range=(0, 1))
        self.scaler = scaler.fit(data)
        self.df_mean = data.mean()

    def fillMissingColumns(self, data: pd.DataFrame):
        copied_data = data.copy()
        missing_columns = set(self.scaler.feature_names_in_) - set(copied_data.columns)
        for column in missing_columns:
            copied_data[column] = self.df_mean[column]
        reordered_data = copied_data[self.scaler.feature_names_in_]
        return reordered_data

    def scale(self, data: pd.DataFrame):
        filled_data = self.fillMissingColumns(data)
        scaled_data = self.scaler.transform(filled_data)

        df_scaled = pd.DataFrame(scaled_data, columns=filled_data.columns)
        return df_scaled[data.columns]

    def inverseScale(self, data: pd.DataFrame):
        filled_data = self.fillMissingColumns(data)
        inverse_scaled_data = self.scaler.inverse_transform(filled_data)

        df_inverse_scaled = pd.DataFrame(
            inverse_scaled_data, columns=filled_data.columns
        )
        return df_inverse_scaled[data.columns]


class ROCCalculator:
    def fromClose(self, close):
        assert isinstance(close, pd.Series)
        return ((close - close.shift(1)) / close.shift(1)).fillna(0)


class CoinValidator:
    def __init__(self, validCoins=coins):
        self.validCoins = validCoins

    def isValidCoin(self, coin):
        return coin in self.validCoins

    def areValidCoins(self, coins):
        return all([self.isValidCoin(coin) for coin in coins])


class FeatureValidator:
    def __init__(self, validFeatures=features):
        self.validFeatures = validFeatures

    def isValidFeature(self, feature):
        return feature in self.validFeatures

    def areValidFeatures(self, features):
        return all([self.isValidFeature(feature) for feature in features])
